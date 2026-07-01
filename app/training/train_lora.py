import os
import wandb
from huggingface_hub import HfApi

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

from app.training.config import load_training_config, merge_config_with_overrides
from app.training.dataset_loader import load_and_prepare_dataset
from app.core.platform_config import get_wandb_project, get_wandb_entity
from app.core.secrets import apply_secrets_to_environment, get_hf_token, get_wandb_api_key


def _export_artifacts(model, tokenizer, output_config, adapter_repo, output_dir):
    """Opt-in merged-16bit / GGUF export. Best-effort: each is isolated so it can
    never fail the training run. Uses Unsloth's helpers (version-pinned in
    setup_pod_env.sh). Only pushes to the Hub when a real adapter_repo is set."""
    info = {}
    hf_token = get_hf_token()
    hub_ok = bool(adapter_repo) and not adapter_repo.startswith("your-hf-username")

    if output_config.get("export_merged"):
        try:
            merged_dir = os.path.join(output_dir, "merged_16bit")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            info["merged_dir"] = merged_dir
            if hub_ok:
                merged_repo = output_config.get("merged_repo") or f"{adapter_repo}-merged"
                model.push_to_hub_merged(merged_repo, tokenizer, save_method="merged_16bit", token=hf_token)
                info["merged_repo"] = merged_repo
        except Exception as exc:
            info["merged_error"] = f"{type(exc).__name__}: {exc}"

    if output_config.get("export_gguf"):
        try:
            quant = output_config.get("gguf_quantization", "q4_k_m")
            gguf_dir = os.path.join(output_dir, "gguf")
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
            info["gguf_dir"] = gguf_dir
            if hub_ok:
                gguf_repo = output_config.get("gguf_repo") or f"{adapter_repo}-gguf"
                model.push_to_hub_gguf(gguf_repo, tokenizer, quantization_method=quant, token=hf_token)
                info["gguf_repo"] = gguf_repo
        except Exception as exc:
            info["gguf_error"] = f"{type(exc).__name__}: {exc}"

    return info


def run_training_from_config(config: dict) -> dict:
    apply_secrets_to_environment()
    experiment_name = config.get("experiment_name", "finetuneit-experiment")
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    output_config = config["output"]

    # Weights & Biases is optional. If no API key is present, disable it so
    # wandb.init() never drops into an interactive login prompt (which hangs a
    # non-interactive backend). This is opt-in by presence of a key.
    use_wandb = bool(get_wandb_api_key())
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb_project = config.get("wandb_project", get_wandb_project())
    wandb_entity = config.get("wandb_entity") or get_wandb_entity() or None

    output_dir = output_config.get("output_dir", f"outputs/{experiment_name}")
    max_seq_length = int(model_config.get("max_seq_length", 1024))

    run = None
    try:
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=experiment_name,
            config=config,
            mode="online" if use_wandb else "disabled",
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["base_model"],
            max_seq_length=max_seq_length,
            dtype=model_config.get("dtype"),
            load_in_4bit=model_config.get("load_in_4bit", True),
            token=get_hf_token(),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_config.get("r", 16)),
            target_modules=lora_config.get("target_modules"),
            lora_alpha=int(lora_config.get("lora_alpha", 16)),
            lora_dropout=float(lora_config.get("lora_dropout", 0.0)),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
            random_state=int(lora_config.get("random_state", 3407)),
            use_rslora=bool(lora_config.get("use_rslora", False)),
            loftq_config=None,
        )

        split_dataset = load_and_prepare_dataset(config, tokenizer)
        eval_dataset = split_dataset.get("test")
        eval_enabled = eval_dataset is not None and len(eval_dataset) > 0
        eval_strategy = training_config.get("eval_strategy", "steps") if eval_enabled else "no"

        # Modern TRL: all SFT-specific fields (max_length, dataset_text_field,
        # packing, dataset_num_proc) live on SFTConfig, not on SFTTrainer.
        # SFTConfig subclasses TrainingArguments, so it also takes every
        # standard training arg below.
        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            warmup_steps=int(training_config.get("warmup_steps", 20)),
            max_steps=int(training_config.get("max_steps", 250)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(training_config.get("logging_steps", 10)),
            optim=training_config.get("optim", "adamw_8bit"),
            weight_decay=float(training_config.get("weight_decay", 0.1)),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
            seed=int(training_config.get("seed", 3407)),
            report_to=["wandb"] if use_wandb else "none",
            save_strategy=training_config.get("save_strategy", "steps"),
            save_steps=int(training_config.get("save_steps", 50)),
            save_total_limit=int(training_config.get("save_total_limit", 2)),
            per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 2)),
            eval_accumulation_steps=int(training_config.get("eval_accumulation_steps", 4)),
            eval_strategy=eval_strategy,
            eval_steps=int(training_config.get("eval_steps", 50)) if eval_enabled else None,
            load_best_model_at_end=bool(training_config.get("load_best_model_at_end", True)) and eval_enabled,
            metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
            # SFT-specific
            max_length=max_seq_length,
            dataset_text_field="text",
            dataset_num_proc=int(training_config.get("dataset_num_proc", 2)),
            packing=bool(training_config.get("packing", False)),
            # Newer TRL (>=1.x) defaults SFTTrainer to padding_free=True, which
            # refuses a set max_length unless packing is on. Disable it so
            # max_length truncates normally with packing=False (older behavior).
            padding_free=bool(training_config.get("padding_free", False)),
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=split_dataset["train"],
            eval_dataset=eval_dataset if eval_enabled else None,
            args=args,
        )

        trainer_stats = trainer.train()

        best_model = trainer.model
        best_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        adapter_repo = output_config.get("output_adapter_repo")
        if adapter_repo and not adapter_repo.startswith("your-hf-username"):
            api = HfApi(token=get_hf_token())
            api.create_repo(adapter_repo, exist_ok=True)
            api.upload_folder(repo_id=adapter_repo, folder_path=output_dir, repo_type="model")

        # Opt-in merged / GGUF exports (config-gated, best-effort).
        exports = _export_artifacts(best_model, tokenizer, output_config, adapter_repo, output_dir)

        result = {
            "status": "completed",
            "experiment_name": experiment_name,
            "base_model": model_config["base_model"],
            "dataset_type": config.get("dataset_type"),
            "prompt_template": config.get("prompt_template"),
            "output_dir": output_dir,
            "adapter_repo": adapter_repo,
            "exports": exports,
            "wandb_entity": wandb_entity,
            "wandb_project": wandb_project,
            "wandb_run_id": run.id if run else None,
            "wandb_run_name": run.name if run else None,
            "wandb_run_url": run.url if run else None,
            "train_metrics": trainer_stats.metrics,
        }
        return result

    except Exception as exc:
        return {
            "status": "error",
            "experiment_name": experiment_name,
            "error": f"{type(exc).__name__}: {exc}",
            "output_dir": output_dir,
            "wandb_run_id": run.id if run else None,
            "wandb_run_url": run.url if run else None,
        }
    finally:
        wandb.finish()


def run_training(config_path: str | None = None, overrides: dict | None = None) -> dict:
    if config_path:
        config = load_training_config(config_path)
    else:
        config = overrides or {}

    config = merge_config_with_overrides(config, overrides)
    return run_training_from_config(config)
