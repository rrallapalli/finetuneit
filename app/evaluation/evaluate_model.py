import os
import time
from pathlib import Path

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.evaluation.metrics import compute_text_generation_metrics
from app.core.secrets import apply_secrets_to_environment, get_hf_token, get_wandb_api_key
from app.training.prompt_templates import build_inference_prompt


def evaluate_model(
    base_model: str,
    adapter_repo: str | None,
    dataset_path: str,
    num_samples: int = 25,
    output_path: str = "outputs/evaluation_metrics.csv",
    wandb_project: str = "finetuneit-lite",
    job_id: str = "eval-job",
    prompt_template: str = "alpaca",
    max_new_tokens: int = 256,
    load_in_4bit: bool = True,
) -> dict:
    apply_secrets_to_environment()

    # W&B optional: disable when no key so it can't hang on interactive login.
    use_wandb = bool(get_wandb_api_key())
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    run = None
    try:
        run = wandb.init(
            project=wandb_project,
            job_type="evaluation",
            name=job_id,
            mode="online" if use_wandb else "disabled",
            config={
                "base_model": base_model,
                "adapter_repo": adapter_repo,
                "dataset_path": dataset_path,
                "num_samples": num_samples,
                "prompt_template": prompt_template,
                "max_new_tokens": max_new_tokens,
            },
        )

        token = get_hf_token()
        # Load through Unsloth (matching training/inference) so its patched
        # attention is wired up; loading via plain transformers while unsloth is
        # imported yields models missing apply_qkv etc. If an adapter repo is
        # given, load it directly (its config references the base model).
        model_name = adapter_repo or base_model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=load_in_4bit,
            token=token,
        )
        model.eval()

        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        predictions, references, latencies, throughputs, perplexities, gpu_mem_usages = [], [], [], [], [], []

        for example in tqdm(dataset, desc="Evaluating model"):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            reference = example.get("output", "")

            prompt = build_inference_prompt(instruction, input_text, prompt_template)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = time.time() - start_time
            output_len = outputs.shape[-1] - input_len

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded.split("### Response:")[-1].strip()

            full_text = prompt + "\n" + reference
            ppl_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                ppl_outputs = model(**ppl_inputs, labels=ppl_inputs["input_ids"])

            predictions.append(response)
            references.append(reference)
            latencies.append(latency)
            throughputs.append(output_len / max(latency, 1e-6))
            perplexities.append(float(torch.exp(ppl_outputs.loss).item()))

            if torch.cuda.is_available():
                gpu_mem_usages.append(float(torch.cuda.max_memory_allocated() / (1024 * 1024)))

        metrics = compute_text_generation_metrics(
            predictions=predictions,
            references=references,
            perplexities=perplexities,
            latencies=latencies,
            throughputs=throughputs,
            gpu_mem_usages=gpu_mem_usages,
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(output_file, index=False)

        if use_wandb:
            wandb.log(metrics)
            wandb.save(str(output_file))

        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": metrics,
            "metrics_path": str(output_file),
            "wandb_run_url": run.url if run else None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "job_id": job_id,
            "error": f"{type(exc).__name__}: {exc}",
            "wandb_run_url": run.url if run else None,
        }
    finally:
        wandb.finish()
