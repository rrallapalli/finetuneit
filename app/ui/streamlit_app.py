import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml
from app.core.platform_config import load_platform_config
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="FineTuneIT Lite", layout="wide")

st.title("FineTuneIT Lite")
st.caption("Configurable fine-tuning, live monitoring, model registry, evaluation, and inference")

platform_config = load_platform_config()
platform_cfg = platform_config.get("platform", {})
wandb_cfg = platform_config.get("wandb", {})
ui_cfg = platform_config.get("ui", {})
registry_cfg = platform_config.get("registry", {})
inference_cfg = platform_config.get("inference", {})

API_BASE = st.sidebar.text_input("API Base URL", platform_cfg.get("api_base_url", "http://localhost:8000"))
st.sidebar.caption(f"Execution mode from platform config: `{platform_cfg.get('execution_mode', 'runpod')}`")
st.sidebar.caption("Edit configs/platform_config.yaml to change runtime settings.")
st.sidebar.caption("Secrets source: configs/secrets/secrets.yaml or environment fallback.")

CONFIG_DIR = Path("configs")
PROFILE_MAP = {
    "Custom JSONL — lightweight demo": CONFIG_DIR / "custom_jsonl.yaml",
    "Non-reasoning instruction tuning — Qwen2.5 7B": CONFIG_DIR / "non_reasoning_qwen25_7b.yaml",
    "Reasoning instruction tuning — Qwen2.5 7B": CONFIG_DIR / "reasoning_qwen25_7b.yaml",
}

HELP = {
    "dataset_sample_size": "Number of records used for training/evaluation. Keep this small for demo runs; increase for stronger adaptation.",
    "train_test_split": "Fraction of data held out for validation. Example: 0.2 means 80% train and 20% validation.",
    "prompt_template": "Controls how instruction, input, and response fields are formatted before training.",
    "base_model": "The pretrained model to adapt. Larger models usually perform better but need more GPU memory.",
    "max_seq_length": "Maximum token length per training example. Higher values support longer reasoning but use more memory.",
    "load_in_4bit": "Loads the model in 4-bit quantized mode to reduce GPU memory usage.",
    "batch_size": "Number of samples processed per device step. Larger values can train faster but require more GPU memory.",
    "gradient_accumulation": "Accumulates gradients across steps to simulate a larger batch size without using as much memory.",
    "max_steps": "Total optimizer steps. Higher values train longer but cost more and may overfit small datasets.",
    "learning_rate": "Controls how aggressively weights are updated. Too high can destabilize training; too low may underfit.",
    "warmup_steps": "Gradually increases learning rate early in training to improve stability.",
    "weight_decay": "Regularization that discourages overly large weights and can reduce overfitting.",
    "logging_steps": "How often training metrics are logged to W&B.",
    "eval_steps": "How often validation metrics are computed during training.",
    "save_steps": "How often model checkpoints are saved.",
    "optimizer": "Optimization algorithm. adamw_8bit is memory-efficient and commonly used for LoRA fine-tuning.",
    "scheduler": "Learning rate schedule over training. Linear is a safe default.",
    "seed": "Random seed for reproducibility.",
    "lora_r": "LoRA rank. Higher values add more trainable capacity but increase memory and training cost.",
    "lora_alpha": "Scaling factor for LoRA updates. Often set close to or 2x the LoRA rank.",
    "lora_dropout": "Dropout applied to LoRA layers. Can reduce overfitting, especially on small datasets.",
    "use_rslora": "Rank-stabilized LoRA. Can improve stability for some configurations.",
    "output_dir": "Local directory where adapters/checkpoints are saved.",
    "adapter_repo": "Hugging Face model repository where the LoRA adapter is pushed.",
}


# --- Model-style detection + cues (auto-detect chat vs base so users don't
# have to reason about template_type / prompt_template combinations) ---
@st.cache_data(show_spinner=False)
def detect_model_style(model_name, api_base):
    """Return {'template_type': 'chat'|'instruction', 'source': ...}. Asks the
    backend (tokenizer chat_template = ground truth); falls back to a name
    heuristic if the backend is unreachable."""
    if not model_name:
        return {"template_type": "instruction", "source": "default", "has_chat_template": None}
    try:
        r = requests.get(f"{api_base}/model/detect", params={"model_name": model_name}, timeout=(10, 120))
        r.raise_for_status()
        return r.json()
    except Exception:
        name = model_name.lower()
        looks_chat = any(h in name for h in ("instruct", "-chat", "chat-", "-it", "_it", "sft"))
        return {"template_type": "chat" if looks_chat else "instruction",
                "source": "name_heuristic", "has_chat_template": None}


def render_model_style_cue(info):
    """Show a short badge + one-line explanation of the detected model style."""
    if info["template_type"] == "chat":
        st.success("**Chat / instruct model** — will use the model's built-in chat template. "
                   "No prompt-template choice needed.")
    else:
        st.info("**Base / completion model** — will use an instruction-style prompt template (choose it below).")
    if info.get("source") == "name_heuristic":
        st.caption("Detected from the model name (couldn't load the tokenizer). Use Advanced to override if wrong.")



def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_post(url, payload, timeout=(10, 7200)):
    # (connect, read). Local-mode POST /jobs/train runs the whole job
    # synchronously, so the read timeout must be long; connect stays short.
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def safe_get(url, params):
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def update_inference_from_model(model_record):
    if not model_record:
        return
    if model_record.get("base_model"):
        st.session_state["infer_base_model"] = model_record["base_model"]
    if model_record.get("adapter_repo"):
        st.session_state["infer_adapter_repo"] = model_record["adapter_repo"]
    if model_record.get("prompt_template"):
        st.session_state["infer_prompt_template"] = model_record["prompt_template"]


tab_train, tab_live, tab_registry, tab_eval, tab_infer = st.tabs([
    "Train",
    "Live Metrics",
    "Experiment Registry",
    "Evaluate",
    "Inference Playground",
])

with tab_train:
    st.header("Configure Fine-Tuning Experiment")

    profile_name = st.selectbox("Experiment Profile", list(PROFILE_MAP.keys()))
    config = load_yaml(PROFILE_MAP[profile_name])

    # Model is the primary choice — select it first (full width) so the template
    # cue below reflects the current selection immediately (no one-run lag).
    st.subheader("Model")
    model_options = [
        "unsloth/Qwen2.5-0.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-0.5B",
        "unsloth/Qwen2.5-1.5B",
        "unsloth/Qwen2.5-3B",
        "unsloth/Qwen2.5-7B",
    ]
    default_model = config["model"].get("base_model", "unsloth/Qwen2.5-0.5B-Instruct")
    if default_model not in model_options:
        model_options.insert(0, default_model)
    config["model"]["base_model"] = st.selectbox(
        "Base Model",
        model_options,
        index=model_options.index(default_model),
        help=HELP["base_model"] + " Chat/instruct vs base is detected automatically.",
        key="train_base_model",
    )
    _style = detect_model_style(config["model"]["base_model"], API_BASE)
    config["template_type"] = _style["template_type"]
    render_model_style_cue(_style)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset")
        config["experiment_name"] = st.text_input("Experiment Name", config["experiment_name"])
        config["dataset_mode"] = st.selectbox(
            "Dataset Mode",
            ["local_jsonl", "hf_dataset"],
            index=["local_jsonl", "hf_dataset"].index(config.get("dataset_mode", "local_jsonl")),
            help="Choose a local JSONL file or a Hugging Face dataset."
        )

        if config["dataset_mode"] == "hf_dataset":
            config["dataset_id"] = st.text_input("Hugging Face Dataset ID", config.get("dataset_id", "tatsu-lab/alpaca"))
            config["dataset_config"] = st.text_input("Dataset Config / Subset", config.get("dataset_config") or "")
            config["dataset_split"] = st.text_input("Dataset Split", config.get("dataset_split", "train"))
        else:
            config["dataset_path"] = st.text_input("Local JSONL Dataset Path", config.get("dataset_path", "data/sample_alpaca.jsonl"))

        config["dataset_sample_size"] = st.number_input(
            "Dataset Sample Size",
            min_value=3,
            value=int(config.get("dataset_sample_size", 25)),
            help=HELP["dataset_sample_size"],
        )
        config["train_test_split"] = st.slider(
            "Validation Split",
            0.05,
            0.5,
            float(config.get("train_test_split", 0.2)),
            0.05,
            help=HELP["train_test_split"],
        )
        # Model style already detected above; render only the fields it gates.
        if config["template_type"] == "instruction":
            config["prompt_template"] = st.selectbox(
                "Prompt Template",
                ["alpaca", "reasoning_alpaca", "instruction_only"],
                index=["alpaca", "reasoning_alpaca", "instruction_only"].index(config.get("prompt_template", "alpaca")),
                help=HELP["prompt_template"],
            )
        else:
            config["prompt_template"] = config.get("prompt_template", "alpaca")
            config["messages_column"] = st.text_input(
                "Conversations Column (optional)",
                config.get("messages_column", "") or "",
                help="Column with role/content turns (e.g. 'messages' or 'conversations'). "
                     "Leave blank to auto-wrap instruction/input/output into a chat.",
            ) or None

        with st.expander("Advanced: override template type"):
            config["template_type"] = st.radio(
                "Template type",
                ["instruction", "chat"],
                index=["instruction", "chat"].index(config["template_type"]),
                help="Auto-set from the model. Override only if detection is wrong.",
                horizontal=True,
            )

    with col2:
        st.subheader("Model Settings")
        config["model"]["max_seq_length"] = st.selectbox(
            "Max Sequence Length",
            [512, 1024, 2048, 4096],
            index=[512, 1024, 2048, 4096].index(int(config["model"].get("max_seq_length", 1024))),
            help=HELP["max_seq_length"],
        )
        config["model"]["load_in_4bit"] = st.checkbox(
            "Load in 4-bit",
            value=bool(config["model"].get("load_in_4bit", True)),
            help=HELP["load_in_4bit"],
        )

    st.subheader("Training Hyperparameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        config["training"]["per_device_train_batch_size"] = st.number_input(
            "Batch Size", 1, 16, int(config["training"].get("per_device_train_batch_size", 1)),
            help=HELP["batch_size"],
        )
        config["training"]["gradient_accumulation_steps"] = st.number_input(
            "Gradient Accumulation", 1, 32, int(config["training"].get("gradient_accumulation_steps", 4)),
            help=HELP["gradient_accumulation"],
        )
        config["training"]["max_steps"] = st.number_input(
            "Max Steps", 1, 5000, int(config["training"].get("max_steps", 25)),
            help=HELP["max_steps"],
        )
    with c2:
        config["training"]["learning_rate"] = st.number_input(
            "Learning Rate", value=float(config["training"].get("learning_rate", 0.0002)), format="%.6f",
            help=HELP["learning_rate"],
        )
        config["training"]["warmup_steps"] = st.number_input(
            "Warmup Steps", 0, 1000, int(config["training"].get("warmup_steps", 5)),
            help=HELP["warmup_steps"],
        )
        config["training"]["weight_decay"] = st.number_input(
            "Weight Decay", value=float(config["training"].get("weight_decay", 0.01)), format="%.4f",
            help=HELP["weight_decay"],
        )
    with c3:
        config["training"]["logging_steps"] = st.number_input(
            "Logging Steps", 1, 1000, int(config["training"].get("logging_steps", 5)),
            help=HELP["logging_steps"],
        )
        config["training"]["eval_steps"] = st.number_input(
            "Eval Steps", 1, 1000, int(config["training"].get("eval_steps", 10)),
            help=HELP["eval_steps"],
        )
        config["training"]["save_steps"] = st.number_input(
            "Save Steps", 1, 1000, int(config["training"].get("save_steps", 10)),
            help=HELP["save_steps"],
        )
    with c4:
        config["training"]["optim"] = st.selectbox(
            "Optimizer", ["adamw_8bit", "adamw_torch"], index=0,
            help=HELP["optimizer"],
        )
        config["training"]["lr_scheduler_type"] = st.selectbox(
            "LR Scheduler", ["linear", "cosine", "constant"], index=0,
            help=HELP["scheduler"],
        )
        config["training"]["seed"] = st.number_input(
            "Seed", 1, 999999, int(config["training"].get("seed", 3407)),
            help=HELP["seed"],
        )

    st.subheader("LoRA Hyperparameters")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        config["lora"]["r"] = st.selectbox(
            "LoRA Rank r", [4, 8, 16, 32, 64, 128],
            index=[4, 8, 16, 32, 64, 128].index(int(config["lora"].get("r", 16))),
            help=HELP["lora_r"],
        )
    with l2:
        config["lora"]["lora_alpha"] = st.selectbox(
            "LoRA Alpha", [8, 16, 32, 64, 128],
            index=[8, 16, 32, 64, 128].index(int(config["lora"].get("lora_alpha", 16))),
            help=HELP["lora_alpha"],
        )
    with l3:
        config["lora"]["lora_dropout"] = st.slider(
            "LoRA Dropout", 0.0, 0.3, float(config["lora"].get("lora_dropout", 0.0)), step=0.01,
            help=HELP["lora_dropout"],
        )
    with l4:
        config["lora"]["use_rslora"] = st.checkbox(
            "Use rsLoRA", value=bool(config["lora"].get("use_rslora", False)),
            help=HELP["use_rslora"],
        )

    st.subheader("Experiment Tracking")
    config["wandb_entity"] = st.text_input(
        "W&B Entity", config.get("wandb_entity", wandb_cfg.get("entity", "")),
        help="Weights & Biases team or username the run logs under. Leave blank to use your default entity.",
    )
    config["wandb_project"] = st.text_input(
        "W&B Project", config.get("wandb_project", wandb_cfg.get("project", "finetuneit-lite")),
        help="Weights & Biases project the run logs to.",
    )

    st.subheader("Output")
    config["output"]["output_dir"] = st.text_input(
        "Output Directory", config["output"].get("output_dir", "outputs/adapter"),
        help=HELP["output_dir"],
    )
    config["output"]["output_adapter_repo"] = st.text_input(
        "Hugging Face Adapter Repo", config["output"].get("output_adapter_repo", "your-hf-username/adapter"),
        help=HELP["adapter_repo"],
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        config["output"]["export_merged"] = st.checkbox(
            "Export merged 16-bit", value=bool(config["output"].get("export_merged", False)),
            help="Also push a merged standalone model (adapter fused into the base; no PEFT needed) to <repo>-merged.",
        )
    with e2:
        config["output"]["export_gguf"] = st.checkbox(
            "Export GGUF", value=bool(config["output"].get("export_gguf", False)),
            help="Also push a GGUF build (llama.cpp / Ollama / LM Studio) to <repo>-gguf. Heavier; needs an -Instruct base for chat use.",
        )
    with e3:
        config["output"]["gguf_quantization"] = st.text_input(
            "GGUF quantization", config["output"].get("gguf_quantization", "q4_k_m"),
        )

    with st.expander("Preview full training config"):
        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

    if st.button("Start Training"):
        result = safe_post(f"{API_BASE}/jobs/train", {"config": config})
        st.session_state["last_train_result"] = result
        st.json(result)

        if result.get("wandb_run_url"):
            st.markdown(f"[Open W&B Run]({result['wandb_run_url']})")
        if result.get("wandb_run_id"):
            st.info(f"W&B Run ID: `{result['wandb_run_id']}`")

with tab_live:
    st.header("Live W&B Training Metrics")

    last_result = st.session_state.get("last_train_result", {})
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wandb_entity = st.text_input("W&B Entity", last_result.get("wandb_entity") or wandb_cfg.get("entity", ""), key="live_entity")
    with col_b:
        wandb_project = st.text_input("W&B Project", last_result.get("wandb_project") or wandb_cfg.get("project", "finetuneit-lite"), key="live_project")
    with col_c:
        wandb_run_id = st.text_input("W&B Run ID", last_result.get("wandb_run_id", ""), key="live_run_id")

    refresh = st.checkbox("Auto-refresh metrics", value=False)
    refresh_seconds = st.slider("Refresh interval seconds", 5, 60, int(ui_cfg.get("refresh_seconds", 10)))

    if refresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="wandb_live_refresh")

    if st.button("Fetch Metrics") or refresh:
        if not wandb_entity or not wandb_project or not wandb_run_id:
            st.warning("Enter W&B entity, project, and run ID.")
        else:
            run_info = safe_get(f"{API_BASE}/wandb/run", {
                "entity": wandb_entity,
                "project": wandb_project,
                "run_id": wandb_run_id,
            })
            history = safe_get(f"{API_BASE}/wandb/history", {
                "entity": wandb_entity,
                "project": wandb_project,
                "run_id": wandb_run_id,
                "samples": wandb_cfg.get("default_samples", 1000),
            })

            if run_info.get("status") == "error":
                st.error(run_info["message"])
            else:
                st.markdown(f"**Run:** [{run_info.get('name', wandb_run_id)}]({run_info.get('url', '#')})")
                st.metric("Run State", run_info.get("state", "unknown"))

            records = history.get("records", [])
            if not records:
                st.info("No W&B history records found yet.")
            else:
                df = pd.DataFrame(records)
                latest = df.tail(1).to_dict(orient="records")[0]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest Step", latest.get("_step", "N/A"))
                m2.metric("Train Loss", latest.get("train/loss", "N/A"))
                m3.metric("Eval Loss", latest.get("eval/loss", "N/A"))
                m4.metric("Learning Rate", latest.get("train/learning_rate", "N/A"))

                chart_cols = [c for c in ["train/loss", "eval/loss", "train/learning_rate", "train/grad_norm"] if c in df.columns]
                if chart_cols:
                    st.subheader("Metric Curves")
                    st.line_chart(df[chart_cols])

                st.subheader("Recent Logs")
                st.dataframe(df.tail(20), use_container_width=True)

with tab_registry:
    st.header("Experiment Registry & Model Ranking")
    st.caption("Compare previous W&B runs, rank candidate adapters, and promote a champion model for inference.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        registry_entity = st.text_input("W&B Entity", wandb_cfg.get("entity", ""), key="registry_entity")
    with col_b:
        registry_project = st.text_input("W&B Project", wandb_cfg.get("project", "finetuneit-lite"), key="registry_project")
    with col_c:
        max_runs = st.number_input("Max Runs", 5, 200, int(ui_cfg.get("default_max_runs", 50)))

    if st.button("Load Experiment Registry"):
        if not registry_entity or not registry_project:
            st.warning("Enter W&B entity and project.")
        else:
            registry_response = safe_get(
                f"{API_BASE}/registry/runs",
                {"entity": registry_entity, "project": registry_project, "max_runs": max_runs},
            )
            st.session_state["registry_runs"] = registry_response.get("runs", [])

    runs = st.session_state.get("registry_runs", [])
    if runs:
        df = pd.DataFrame(runs)
        sort_col = "model_score" if "model_score" in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)

        st.subheader("Ranked Runs")
        display_cols = [
            "model_score", "run_name", "state", "base_model", "dataset_type",
            "prompt_template", "adapter_repo", "eval/loss", "ROUGE-L",
            "BERTScore (F1)", "SQuAD F1", "Average Latency (sec)", "url"
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        st.subheader("Compare Selected Runs")
        run_options = {
            f"{row.get('run_name') or row.get('run_id')} | score={row.get('model_score')}": row
            for row in runs
        }

        selected_labels = st.multiselect(
            "Select runs to compare",
            list(run_options.keys()),
            default=list(run_options.keys())[: min(3, len(run_options))]
        )

        if selected_labels:
            compare_df = pd.DataFrame([run_options[label] for label in selected_labels])
            metric_cols = [c for c in ["model_score", "eval/loss", "ROUGE-L", "BERTScore (F1)", "SQuAD F1", "Average Latency (sec)"] if c in compare_df.columns]
            st.bar_chart(compare_df.set_index("run_name")[metric_cols])

        st.subheader("Promote Champion for Inference")
        champion_label = st.selectbox("Choose champion run", list(run_options.keys()))
        champion_record = run_options[champion_label]

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Promote Champion"):
                result = safe_post(f"{API_BASE}/registry/champion", {"name": registry_cfg.get("champion_model_name", "champion_model"), "model": champion_record})
                st.session_state["champion_model"] = champion_record
                st.json(result)
        with c2:
            if st.button("Use Champion in Inference"):
                update_inference_from_model(champion_record)
                st.success("Inference fields updated from selected champion.")
        with c3:
            if st.button("Export Registry CSV"):
                result = safe_post(f"{API_BASE}/registry/export", {"rows": runs, "path": registry_cfg.get("export_path", "registry/experiment_registry.csv")})
                st.json(result)

    if st.button("Load Saved Champion"):
        champion = safe_get(f"{API_BASE}/registry/champion", {"name": registry_cfg.get("champion_model_name", "champion_model")})
        if champion:
            st.session_state["champion_model"] = champion
            st.json(champion)
        else:
            st.info("No saved champion model found.")

with tab_eval:
    st.header("Launch Evaluation Job")

    eval_job_id = st.text_input("Evaluation Job ID", "eval-demo-001")
    eval_base_model = st.text_input("Evaluation Base Model", "unsloth/Qwen2.5-0.5B-Instruct")
    adapter_repo = st.text_input("Adapter Repo", "your-hf-username/finetuneit-demo-adapter")
    eval_dataset = st.text_input("Evaluation Dataset Path", "data/sample_alpaca.jsonl")
    _eval_style = detect_model_style(eval_base_model, API_BASE)
    render_model_style_cue(_eval_style)
    eval_template_type = _eval_style["template_type"]
    if eval_template_type == "instruction":
        prompt_template = st.selectbox("Evaluation Prompt Template", ["alpaca", "reasoning_alpaca", "instruction_only"])
    else:
        prompt_template = "alpaca"
    with st.expander("Advanced: override template type"):
        eval_template_type = st.radio(
            "Template type", ["instruction", "chat"],
            index=["instruction", "chat"].index(eval_template_type),
            horizontal=True, key="eval_tt_override",
        )
    num_samples = st.number_input("Number of Eval Samples", min_value=1, value=5)

    if st.button("Start Evaluation"):
        payload = {
            "job_id": eval_job_id,
            "base_model": eval_base_model,
            "adapter_repo": adapter_repo,
            "dataset": eval_dataset,
            "num_samples": num_samples,
            "prompt_template": prompt_template,
            "template_type": eval_template_type,
            "output_path": "outputs/evaluation_metrics.csv",
        }
        result = safe_post(f"{API_BASE}/evaluation/run", payload)
        st.json(result)
        if result.get("metrics"):
            st.subheader("Evaluation Metrics")
            st.dataframe(pd.DataFrame([result["metrics"]]), use_container_width=True)

with tab_infer:
    st.header("Inference Playground")

    champion = st.session_state.get("champion_model", {})
    if champion:
        st.info(f"Champion selected: {champion.get('run_name') or champion.get('run_id')} | score={champion.get('model_score')}")

    infer_base_model = st.text_input(
        "Inference Base Model",
        st.session_state.get("infer_base_model", champion.get("base_model", inference_cfg.get("default_base_model", "unsloth/Qwen2.5-0.5B-Instruct"))),
        key="infer_base_model",
    )
    infer_adapter_repo = st.text_input(
        "Inference Adapter Repo",
        st.session_state.get("infer_adapter_repo", champion.get("adapter_repo", inference_cfg.get("default_adapter_repo", "your-hf-username/finetuneit-demo-adapter"))),
        key="infer_adapter_repo",
    )

    _infer_style = detect_model_style(infer_base_model, API_BASE)
    render_model_style_cue(_infer_style)
    infer_template_type = _infer_style["template_type"]

    template_options = ["alpaca", "reasoning_alpaca", "instruction_only"]
    default_template = st.session_state.get("infer_prompt_template", champion.get("prompt_template", inference_cfg.get("default_prompt_template", "alpaca")))
    if default_template not in template_options:
        default_template = "alpaca"

    if infer_template_type == "instruction":
        infer_prompt_template = st.selectbox(
            "Inference Prompt Template",
            template_options,
            index=template_options.index(default_template),
            key="infer_prompt_template",
        )
    else:
        infer_prompt_template = "alpaca"
    with st.expander("Advanced: override template type"):
        infer_template_type = st.radio(
            "Template type", ["instruction", "chat"],
            index=["instruction", "chat"].index(infer_template_type),
            horizontal=True, key="infer_tt_override",
        )
    prompt = st.text_area("Prompt", "Explain credit risk in simple terms.")
    input_text = st.text_area("Optional Input Context", "")
    max_new_tokens = st.slider("Max New Tokens", 32, 1024, int(inference_cfg.get("default_max_new_tokens", 128)))

    if st.button("Generate Response"):
        payload = {
            "base_model": infer_base_model,
            "adapter_repo": infer_adapter_repo,
            "prompt": prompt,
            "input_text": input_text,
            "prompt_template": infer_prompt_template,
            "template_type": infer_template_type,
            "max_new_tokens": max_new_tokens,
        }
        result = safe_post(f"{API_BASE}/inference/", payload)

        if "response" in result:
            st.subheader("Generated Response")
            st.write(result["response"])
        else:
            st.json(result)
