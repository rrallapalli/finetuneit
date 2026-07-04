# FineTuneIT Lite

A configurable LoRA/QLoRA fine-tuning platform for open LLMs, with live W&B
monitoring, a model registry, evaluation, and an inference playground — served
through a FastAPI backend and a Streamlit UI, designed to run on a RunPod GPU
pod in local mode.

Built on [Unsloth](https://github.com/unslothai/unsloth) for fast, memory-
efficient training, with Hugging Face Transformers, TRL, and PEFT underneath.

---

## What it does

| Tab | Purpose |
|-----|---------|
| **Train** | Configure and launch a LoRA/QLoRA job — model, dataset, template type, LoRA + training hyperparameters, W&B entity/project, HF adapter repo, optional merged/GGUF export. |
| **Live Metrics** | Chart a run's loss/metric history from W&B (auto-fills entity/project/run from the last job). |
| **Experiment Registry** | Rank W&B runs by a composite `model_score` and promote a champion. |
| **Evaluate** | Score an adapter on a dataset (ROUGE, BLEU, METEOR, BERTScore, SQuAD F1/EM, perplexity, latency, throughput, GPU). |
| **Inference Playground** | Load a base model + adapter and generate responses interactively. |

## Model types & templates (auto-detected)

The platform supports both **base** and **instruct** models, and picks the right
prompting automatically — you choose a model, it detects the rest:

- **Instruct / chat models** → uses the model's own chat template
  (`apply_chat_template`). No prompt-template choice needed. Data can be
  conversational (`messages`/`conversations`, OpenAI or ShareGPT form) or plain
  `instruction/input/output` (auto-wrapped into a chat).
- **Base / completion models** → uses an instruction-style prompt template
  (`alpaca`, `reasoning_alpaca`, or `instruction_only`), selectable in the UI.

Selecting a model shows a cue ("Chat / instruct model" vs "Base / completion
model") and reveals only the controls that apply. An **Advanced: override**
expander is available if detection is ever wrong. Detection is served by the
`/model/detect` endpoint (checks whether the tokenizer has a chat template),
with a name heuristic as an offline fallback.

## Execution modes

Set in `configs/platform_config.yaml` under `platform.execution_mode`:

- **`local`** (default, primary) — training/eval/inference run in-process on the
  pod's GPU inside the FastAPI backend. No RunPod serverless endpoints or RunPod
  API key needed.
- **`runpod`** — jobs dispatch to RunPod serverless endpoints (requires a RunPod
  API key and endpoint IDs in `platform_config.yaml`).

---

## Repository layout

```
app/
  api/            FastAPI routes: jobs, evaluation, inference, wandb, registry, model (detect)
  core/           platform config, secrets, json_utils (NaN-safe serialization)
  training/       train_lora.py, config, dataset_loader, prompt_templates
  evaluation/     evaluate_model.py + metrics (each metric guarded)
  inference/      serve_adapter.py (Unsloth load + model cache, template-aware)
  monitoring/     wandb_metrics.py (run summary + history)
  registry/       experiment_registry.py + scoring.py (champion ranking)
  ui/             streamlit_app.py (the whole UI)
configs/
  platform_config.yaml       execution mode, W&B + inference defaults, registry
  custom_jsonl.yaml          default training profile (edit to change a run)
  chat_conversational.yaml   example: instruct model + chat template
  non_reasoning_qwen25_7b.yaml, reasoning_qwen25_7b.yaml
  secrets/secrets.example.yaml
data/sample_alpaca.jsonl     small demo dataset
runpod_pod/                  setup + start scripts for a RunPod GPU pod (local mode)
runpod/                      serverless handlers (runpod mode)
smoke_test.py                ~10-step end-to-end training check
requirements.txt             pinned dependency stack
```

## Pinned stack

Dependencies are pinned to a GPU-verified combination: `transformers==5.12.1`,
`trl==1.7.0`, `peft==0.19.1`, `datasets==5.0.0`, `accelerate==1.14.0`,
`bitsandbytes==0.49.2`, `wandb==0.28.0`, with `unsloth==2026.6.9` installed
separately (`--no-deps`) so it doesn't disturb the pod's CUDA-matched PyTorch.
`torch` is intentionally **not** pinned — it's reused from the RunPod PyTorch
template. To move any version, bump the pin and run `python3.12 smoke_test.py`.

## API endpoints

Backend default `http://localhost:8000`:

- `POST /jobs/train` — start a training job (`{"config": {...}}`)
- `POST /evaluation/run` — evaluate an adapter
- `POST /inference/` — generate a response
- `GET  /model/detect?model_name=…` — is this a chat or base model?
- `GET  /wandb/run`, `GET /wandb/history` — W&B run summary + history
- `GET  /registry/runs` — ranked runs; `GET/POST /registry/champion`; `POST /registry/export`

## Getting started

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for the full RunPod local-mode setup, run
steps, the smoke test, and troubleshooting.

## License

Provided as-is. Base models and datasets carry their own licenses.
