# How to Run — RunPod Local Mode

Run the whole platform on a single RunPod GPU pod, with training / evaluation /
inference executing in-process on the pod's GPU (`execution_mode: local`).

---

## 1. Create the pod

- Use a **RunPod PyTorch template** (a CUDA-matched PyTorch is preinstalled and
  reused — setup does not reinstall torch).
- **Expose HTTP port 8501** (the Streamlit UI). Port 8000 (backend) stays
  internal; the UI reaches it over `localhost`.
- A small GPU handles the 0.5B demo (e.g. RTX A4000/A4500/A5000).

## 2. Pre-flight check

Setup uses `python3.12`; the reused torch must live under that Python:

```bash
python3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect a version and `True`. If `ModuleNotFoundError`, use a Python-3.12 template
or edit `PYTHON=python3.12` in `runpod_pod/setup_pod_env.sh`.

## 3. Clone and set up

```bash
cd /workspace
git clone https://github.com/rrallapalli/finetuneit.git
cd finetuneit
bash runpod_pod/setup_pod_env.sh
```

Setup probes CUDA (fails loudly if unavailable), pre-overlays a couple of
apt-owned packages so pip can't collide, installs the pinned requirements into
the system Python (reusing the pod's torch), installs Unsloth `--no-deps`, and
verifies the stack imports.

## 4. Configure secrets

```bash
cp configs/secrets/secrets.example.yaml configs/secrets/secrets.yaml
nano configs/secrets/secrets.yaml
```

```yaml
wandb:
  api_key: <your-wandb-api-key>     # NOTE: a space is required after the colon
huggingface:
  token: <your-hf-token>            # only for push-to-hub / gated models
```

W&B auto-disables if no key is present (so it can't hang on interactive login).
No RunPod API key is needed in local mode. `secrets.yaml` is gitignored.

Confirm mode and that the W&B key parses:

```bash
grep execution_mode configs/platform_config.yaml   # -> local
python3.12 -c "from app.core.secrets import get_wandb_api_key; print('wandb key:', bool(get_wandb_api_key()))"
```

## 5. Smoke test (recommended)

```bash
python3.12 smoke_test.py     # expect: [PASS] training completed
```

Run after any dependency bump or code change before trusting a real job.

## 6. Start backend + UI

Two separate terminals:

```bash
bash runpod_pod/start_api.sh          # FastAPI backend on :8000
```

```bash
bash runpod_pod/start_streamlit.sh    # Streamlit UI on :8501
```

## 7. Open the UI

Pod's proxy URL for port 8501 (from the pod's **Connect** panel):

```
https://<your-pod-id>-8501.proxy.runpod.net
```

Pick a model — the tab detects chat vs base and shows only the relevant
controls — then submit. In local mode the job runs on the pod's GPU inside the
backend; the loss curve streams to W&B. Then use Live Metrics, Experiment
Registry, Evaluate, and Inference Playground.

## Choosing a model & template (auto-detected)

- **Instruct model** (e.g. `unsloth/Qwen2.5-0.5B-Instruct`) → detected as chat;
  uses the model's chat template. Best default; always responds.
- **Base model** (e.g. `unsloth/Qwen2.5-0.5B`) → detected as base; pick an
  instruction prompt template (`alpaca` / `reasoning_alpaca` / `instruction_only`).
- Use **Advanced: override template type** only if detection is wrong.

## Train without the UI

```bash
bash runpod_pod/run_training.sh configs/custom_jsonl.yaml
# or the chat example:
bash runpod_pod/run_training.sh configs/chat_conversational.yaml
```

## Configuring a run

Edit `configs/custom_jsonl.yaml` (or `chat_conversational.yaml`):

- `model.base_model` — any HF model; chat vs base is auto-detected.
- `template_type` — `instruction` (base) or `chat` (instruct). `messages_column`
  for conversational data under `chat`.
- `dataset_mode` — `local_jsonl` (a path) or `hf_dataset` (a Hub id). Local JSONL
  uses `instruction` / `input` / `output` fields.
- `training.max_steps` (or high + `num_train_epochs`), plus LoRA/training params.
- `output.output_adapter_repo` — real HF repo to push to (leave the
  `your-hf-username/...` placeholder to skip pushing).
- `output.export_merged` / `output.export_gguf` — opt-in standalone/GGUF exports
  (best on an instruct base; require a chat template for chat use).

---

## Updating an already-set-up pod

Code-only changes need no re-setup and no secrets change:

```bash
cd /workspace/finetuneit
git pull origin main
# restart so the new code loads:
#   Ctrl+C the start_api.sh terminal, then re-run it
#   Ctrl+C the start_streamlit.sh terminal, then re-run it
```

`git pull` only updates files on disk; the running backend/UI serve old code
until restarted. Re-run `setup_pod_env.sh` only if `requirements.txt` changed or
the pod is brand new.

## Troubleshooting

- **Streamlit "Rejecting WebSocket connection from disallowed origin"** — the
  start script passes `--server.enableCORS false --server.enableXsrfProtection
  false`; include those flags if launching Streamlit manually behind the proxy.
- **UI "Read timed out" on Start Training** — local training runs synchronously;
  the UI timeout is long by default and the backend finishes regardless. Watch
  the backend terminal for progress.
- **W&B run doesn't appear** — the `api_key:` line needs a space after the colon,
  and `platform_config.yaml` should have your real entity (not the placeholder).
- **`chat` template error** — you selected chat on a model with no chat template
  (base models have none). Use an instruct model, or override to `instruction`.
- **Empty generation on a base model** — base models are sensitive to prompt
  format; the instruction templates now omit the empty `### Input:` block, so
  no-input prompts respond. Instruct models are the more reliable default.
- **`TimeoutError: HuggingFace seems to be down`** — a model download stalled
  (often first use of a new model). Retry, or pre-warm:
  `python3.12 -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('<model>', max_seq_length=1024, load_in_4bit=True)"`.
- **Deprecation warnings** (`AttentionMaskConverter`, `max_new_tokens will take
  precedence`) — cosmetic; ignore.
- Keep the pod running for the duration — local mode trains in-process.
