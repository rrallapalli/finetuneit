
## v5 Secrets Setup

In v5, you can source tokens from a local secrets file.

Copy:

```bash
cp configs/secrets/secrets.example.yaml configs/secrets/secrets.yaml
```

Edit:

```yaml
huggingface:
  token: your_hf_token

wandb:
  api_key: your_wandb_api_key

runpod:
  api_key: your_runpod_api_key
```

`configs/secrets/secrets.yaml` is ignored by Git and should not be committed.

You can still use environment variables as fallback:

```bash
HF_TOKEN=
WANDB_API_KEY=
RUNPOD_API_KEY=
```



## v4 Configuration Update

In v4, `.env` is only for secrets:

```bash
HF_TOKEN=
WANDB_API_KEY=
RUNPOD_API_KEY=
```

Runtime settings are now in:

```text
configs/platform_config.yaml
```

For local mode, set:

```yaml
platform:
  execution_mode: local
```

For RunPod mode, set:

```yaml
platform:
  execution_mode: runpod

runpod:
  train_endpoint_id: your_train_endpoint
  eval_endpoint_id: your_eval_endpoint
  inference_endpoint_id: your_inference_endpoint
```


# How to Run FineTuneIT Lite

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure environment

```bash
cp .env.example .env
```

Set:

```bash
HF_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=finetuneit-lite
RUNPOD_API_KEY=
RUNPOD_TRAIN_ENDPOINT_ID=
RUNPOD_EVAL_ENDPOINT_ID=
RUNPOD_INFERENCE_ENDPOINT_ID=
```

## 3. Start API

```bash
uvicorn app.api.main:app --reload
```

## 4. Start UI

```bash
streamlit run app/ui/streamlit_app.py
```

## 5. Configure experiment in UI

Choose:

```text
Experiment Profile:
- Non-reasoning instruction tuning
- Reasoning instruction tuning
- Custom JSONL

Base Model:
- unsloth/Qwen2.5-0.5B
- unsloth/Qwen2.5-1.5B
- unsloth/Qwen2.5-3B
- unsloth/Qwen2.5-7B

Training:
- max_seq_length
- dataset_sample_size
- train_test_split
- batch_size
- gradient_accumulation_steps
- warmup_steps
- max_steps
- learning_rate
- weight_decay
- logging_steps
- eval_steps
- save_steps

LoRA:
- r
- lora_alpha
- lora_dropout
- target_modules
- use_rslora
- use_gradient_checkpointing
```

## 6. Submit training job

The UI sends the full configuration to:

```text
POST /jobs/train
```

The backend forwards it to the RunPod training worker.

## 7. Submit evaluation job

After the adapter is pushed to Hugging Face, run:

```text
POST /evaluation/run
```

## 8. Run inference

Use:

```text
POST /inference/
```

with:

```json
{
  "base_model": "unsloth/Qwen2.5-0.5B",
  "adapter_repo": "your-hf-username/finetuneit-demo-adapter",
  "prompt": "Explain credit risk in simple terms."
}
```
