# Platform Configuration

FineTuneIT Lite v4 separates **secrets** from **runtime configuration**.

## Secrets

Keep secrets in `.env` or environment variables:

```bash
HF_TOKEN=
WANDB_API_KEY=
RUNPOD_API_KEY=
```

## Runtime settings

Keep runtime settings in:

```text
configs/platform_config.yaml
```

Example:

```yaml
platform:
  execution_mode: local
  api_base_url: http://localhost:8000

wandb:
  entity: your-wandb-entity
  project: finetuneit-lite
  default_samples: 1000

runpod:
  train_endpoint_id:
  eval_endpoint_id:
  inference_endpoint_id:

ui:
  refresh_seconds: 10
  default_max_runs: 50

registry:
  champion_model_name: champion_model
  export_path: registry/experiment_registry.csv

inference:
  default_base_model: unsloth/Qwen2.5-0.5B
  default_adapter_repo: your-hf-username/finetuneit-demo-adapter
  default_prompt_template: alpaca
  default_max_new_tokens: 128
  default_temperature: 0.7
```

## Why this design

This gives the project a more platform-like architecture:

```text
Secrets
  → .env

Runtime behavior
  → platform_config.yaml

Experiment setup
  → experiment YAML configs
```

## Changing execution mode

To run locally:

```yaml
platform:
  execution_mode: local
```

To run on RunPod serverless:

```yaml
platform:
  execution_mode: runpod
```
