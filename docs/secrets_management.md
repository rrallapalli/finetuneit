# Secrets Management

FineTuneIT Lite v5 supports sourcing secrets from a local YAML file.

## Recommended local setup

Copy the example file:

```bash
cp configs/secrets/secrets.example.yaml configs/secrets/secrets.yaml
```

Edit:

```yaml
huggingface:
  token: hf_xxxxx

wandb:
  api_key: xxxxx

runpod:
  api_key: xxxxx
```

## Important

Do not commit:

```text
configs/secrets/secrets.yaml
```

It is already included in `.gitignore`.

## Fallback behavior

If `configs/secrets/secrets.yaml` is missing, the project falls back to environment variables:

```bash
HF_TOKEN
WANDB_API_KEY
RUNPOD_API_KEY
```

## Architecture

```text
configs/platform_config.yaml
  → runtime settings

configs/experiments/*.yaml
  → experiment settings

configs/secrets/secrets.yaml
  → local credentials
```

This keeps the project clean and closer to enterprise-style AI platform configuration.
