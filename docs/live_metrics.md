# Live W&B Metrics

The Live Metrics tab uses W&B as the source of truth.

## Flow

```text
Training script logs to W&B
  ↓
FastAPI exposes /wandb/history
  ↓
Streamlit polls the endpoint
  ↓
Charts update every few seconds
```

## Metrics shown

```text
train/loss
eval/loss
train/learning_rate
train/grad_norm
_step
_runtime
```
