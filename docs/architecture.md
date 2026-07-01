# Architecture

## Platform workflow

```text
User / Streamlit UI
  ↓
Configurable Experiment Form
  ↓
FastAPI Control Plane
  ↓
RunPod Serverless Workers
  ├── Training Worker
  ├── Evaluation Worker
  └── Inference Worker
  ↓
External Services
  ├── Hugging Face Hub
  └── Weights & Biases
```

## Training flow

```text
Experiment Config
  ↓
Dataset Loader
  ↓
Prompt Formatter
  ↓
Unsloth Base Model Loader
  ↓
LoRA Adapter Training
  ↓
W&B Metrics
  ↓
Adapter Push to Hugging Face
```

## Evaluation flow

```text
Base Model + Adapter
  ↓
Evaluation Dataset
  ↓
Response Generation
  ↓
Metric Computation
  ↓
Evaluation CSV
  ↓
W&B Evaluation Logs
```

## Inference flow

```text
Prompt
  ↓
Base Model from Hugging Face
  ↓
LoRA Adapter from Hugging Face
  ↓
Generated Response
```
