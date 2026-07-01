# Configurable Fine-Tuning

The original notebooks had separate logic for:

1. Non-reasoning instruction tuning
2. Reasoning-style instruction tuning

This package turns those notebook-level choices into UI parameters.

## Key configurable fields

### Dataset

```yaml
dataset_mode: hf_dataset
dataset_type: reasoning
dataset_id: FreedomIntelligence/medical-o1-reasoning-SFT
dataset_config: en
dataset_sample_size: 1500
source_columns:
  instruction: Question
  input: Complex_CoT
  output: Response
```

For non-reasoning:

```yaml
dataset_mode: hf_dataset
dataset_type: non_reasoning
dataset_id: tatsu-lab/alpaca
dataset_sample_size: 2500
```

### Model

```yaml
model:
  base_model: unsloth/Qwen2.5-7B
  max_seq_length: 1024
  load_in_4bit: true
```

### LoRA

```yaml
lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### Training

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  warmup_steps: 20
  max_steps: 250
  learning_rate: 0.0002
  logging_steps: 25
  optim: adamw_8bit
  weight_decay: 0.1
  eval_steps: 50
  save_steps: 50
```

## UI design

The Streamlit UI exposes these as form controls so the user can run new experiments without editing Python notebooks.

## Portfolio message

This shows the productization step clearly:

```text
Notebook experiments
  → reusable configs
  → UI controls
  → RunPod jobs
  → W&B comparison
  → Hugging Face adapter registry
```
