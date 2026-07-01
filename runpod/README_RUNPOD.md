# RunPod Setup

This project contains three RunPod workers:

```text
Training worker     runpod/train_handler.py
Evaluation worker   runpod/evaluation_handler.py
Inference worker    runpod/inference_handler.py
```

## Training worker

Build:

```bash
docker build -f runpod/Dockerfile.train -t your-dockerhub/finetuneit-train:latest .
docker push your-dockerhub/finetuneit-train:latest
```

Required environment variables in RunPod:

```bash
HF_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=finetuneit-lite
```

Training payload:

```json
{
  "input": {
    "config": {
      "experiment_name": "qwen25_7b_reasoning",
      "dataset_mode": "hf_dataset",
      "dataset_type": "reasoning",
      "dataset_id": "FreedomIntelligence/medical-o1-reasoning-SFT",
      "dataset_config": "en",
      "dataset_split": "train",
      "dataset_sample_size": 1500,
      "train_test_split": 0.2,
      "prompt_template": "reasoning_alpaca",
      "source_columns": {
        "instruction": "Question",
        "input": "Complex_CoT",
        "output": "Response"
      },
      "filter_max_tokens": 1024,
      "model": {
        "base_model": "unsloth/Qwen2.5-7B",
        "max_seq_length": 1024,
        "load_in_4bit": true
      },
      "lora": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "use_gradient_checkpointing": "unsloth",
        "use_rslora": false,
        "random_state": 3407
      },
      "training": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 20,
        "max_steps": 250,
        "learning_rate": 0.0002,
        "logging_steps": 25,
        "optim": "adamw_8bit",
        "weight_decay": 0.1,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "fp16_full_eval": true,
        "per_device_eval_batch_size": 2,
        "eval_accumulation_steps": 4,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "load_best_model_at_end": true,
        "metric_for_best_model": "eval_loss"
      },
      "output": {
        "output_dir": "outputs/qwen25_7b_reasoning",
        "output_adapter_repo": "your-hf-username/qwen25-7b-reasoning-lora"
      }
    }
  }
}
```
