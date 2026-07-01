# Notebook Sources

The original reasoning and non-reasoning notebooks were used to derive the configurable training profiles in `configs/`.

They are intentionally not included in this GitHub package because notebook exports can contain secrets such as W&B keys or Hugging Face tokens.

The platformized equivalents are:

- `configs/reasoning_qwen25_7b.yaml`
- `configs/non_reasoning_qwen25_7b.yaml`
- `app/training/train_lora.py`
- `app/training/dataset_loader.py`
- `app/training/prompt_templates.py`
