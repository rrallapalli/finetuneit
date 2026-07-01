"""
smoke_test.py - fast end-to-end check of the training path.

Runs a ~10-step LoRA job on the sample dataset to exercise the full pipeline:
load -> LoRA -> SFTConfig -> train() -> save. Finishes in ~1-2 minutes on a GPU.

Run it after ANY dependency bump or code change, before trusting a real run:

    python3.12 smoke_test.py

Exit code 0 = path healthy. 1 = something broke (the error is printed).
This is the check that would have caught every breakage we hit, in ~2 minutes.
"""
import sys

from app.training.config import load_training_config, merge_config_with_overrides
from app.training.train_lora import run_training_from_config

BASE_CONFIG = "configs/custom_jsonl.yaml"

OVERRIDES = {
    "experiment_name": "smoke_test",
    "wandb_project": "finetuneit-smoke",   # keeps smoke runs out of your main project
    "training": {
        "max_steps": 10,
        "warmup_steps": 2,
        "logging_steps": 1,
        "save_steps": 10,
        "save_total_limit": 1,
        "eval_steps": 5,
    },
    "output": {
        "output_dir": "outputs/smoke_test",
        # keep the placeholder prefix so the run never pushes to the HF Hub
        "output_adapter_repo": "your-hf-username/smoke",
        "export_merged": False,
        "export_gguf": False,
    },
}


def main() -> int:
    print("Running smoke test: 10 steps on the sample dataset...\n")
    config = merge_config_with_overrides(load_training_config(BASE_CONFIG), OVERRIDES)
    result = run_training_from_config(config)

    if result.get("status") == "completed":
        loss = result.get("train_metrics", {}).get("train_loss", "n/a")
        print(f"\n[PASS] training completed. train_loss={loss}")
        print(f"       adapter saved to: {result.get('output_dir')}")
        return 0

    print("\n[FAIL] training did not complete.")
    print(f"       status: {result.get('status')}")
    print(f"       error : {result.get('error')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
