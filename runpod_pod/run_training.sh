#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CONFIG_PATH=${1:-configs/custom_jsonl.yaml}
export PYTHONPATH=.
python3.12 - <<PY
from app.training.config import load_training_config
from app.training.train_lora import run_training_from_config
config = load_training_config("$CONFIG_PATH")
print(run_training_from_config(config))
PY
