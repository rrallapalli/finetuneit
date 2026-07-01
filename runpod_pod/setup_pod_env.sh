#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Install into the pod's system Python via --break-system-packages so we REUSE
# the RunPod PyTorch template's CUDA-matched torch instead of a fresh venv that
# would reinstall a possibly-mismatched wheel. The pod is the sandbox.
PYTHON=python3.12

echo "Using Python: $($PYTHON --version)"

echo "Checking the pod's existing PyTorch / CUDA..."
$PYTHON - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA not available under python3.12. Use a RunPod PyTorch template, "
        "or set PYTHON in this script to the interpreter that owns torch."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Upgrading pip..."
$PYTHON -m pip install --upgrade pip --break-system-packages

echo "Installing project requirements (torch is reused, not reinstalled)..."
$PYTHON -m pip install -r requirements.txt --break-system-packages

echo "Installing Unsloth WITHOUT touching torch..."
$PYTHON -m pip install unsloth unsloth_zoo --no-deps --break-system-packages

echo "Verifying..."
$PYTHON - <<'PY'
import torch, transformers, trl, peft
from unsloth import FastLanguageModel
print("Torch:", torch.__version__, "| Transformers:", transformers.__version__, "| TRL:", trl.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Setup OK")
PY

echo "RunPod pod environment setup completed."
