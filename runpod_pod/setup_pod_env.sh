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

# Some base-image packages (e.g. cryptography, PyYAML) are installed by apt
# WITHOUT a pip RECORD file. When a later dependency needs a newer version, pip
# tries to uninstall the apt copy, can't (no RECORD), and aborts with
# "uninstall-no-record-file". We pre-empt this: overlay pip-owned copies of the
# known apt-managed packages FIRST. After this, pip owns them (with a RECORD),
# so the main requirements install can upgrade them freely and never collides.
# This stays fast -- only these 1-2 packages are overlaid; the rest of the
# pod's baked stack is reused, not reinstalled. Add offenders here if a new
# template surprises us.
APT_OWNED="cryptography PyYAML"
echo "Pre-overlaying apt-owned packages so pip owns them: $APT_OWNED"
$PYTHON -m pip install --ignore-installed --break-system-packages $APT_OWNED

echo "Installing project requirements (torch is reused, not reinstalled)..."
$PYTHON -m pip install -r requirements.txt --break-system-packages

echo "Installing Unsloth WITHOUT touching torch..."
$PYTHON -m pip install "unsloth==2026.6.9" "unsloth_zoo==2026.6.7" --no-deps --break-system-packages

echo "Verifying..."
$PYTHON - <<'PY'
import torch, transformers, trl, peft
from unsloth import FastLanguageModel
print("Torch:", torch.__version__, "| Transformers:", transformers.__version__, "| TRL:", trl.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Setup OK")
PY

echo "RunPod pod environment setup completed."
