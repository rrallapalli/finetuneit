#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python3.12 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
