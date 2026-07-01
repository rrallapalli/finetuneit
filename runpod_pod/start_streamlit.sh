#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python3.12 -m streamlit run app/ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
