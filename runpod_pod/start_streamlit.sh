#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# --server.enableCORS false / --enableXsrfProtection false are required to serve
# Streamlit through RunPod's reverse proxy; without them the proxy origin is
# rejected and the WebSocket never connects. Safe for a single-user pod behind
# RunPod's authenticated proxy.
python3.12 -m streamlit run app/ui/streamlit_app.py \
  --server.address 0.0.0.0 --server.port 8501 \
  --server.enableCORS false --server.enableXsrfProtection false
