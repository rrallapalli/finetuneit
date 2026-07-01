# RunPod Pod Mode

This mode is closest to the original workflow where notebooks were executed on a RunPod GPU pod.

Instead of running notebooks manually, this mode runs reusable scripts.

## Setup

```bash
bash runpod_pod/setup_pod_env.sh
```

## Run training

```bash
bash runpod_pod/run_training.sh configs/custom_jsonl.yaml
```

## Start API

```bash
bash runpod_pod/start_api.sh
```

## Start Streamlit

```bash
bash runpod_pod/start_streamlit.sh
```
