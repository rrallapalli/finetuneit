import runpod
from app.training.train_lora import run_training_from_config


def handler(job):
    try:
        payload = job["input"]
        config = payload["config"] if "config" in payload else payload
        return run_training_from_config(config)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
