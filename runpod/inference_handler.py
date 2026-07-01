import runpod
from app.inference.serve_adapter import generate_response


def handler(job):
    try:
        payload = job["input"]
        response = generate_response(
            base_model=payload["base_model"],
            adapter_repo=payload["adapter_repo"],
            prompt=payload["prompt"],
            input_text=payload.get("input_text", ""),
            prompt_template=payload.get("prompt_template", "alpaca"),
            max_new_tokens=payload.get("max_new_tokens", 128),
        )
        return {"response": response}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
