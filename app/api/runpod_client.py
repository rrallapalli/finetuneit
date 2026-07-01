import requests

from app.core.secrets import get_runpod_api_key


def submit_runpod_job(endpoint_id: str, payload: dict) -> dict:
    api_key = get_runpod_api_key()

    if not api_key:
        raise RuntimeError(
            "RunPod API key is not configured. Add it to configs/secrets/secrets.yaml "
            "or set RUNPOD_API_KEY as an environment variable."
        )

    if not endpoint_id:
        raise RuntimeError("RunPod endpoint ID is not configured in configs/platform_config.yaml.")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    response = requests.post(
        url,
        headers={"Authorization": api_key},
        json={"input": payload},
        timeout=60,
    )

    response.raise_for_status()
    return response.json()
