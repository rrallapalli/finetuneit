from fastapi import APIRouter

from app.api.runpod_client import submit_runpod_job
from app.core.platform_config import get_execution_mode, get_runpod_endpoint
from app.inference.serve_adapter import generate_response

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/")
def inference(payload: dict):
    if get_execution_mode() == "local":
        response = generate_response(
            base_model=payload["base_model"],
            adapter_repo=payload["adapter_repo"],
            prompt=payload["prompt"],
            input_text=payload.get("input_text", ""),
            prompt_template=payload.get("prompt_template", "alpaca"),
            max_new_tokens=payload.get("max_new_tokens", 128),
        )
        return {"response": response}

    endpoint_id = get_runpod_endpoint("inference")
    return submit_runpod_job(endpoint_id, payload)
