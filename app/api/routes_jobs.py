from fastapi import APIRouter

from app.api.runpod_client import submit_runpod_job
from app.core.platform_config import get_execution_mode, get_runpod_endpoint
from app.training.train_lora import run_training_from_config

router = APIRouter(prefix="/jobs", tags=["training"])


@router.post("/train")
def create_training_job(payload: dict):
    config = payload.get("config", payload)

    if get_execution_mode() == "local":
        return run_training_from_config(config)

    endpoint_id = get_runpod_endpoint("train")
    return submit_runpod_job(endpoint_id, {"config": config})
