from fastapi import APIRouter

from app.api.runpod_client import submit_runpod_job
from app.core.platform_config import get_execution_mode, get_runpod_endpoint, get_wandb_project
from app.evaluation.evaluate_model import evaluate_model

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
def create_evaluation_job(payload: dict):
    if get_execution_mode() == "local":
        return evaluate_model(
            base_model=payload["base_model"],
            adapter_repo=payload.get("adapter_repo"),
            dataset_path=payload["dataset"],
            num_samples=payload.get("num_samples", 25),
            output_path=payload.get("output_path", "outputs/evaluation_metrics.csv"),
            wandb_project=payload.get("wandb_project", get_wandb_project()),
            job_id=payload.get("job_id", "eval-job"),
            prompt_template=payload.get("prompt_template", "alpaca"),
            max_new_tokens=payload.get("max_new_tokens", 256),
            template_type=payload.get("template_type", "instruction"),
        )

    endpoint_id = get_runpod_endpoint("eval")
    return submit_runpod_job(endpoint_id, payload)
