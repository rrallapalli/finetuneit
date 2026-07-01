from fastapi import APIRouter
from app.monitoring.wandb_metrics import fetch_wandb_run_summary, fetch_wandb_history

router = APIRouter(prefix="/wandb", tags=["wandb"])


@router.get("/run")
def get_run(entity: str, project: str, run_id: str):
    return fetch_wandb_run_summary(entity, project, run_id)


@router.get("/history")
def get_history(entity: str, project: str, run_id: str, samples: int = 500):
    return fetch_wandb_history(entity, project, run_id, samples=samples)
