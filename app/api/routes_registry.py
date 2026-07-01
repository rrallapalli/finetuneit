from fastapi import APIRouter
from app.registry.experiment_registry import (
    list_wandb_runs,
    save_champion_model,
    load_champion_model,
    export_registry_csv,
)

router = APIRouter(prefix="/registry", tags=["registry"])


@router.get("/runs")
def get_runs(entity: str, project: str, max_runs: int = 50):
    return {
        "runs": list_wandb_runs(entity=entity, project=project, max_runs=max_runs)
    }


@router.post("/champion")
def promote_champion(payload: dict):
    name = payload.get("name", "champion_model")
    model_record = payload["model"]
    return save_champion_model(model_record, name=name)


@router.get("/champion")
def get_champion(name: str = "champion_model"):
    return load_champion_model(name=name)


@router.post("/export")
def export_registry(payload: dict):
    rows = payload.get("rows", [])
    path = payload.get("path", "registry/experiment_registry.csv")
    return export_registry_csv(rows, path)
