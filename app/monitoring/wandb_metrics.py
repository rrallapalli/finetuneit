import wandb
import pandas as pd

from app.core.secrets import apply_secrets_to_environment


def fetch_wandb_run_summary(entity: str, project: str, run_id: str) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "summary": dict(run.summary),
        "config": dict(run.config),
    }


def fetch_wandb_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    history = run.history(samples=samples)

    if history is None or history.empty:
        return {"columns": [], "records": []}

    history = history.replace({pd.NA: None})
    return {
        "columns": list(history.columns),
        "records": history.to_dict(orient="records"),
    }
