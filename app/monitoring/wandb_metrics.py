import math

import wandb

from app.core.secrets import apply_secrets_to_environment


def _json_safe(obj):
    """Make W&B data JSON-serializable. W&B history uses NaN for metrics that
    weren't logged at a given step (e.g. eval_loss on non-eval steps), and JSON
    has no NaN/Infinity, so FastAPI's encoder 500s on them. Recursively convert
    numpy scalars to native types and replace any non-finite float with None."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()  # numpy scalar -> native python
        except Exception:
            pass
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def fetch_wandb_run_summary(entity: str, project: str, run_id: str) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    return _json_safe({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "summary": dict(run.summary),
        "config": dict(run.config),
    })


def fetch_wandb_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    history = run.history(samples=samples)

    if history is None or history.empty:
        return {"columns": [], "records": []}

    return _json_safe({
        "columns": list(history.columns),
        "records": history.to_dict(orient="records"),
    })
