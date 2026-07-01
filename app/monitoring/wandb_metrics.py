import math

import wandb

from app.core.secrets import apply_secrets_to_environment


def _finite(x):
    """Return the float if finite, else None (JSON has no NaN/Infinity)."""
    try:
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return x


def _json_safe(obj):
    """Make W&B data JSON-serializable. W&B history uses NaN for metrics that
    weren't logged at a given step, and JSON has no NaN/Infinity, so FastAPI's
    encoder 500s on them. Recursively replace non-finite floats with None.

    Note: we must NOT call hasattr()/getattr() on arbitrary W&B objects here.
    Objects like SummarySubDict override __getattr__ to look up summary keys, so
    hasattr(obj, "item") raises KeyError instead of returning False. Callers
    convert such objects to plain dicts before passing them in."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _finite(obj)
    # numpy scalar (np.float32/64, np.int64, ...): convert without hasattr,
    # which could trigger a custom __getattr__ on non-numpy objects.
    if type(obj).__module__ == "numpy" and hasattr(type(obj), "item"):
        val = obj.item()
        return _finite(val) if isinstance(val, float) else val
    return obj


def fetch_wandb_run_summary(entity: str, project: str, run_id: str) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Convert W&B's special summary/config objects to plain dicts first.
    summary = {k: v for k, v in dict(run.summary).items()}
    config = {k: v for k, v in dict(run.config).items()}

    return _json_safe({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "summary": summary,
        "config": config,
    })


def fetch_wandb_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    history = run.history(samples=samples)

    if history is None or history.empty:
        return {"columns": [], "records": []}

    return _json_safe({
        "columns": [str(c) for c in history.columns],
        "records": history.to_dict(orient="records"),
    })
