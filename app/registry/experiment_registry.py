import os
import json
from pathlib import Path

import pandas as pd
import wandb

from app.registry.scoring import compute_model_score
from app.core.secrets import apply_secrets_to_environment


REGISTRY_DIR = Path("registry")
REGISTRY_DIR.mkdir(exist_ok=True)


def _extract_run_metrics(run) -> dict:
    summary = dict(run.summary)
    config = dict(run.config)

    row = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "base_model": config.get("model", {}).get("base_model") or config.get("base_model"),
        "dataset_type": config.get("dataset_type"),
        "dataset_id": config.get("dataset_id"),
        "prompt_template": config.get("prompt_template"),
        "adapter_repo": (
            config.get("output", {}).get("output_adapter_repo")
            or config.get("output_adapter_repo")
            or summary.get("adapter_repo")
        ),
        "train/loss": summary.get("train/loss"),
        "eval/loss": summary.get("eval/loss"),
        "ROUGE-L": summary.get("ROUGE-L"),
        "BERTScore (F1)": summary.get("BERTScore (F1)"),
        "SQuAD F1": summary.get("SQuAD F1"),
        "Exact Match": summary.get("Exact Match"),
        "BLEU": summary.get("BLEU"),
        "Average Perplexity": summary.get("Average Perplexity"),
        "Average Latency (sec)": summary.get("Average Latency (sec)"),
    }

    row["model_score"] = compute_model_score(row)
    return row


def list_wandb_runs(entity: str, project: str, max_runs: int = 50) -> list[dict]:
    apply_secrets_to_environment()
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=max_runs)

    rows = []
    for run in runs:
        try:
            rows.append(_extract_run_metrics(run))
        except Exception as exc:
            rows.append({
                "run_id": getattr(run, "id", None),
                "run_name": getattr(run, "name", None),
                "state": getattr(run, "state", None),
                "url": getattr(run, "url", None),
                "error": str(exc),
                "model_score": 0.0,
            })

    rows = sorted(rows, key=lambda x: x.get("model_score", 0), reverse=True)
    return rows


def save_champion_model(model_record: dict, name: str = "champion_model") -> dict:
    path = REGISTRY_DIR / f"{name}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(model_record, f, indent=2)

    return {
        "status": "saved",
        "registry_path": str(path),
        "model": model_record,
    }


def load_champion_model(name: str = "champion_model") -> dict:
    path = REGISTRY_DIR / f"{name}.json"

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_registry_csv(rows: list[dict], path: str = "registry/experiment_registry.csv") -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return {
        "status": "exported",
        "path": str(output_path),
        "rows": len(rows),
    }
