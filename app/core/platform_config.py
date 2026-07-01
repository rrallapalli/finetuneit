import os
from pathlib import Path
from functools import lru_cache

import yaml


DEFAULT_CONFIG_PATH = Path(os.getenv("PLATFORM_CONFIG_PATH", "configs/platform_config.yaml"))


@lru_cache(maxsize=1)
def load_platform_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Platform config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_execution_mode() -> str:
    config = load_platform_config()
    return config.get("platform", {}).get("execution_mode", "runpod").lower()


def get_wandb_project() -> str:
    config = load_platform_config()
    return config.get("wandb", {}).get("project", "finetuneit-lite")


def get_wandb_entity() -> str:
    config = load_platform_config()
    return config.get("wandb", {}).get("entity", "")


def get_runpod_endpoint(kind: str) -> str | None:
    config = load_platform_config()
    runpod_cfg = config.get("runpod", {})

    key_map = {
        "train": "train_endpoint_id",
        "eval": "eval_endpoint_id",
        "inference": "inference_endpoint_id",
    }

    return runpod_cfg.get(key_map[kind])


def get_api_base_url() -> str:
    config = load_platform_config()
    return config.get("platform", {}).get("api_base_url", "http://localhost:8000")


def get_ui_config() -> dict:
    return load_platform_config().get("ui", {})


def get_registry_config() -> dict:
    return load_platform_config().get("registry", {})


def get_inference_config() -> dict:
    return load_platform_config().get("inference", {})
