import os
from pathlib import Path
from functools import lru_cache

import yaml


DEFAULT_SECRETS_PATH = Path(os.getenv("SECRETS_CONFIG_PATH", "configs/secrets/secrets.yaml"))


@lru_cache(maxsize=1)
def load_secrets(secrets_path: str | None = None) -> dict:
    path = Path(secrets_path) if secrets_path else DEFAULT_SECRETS_PATH

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_secret(path: str, env_var: str | None = None, default: str | None = None) -> str | None:
    """
    Read a nested secret from configs/secrets/secrets.yaml with environment variable fallback.

    Example:
        get_secret("huggingface.token", "HF_TOKEN")
    """
    data = load_secrets()
    current = data

    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            current = None
            break
        current = current[part]

    if current:
        return str(current)

    if env_var:
        return os.getenv(env_var, default)

    return default


def get_hf_token() -> str | None:
    return get_secret("huggingface.token", "HF_TOKEN")


def get_wandb_api_key() -> str | None:
    return get_secret("wandb.api_key", "WANDB_API_KEY")


def get_runpod_api_key() -> str | None:
    return get_secret("runpod.api_key", "RUNPOD_API_KEY")


def apply_secrets_to_environment() -> None:
    """
    Populate process environment variables for libraries that expect them.

    W&B, Hugging Face, and RunPod SDKs often read environment variables internally.
    This function allows secrets.yaml to be the local source while preserving
    library compatibility.
    """
    hf_token = get_hf_token()
    wandb_key = get_wandb_api_key()
    runpod_key = get_runpod_api_key()

    if hf_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    if wandb_key and not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = wandb_key

    if runpod_key and not os.getenv("RUNPOD_API_KEY"):
        os.environ["RUNPOD_API_KEY"] = runpod_key
