from pathlib import Path
import yaml


def load_training_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config_with_overrides(base_config: dict, overrides: dict | None = None) -> dict:
    if not overrides:
        return base_config

    merged = base_config.copy()

    def deep_update(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                deep_update(target[key], value)
            else:
                target[key] = value

    deep_update(merged, overrides)
    return merged
