import math


def _finite(x):
    """Return the number if it's a finite float, else None (JSON has no NaN/Inf)."""
    try:
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return x


def json_safe(obj):
    """Recursively make an object JSON-serializable for FastAPI.

    W&B summaries/history use NaN for metrics not logged at a step, and JSON has
    no NaN/Infinity, so FastAPI's encoder raises 'Out of range float values are
    not JSON compliant: nan'. This replaces non-finite floats with None.

    Important: do NOT call hasattr()/getattr() on arbitrary objects here -- some
    W&B objects (e.g. SummarySubDict) override __getattr__ to look up keys and
    raise KeyError for unknown names. We gate numpy handling on the type's module.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _finite(obj)
    if type(obj).__module__ == "numpy" and hasattr(type(obj), "item"):
        val = obj.item()
        return _finite(val) if isinstance(val, float) else val
    return obj
