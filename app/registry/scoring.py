def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_model_score(row: dict) -> float:
    """
    Heuristic model ranking score.

    Higher is better. Uses available metrics and gracefully handles missing values.

    Positive contributors:
    - BERTScore F1
    - ROUGE-L
    - SQuAD F1
    - Exact Match
    - BLEU

    Penalties:
    - eval/loss
    - train/loss
    - latency
    - perplexity
    """

    # Quality metrics with weights. A metric that is MISSING (None) is excluded
    # and the remaining weights are renormalized — otherwise an absent metric
    # (e.g. BERTScore failing on newer transformers) silently acts as a zero
    # with 0.35 weight and distorts the ranking. A metric that is present but
    # 0.0 still counts as a genuine zero.
    weighted = [
        (0.35, safe_float(row.get("BERTScore (F1)"), None)),
        (0.25, safe_float(row.get("ROUGE-L"), None)),
        (0.20, safe_float(row.get("SQuAD F1"), None)),
        (0.10, safe_float(row.get("Exact Match"), None)),
        (0.10, safe_float(row.get("BLEU"), None)),
    ]
    present = [(w, v) for w, v in weighted if v is not None]
    if present:
        total_w = sum(w for w, _ in present)
        quality_score = sum(w * v for w, v in present) / total_w
    else:
        quality_score = 0.0

    eval_loss = safe_float(row.get("eval/loss"), None)
    train_loss = safe_float(row.get("train/loss"), None)
    latency = safe_float(row.get("Average Latency (sec)"), None)
    perplexity = safe_float(row.get("Average Perplexity"), None)

    penalty = 0.0

    if eval_loss is not None:
        penalty += min(eval_loss / 10, 0.25)

    if train_loss is not None:
        penalty += min(train_loss / 10, 0.10)

    if latency is not None:
        penalty += min(latency / 100, 0.10)

    if perplexity is not None:
        penalty += min(perplexity / 100, 0.15)

    return round(max(quality_score - penalty, 0.0), 6)
