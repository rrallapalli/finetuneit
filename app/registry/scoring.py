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

    bert = safe_float(row.get("BERTScore (F1)"), 0.0)
    rouge_l = safe_float(row.get("ROUGE-L"), 0.0)
    squad_f1 = safe_float(row.get("SQuAD F1"), 0.0)
    exact = safe_float(row.get("Exact Match"), 0.0)
    bleu = safe_float(row.get("BLEU"), 0.0)

    eval_loss = safe_float(row.get("eval/loss"), None)
    train_loss = safe_float(row.get("train/loss"), None)
    latency = safe_float(row.get("Average Latency (sec)"), None)
    perplexity = safe_float(row.get("Average Perplexity"), None)

    quality_score = (
        0.35 * bert +
        0.25 * rouge_l +
        0.20 * squad_f1 +
        0.10 * exact +
        0.10 * bleu
    )

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
