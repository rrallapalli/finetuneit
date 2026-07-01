from collections import Counter
from statistics import mean
import re
import string

import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(reference: str, prediction: str) -> int:
    return int(normalize_answer(reference) == normalize_answer(prediction))


def compute_f1(reference: str, prediction: str) -> float:
    gold_toks = normalize_answer(reference).split()
    pred_toks = normalize_answer(prediction).split()

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)

    return 2 * precision * recall / (precision + recall)


def compute_text_generation_metrics(
    predictions: list[str],
    references: list[str],
    perplexities: list[float] | None = None,
    latencies: list[float] | None = None,
    throughputs: list[float] | None = None,
    gpu_mem_usages: list[float] | None = None,
) -> dict:
    predictions = [p.strip() for p in predictions]
    references = [r.strip() for r in references]

    rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
    bleu = evaluate.load("bleu").compute(predictions=predictions, references=[[r] for r in references])
    bert = evaluate.load("bertscore").compute(predictions=predictions, references=references, lang="en")
    meteor = evaluate.load("meteor").compute(predictions=predictions, references=references)

    exact_matches = [int(p == r) for p, r in zip(predictions, references)]
    squad_exact = [compute_exact(ref, pred) for ref, pred in zip(references, predictions)]
    squad_f1 = [compute_f1(ref, pred) for ref, pred in zip(references, predictions)]

    smooth = SmoothingFunction().method1
    mt_bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        for ref, pred in zip(references, predictions)
    ]

    mt_length_ratios = [
        len(pred.split()) / max(len(ref.split()), 1)
        for ref, pred in zip(references, predictions)
    ]

    return {
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "ROUGE-L": rouge["rougeL"],
        "BLEU": bleu["bleu"],
        "BERTScore (F1)": float(np.mean(bert["f1"])),
        "Exact Match": float(np.mean(exact_matches)),
        "METEOR Score": meteor["meteor"],
        "SQuAD EM": float(np.mean(squad_exact)),
        "SQuAD F1": float(np.mean(squad_f1)),
        "MT-BLEU mean": float(mean(mt_bleu_scores)),
        "MT length ratio": float(mean(mt_length_ratios)),
        "Average Perplexity": float(np.mean(perplexities)) if perplexities else None,
        "Average Latency (sec)": float(np.mean(latencies)) if latencies else None,
        "Average Throughput (token/sec)": float(np.mean(throughputs)) if throughputs else None,
        "Max GPU Used": float(max(gpu_mem_usages)) if gpu_mem_usages else None,
    }
