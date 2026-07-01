# Experiment Registry and Model Ranking

The Experiment Registry tab allows users to compare previous W&B runs and select the best candidate model for inference.

## Capabilities

```text
List W&B runs
Rank models using a heuristic model score
Compare selected runs
Promote a champion model
Export registry CSV
Use champion model in Inference Playground
```

## Ranking score

The current scoring function is intentionally simple and explainable:

```text
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
```

The score is not meant to be a universal quality score. It is a practical product feature that helps users shortlist candidate adapters before manual review.

## Champion model

When a model is promoted, the metadata is saved to:

```text
registry/champion_model.json
```

This makes the inference flow more product-like:

```text
Experiment tracking
  → model comparison
  → champion promotion
  → inference testing
```
