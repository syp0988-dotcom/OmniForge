"""Retrieval quality metrics.

All functions accept:
    relevant: set[int]  — ground-truth relevant chunk IDs
    retrieved: list[int] — ordered list of retrieved chunk IDs (rank 1 = index 0)
"""

from __future__ import annotations

import math


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Fraction of relevant chunks found in the top-k results."""
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / min(k, len(retrieved)) if retrieved else 0.0


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result.

    Returns 0.0 when no relevant result is found.
    """
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).  IDCG is calculated
    from an ideal ranking where all relevant items appear first.
    """
    if not relevant or k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        rel = 1 if chunk_id in relevant else 0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0 for rank 1

    # IDCG — ideal order: all relevant first
    ideal_rels = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(relevant: set[int], retrieved: list[int], k: int) -> bool:
    """Whether at least one relevant chunk appears in the top-k results."""
    top_k = set(retrieved[:k])
    return bool(top_k & relevant)


def compute_all(relevant: set[int], retrieved: list[int], k_values: list[int] | None = None
                ) -> dict[str, float]:
    """Compute all standard retrieval metrics at once.

    Args:
        relevant: Ground-truth relevant chunk IDs.
        retrieved: Ordered list of retrieved chunk IDs.
        k_values: Cutoffs to evaluate (default [1, 3, 5, 10]).

    Returns:
        Dict mapping metric name to value.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        metrics[f"precision@{k}"] = precision_at_k(relevant, retrieved, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevant, retrieved, k)
        metrics[f"hit@{k}"] = 1.0 if hit_rate(relevant, retrieved, k) else 0.0
    metrics["mrr"] = mrr(relevant, retrieved)
    return metrics


def aggregate(per_sample_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-sample metric dicts into summary statistics.

    Hit-rate metrics (``hit@k``) are already 0/1 so averaging yields proportion.
    """
    if not per_sample_metrics:
        return {}
    keys = per_sample_metrics[0].keys()
    n = len(per_sample_metrics)
    return {key: sum(m[key] for m in per_sample_metrics) / n for key in keys}
