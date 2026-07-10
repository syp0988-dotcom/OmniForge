"""Evaluation runner: execute queries against KnowledgeStore, compute metrics."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agentflow.knowledge.eval.dataset import EvalDataset
from agentflow.knowledge.eval.metrics import compute_all, aggregate

logger = logging.getLogger("knowledge.eval.runner")


class EvalRunner:
    """Runs a dataset against a KnowledgeStore and collects metrics.

    Parameters
    ----------
    store : KnowledgeStore
        The knowledge store to evaluate. Must have indexed documents.
    dataset : EvalDataset
        Loaded evaluation dataset.
    k_values : list[int]
        Cutoff values for recall/precision/ndcg/hit metrics.
    """

    def __init__(
        self,
        store,
        dataset: EvalDataset,
        k_values: list[int] | None = None,
    ) -> None:
        self.store = store
        self.dataset = dataset
        self.k_values = k_values or [1, 3, 5, 10]
        self.per_sample: list[dict[str, Any]] = []
        self.summary: dict[str, float] = {}

    # -- Main entry point ---------------------------------------------------

    def run(
        self,
        top_k: int | None = None,
        min_score: float | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Execute all queries and return a full result dict.

        Args:
            top_k: Override default top_k for search.
            min_score: Override default min_score threshold.
            verbose: Whether to log progress.

        Returns:
            Dict with keys: ``summary``, ``per_sample``, ``config``.
        """
        if top_k is None:
            top_k = max(self.k_values)

        per_sample_metrics: list[dict[str, float]] = []
        self.per_sample = []

        total = len(self.dataset)
        for idx, sample in enumerate(self.dataset.samples):
            question = sample["question"]
            relevant = set(sample["relevant_chunk_ids"])

            try:
                results = self.store.search(question, top_k=top_k, min_score=min_score)
            except Exception as exc:
                logger.warning(
                    "Search failed for sample %s: %s. Treating as empty results.",
                    sample.get("id", idx), exc,
                )
                results = []

            retrieved_ids = [r["chunk_id"] for r in results]
            m = compute_all(relevant, retrieved_ids, self.k_values)

            per_sample_metrics.append(m)
            self.per_sample.append({
                "sample_id": sample.get("id", f"q{idx}"),
                "question": question,
                "relevant_chunk_ids": sorted(relevant),
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": m,
            })

            if verbose and (idx + 1) % max(1, total // 10) == 0:
                logger.info("Evaluated %d/%d samples", idx + 1, total)

        self.summary = aggregate(per_sample_metrics)
        if verbose:
            logger.info("Evaluation complete. Summary:\n%s",
                         json.dumps(self.summary, indent=2, ensure_ascii=False))

        return {
            "summary": self.summary,
            "per_sample": self.per_sample,
            "config": {
                "k_values": self.k_values,
                "top_k": top_k,
                "min_score": min_score,
                "total_samples": total,
            },
        }

    # -- I/O ----------------------------------------------------------------

    def save_results(self, path: str | Path) -> None:
        """Save full evaluation results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary,
                    "per_sample": self.per_sample,
                    "config": {
                        "k_values": self.k_values,
                        "total_samples": len(self.dataset),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Results saved to %s", path)

    # -- Quick per-sample inspection ----------------------------------------

    def worst_samples(self, metric: str = "recall@5", n: int = 10) -> list[dict[str, Any]]:
        """Return the *n* samples with the lowest score on a given metric."""
        sorted_samples = sorted(
            self.per_sample,
            key=lambda s: s["metrics"].get(metric, 0.0),
        )
        return sorted_samples[:n]

    def best_samples(self, metric: str = "recall@5", n: int = 10) -> list[dict[str, Any]]:
        """Return the *n* samples with the highest score on a given metric."""
        sorted_samples = sorted(
            self.per_sample,
            key=lambda s: s["metrics"].get(metric, 0.0),
            reverse=True,
        )
        return sorted_samples[:n]


def quick_eval(
    store,
    dataset_path: str | Path,
    k_values: list[int] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """One-liner: load a dataset, run evaluation, return summary.

    >>> results = quick_eval(store, "eval_data.jsonl")
    >>> print(results["summary"])
    """
    dataset = EvalDataset.load(dataset_path)
    runner = EvalRunner(store, dataset, k_values=k_values)
    return runner.run(top_k=top_k, min_score=min_score)
