"""Grid search over hybrid-retrieval parameters using the eval dataset.

The retriever's fusion weights (alpha/beta), score floor (min_score) and
top-k are all configurable; this script sweeps them against the evaluation
dataset and reports the combination that maximises NDCG@5 / recall@5.

Usage::

    python -m agentflow.knowledge.eval.tune --samples 20
    python -m agentflow.knowledge.eval.tune --dataset path/to/dataset.jsonl

The first pass may call the embedding API; subsequent combinations reuse the
embedding cache (``EMBEDDING_CACHE_ENABLED=true`` by default), so the sweep is
cheap after the first pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from agentflow.knowledge.eval.dataset import EvalDataset
from agentflow.knowledge.eval.runner import EvalRunner
from agentflow.knowledge.store import KnowledgeStore

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "eval_data_current.jsonl"

# Parameter grid — chunk_size/chunk_overlap are ingestion-time and must be
# swept by re-indexing, so they are intentionally left out here.
ALPHAS = (0.5, 0.7, 0.9)
MIN_SCORES = (0.0, 0.05, 0.10, 0.15, 0.20)
TOP_KS = (5, 10)


class TunedStore:
    """Adapter that pins alpha/beta on the shared retriever before searching."""

    def __init__(self, store: KnowledgeStore, alpha: float) -> None:
        self.store = store
        self.alpha = alpha

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None,
               document_ids: list[int] | None = None) -> list[dict[str, Any]]:
        self.store._ensure_retriever()
        if self.store.retriever is not None:
            self.store.retriever.alpha = self.alpha
            self.store.retriever.beta = round(1.0 - self.alpha, 4)
        return self.store.search(
            query, top_k=top_k, min_score=min_score, document_ids=document_ids,
        )


def run_sweep(
    dataset: EvalDataset,
    samples: int = 0,
    max_combos: int = 0,
    k_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run the sweep and return sorted per-combination summaries."""
    store = KnowledgeStore()
    sample_slice = EvalDataset(dataset.samples[:samples]) if samples else dataset
    combos = list(itertools.product(ALPHAS, MIN_SCORES, TOP_KS))
    if max_combos:
        combos = combos[:max_combos]

    results: list[dict[str, Any]] = []
    for alpha, min_score, top_k in combos:
        runner = EvalRunner(
            TunedStore(store, alpha), sample_slice, k_values=k_values,
        )
        summary = runner.run(top_k=top_k, min_score=min_score, verbose=False)["summary"]
        results.append({
            "alpha": alpha,
            "beta": round(1.0 - alpha, 4),
            "min_score": min_score,
            "top_k": top_k,
            "ndcg@5": summary.get("ndcg@5", 0.0),
            "recall@5": summary.get("recall@5", 0.0),
            "mrr": summary.get("mrr", 0.0),
        })
        print(
            f"alpha={alpha:.1f} min_score={min_score:.2f} top_k={top_k:>2}  "
            f"ndcg@5={summary.get('ndcg@5', 0):.3f} recall@5={summary.get('recall@5', 0):.3f}"
        )

    results.sort(key=lambda r: r["ndcg@5"], reverse=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-combos", type=int, default=0, help="0 = full grid")
    parser.add_argument("--top-n", type=int, default=5, help="rows to print in ranking")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    dataset = EvalDataset.load(dataset_path)
    print(f"Loaded {len(dataset.samples)} sample(s) from {dataset_path.name}")

    results = run_sweep(dataset, samples=args.samples, max_combos=args.max_combos)
    if not results:
        print("No combinations evaluated.", file=sys.stderr)
        return 1

    print("\n=== Best combinations (by NDCG@5) ===")
    for row in results[: args.top_n]:
        print(
            f"  alpha={row['alpha']:.1f} beta={row['beta']:.2f} "
            f"min_score={row['min_score']:.2f} top_k={row['top_k']:>2}  "
            f"ndcg@5={row['ndcg@5']:.3f} recall@5={row['recall@5']:.3f} mrr={row['mrr']:.3f}"
        )

    out = Path(__file__).resolve().parent / "data" / "tune_results.json"
    out.write_text(
        json.dumps(
            {"dataset": str(dataset_path), "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
