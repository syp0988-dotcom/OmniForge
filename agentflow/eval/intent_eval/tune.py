"""Sweep the embedding intent fast-path thresholds and anchor strategy.

The embedding fast path is the cheapest part of intent analysis, but its
coverage is limited by the confidence floor/ratio and by how representative
the anchors are.  This script measures both on the intent eval dataset:

    python -m agentflow.eval.intent_eval.tune

Metrics per configuration:
  - coverage: matched queries / total
  - hit_rate: correctly matched queries / total  (embedding_hit_rate)
  - accuracy: correctly matched / matched        (embedding_accuracy)

The best configuration prefers accuracy >= 0.98 and maximises hit_rate.
The first pass embeds anchors + queries once; the embedding cache makes every
subsequent configuration local-only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from agentflow.agents.goal_analyzer.intent_index import (
    INTENT_ANCHORS,
    IntentIndex,
)
from agentflow.config import settings as settings_module
from agentflow.eval.intent_eval.dataset import IntentEvalDataset
from agentflow.knowledge.embedder import QwenEmbedder

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "intent_dataset.jsonl"
FLOORS = (0.15, 0.20, 0.25, 0.30, 0.35)
RATIOS = (1.2, 1.5, 1.8, 2.0)


def build_variants() -> dict[str, dict[str, list[str]]]:
    """Return anchor strategies: one soup-vector per label vs multi anchors."""
    single = {
        label: [" ".join(phrases)]
        for label, phrases in INTENT_ANCHORS.items()
    }
    return {"single": single, "multi": INTENT_ANCHORS}


def evaluate(
    index: IntentIndex,
    samples: list[dict[str, Any]],
    floor: float,
    ratio: float,
) -> dict[str, Any]:
    settings = settings_module.settings
    original_floor = settings.intent_min_score_floor
    original_ratio = settings.intent_confidence_ratio
    try:
        settings.intent_min_score_floor = floor
        settings.intent_confidence_ratio = ratio
        matched = correct = total = 0
        mismatches: list[dict[str, Any]] = []
        for sample in samples:
            question = str(sample.get("question", ""))
            expected = str(sample.get("expected_goal_type", ""))
            total += 1
            result = index.match(question)
            if result is None:
                continue
            label, goal_type, score = result
            matched += 1
            ok = goal_type == expected
            if ok:
                correct += 1
            else:
                mismatches.append({
                    "id": sample.get("id"),
                    "question": question[:50],
                    "expected": expected,
                    "got": goal_type,
                    "score": round(float(score), 3),
                })
        return {
            "matched": matched,
            "correct": correct,
            "coverage": matched / total if total else 0.0,
            "hit_rate": correct / total if total else 0.0,
            "accuracy": correct / matched if matched else 0.0,
            "mismatches": mismatches[:10],
        }
    finally:
        settings.intent_min_score_floor = original_floor
        settings.intent_confidence_ratio = original_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--samples", type=int, default=0, help="0 = all samples")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    dataset = IntentEvalDataset.load(dataset_path)
    samples = dataset.samples[: args.samples] if args.samples else dataset.samples
    print(f"Loaded {len(samples)} sample(s) from {dataset_path.name}")

    embedder = QwenEmbedder()
    results: list[dict[str, Any]] = []
    for variant_name, anchors in build_variants().items():
        index = IntentIndex(anchors=anchors, embedder=embedder)
        index._ensure_ready()
        print(f"\n=== variant: {variant_name} ===")
        for floor, ratio in itertools.product(FLOORS, RATIOS):
            row = evaluate(index, samples, floor, ratio)
            entry = {
                "variant": variant_name,
                "floor": floor,
                "ratio": ratio,
                "coverage": round(row["coverage"], 3),
                "hit_rate": round(row["hit_rate"], 3),
                "accuracy": round(row["accuracy"], 3),
                "matched": row["matched"],
                "correct": row["correct"],
                "mismatches": row["mismatches"],
            }
            results.append(entry)
            print(
                f"floor={floor:.2f} ratio={ratio:.1f}  "
                f"coverage={entry['coverage']:.3f} hit={entry['hit_rate']:.3f} "
                f"acc={entry['accuracy']:.3f} ({row['correct']}/{row['matched']})"
            )

    best = max(
        results,
        key=lambda r: (r["accuracy"] >= 0.98, r["hit_rate"], r["accuracy"]),
    )
    print("\n=== BEST ===")
    print(
        f"variant={best['variant']} floor={best['floor']} ratio={best['ratio']}  "
        f"coverage={best['coverage']:.3f} hit={best['hit_rate']:.3f} "
        f"acc={best['accuracy']:.3f}"
    )
    if best["mismatches"]:
        print("remaining mismatches:")
        for m in best["mismatches"]:
            print(f"  {m}")

    out = Path(__file__).resolve().parent / "data" / "intent_tune_results.json"
    out.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "samples": len(samples),
                "best": best,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
