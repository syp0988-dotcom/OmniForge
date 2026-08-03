"""Export runtime feedback into eval-ready summaries and failure cases.

The chat endpoints append one JSON object per line to
``data/feedback/feedback.jsonl`` (see ``agentflow.eval.feedback``).  These
helpers turn those records into statistics and a reviewable failure-case file.
The ``scripts/export_feedback.py`` CLI wraps them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentflow.eval.feedback import feedback_path


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all feedback records from the JSONL file."""
    p = path or feedback_path()
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (ValueError, TypeError):
                continue
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate feedback records into compact statistics."""
    outcome_counts = Counter(r.get("outcome", "unknown") for r in records)
    goal_type_counts = Counter(r.get("goal_type", "other") for r in records)

    error_type_counts: Counter[str] = Counter()
    for r in records:
        for e in (r.get("errors") or []):
            if isinstance(e, dict):
                error_type_counts[e.get("type", "unknown")] += 1

    return {
        "total": len(records),
        "by_outcome": dict(outcome_counts),
        "by_goal_type": dict(goal_type_counts),
        "by_error_type": dict(error_type_counts.most_common()),
    }


def export_feedback(out: Path | None = None) -> dict[str, Any]:
    """Summarize records and write failure cases to *out* (JSON)."""
    records = load_records()
    stats = summarize(records)

    failures = [
        r for r in records
        if r.get("outcome") == "failure"
    ]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(
                {"stats": stats, "failure_cases": failures},
                fh,
                ensure_ascii=False,
                indent=2,
            )
    return stats
