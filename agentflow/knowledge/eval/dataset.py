"""Evaluation dataset: load, save, validate, and inspect JSONL-based RAG test sets.

Format (one JSON object per line)::

    {"id": "q001", "question": "如何配置数据库？", "relevant_chunk_ids": [1, 2, 3]}
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class EvalDataset:
    """A collection of evaluation queries with ground-truth relevant chunk IDs."""

    def __init__(self, samples: list[dict[str, Any]] | None = None) -> None:
        self.samples: list[dict[str, Any]] = samples or []

    # -- I/O ----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        """Load a JSONL evaluation dataset from disk."""
        path = Path(path)
        samples: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_num}: invalid JSON: {exc}") from exc
                if not cls._validate_sample(sample, line_num):
                    continue
                samples.append(sample)
        return cls(samples)

    def save(self, path: str | Path) -> None:
        """Save the dataset to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # -- Mutation -----------------------------------------------------------

    def add(self, question: str, relevant_chunk_ids: list[int],
            sample_id: str | None = None, **extra) -> str:
        """Append a single sample. Returns the sample id."""
        sid = sample_id or f"q{uuid.uuid4().hex[:8]}"
        sample = {"id": sid, "question": question,
                  "relevant_chunk_ids": relevant_chunk_ids, **extra}
        self.samples.append(sample)
        return sid

    def extend(self, other: EvalDataset) -> None:
        """Merge another dataset into this one."""
        self.samples.extend(other.samples)

    # -- Inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def chunk_id_coverage(self) -> set[int]:
        """Return the union of all referenced relevant chunk IDs."""
        ids: set[int] = set()
        for s in self.samples:
            ids.update(s.get("relevant_chunk_ids", []))
        return ids

    def stats(self) -> dict[str, Any]:
        """Summary statistics about the dataset."""
        n = len(self.samples)
        if n == 0:
            return {"total_samples": 0}
        chunk_counts = [len(s.get("relevant_chunk_ids", [])) for s in self.samples]
        return {
            "total_samples": n,
            "avg_relevant_chunks": sum(chunk_counts) / n,
            "min_relevant_chunks": min(chunk_counts),
            "max_relevant_chunks": max(chunk_counts),
            "total_unique_chunks": len(self.chunk_id_coverage()),
        }

    # -- Validation ---------------------------------------------------------

    @staticmethod
    def _validate_sample(sample: dict[str, Any], line_num: int) -> bool:
        required = ["question", "relevant_chunk_ids"]
        for key in required:
            if key not in sample:
                raise ValueError(
                    f"Line {line_num}: missing required key '{key}'"
                )
        if not isinstance(sample["question"], str) or not sample["question"].strip():
            raise ValueError(f"Line {line_num}: 'question' must be a non-empty string")
        ids = sample["relevant_chunk_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, (int, float)) for i in ids):
            raise ValueError(
                f"Line {line_num}: 'relevant_chunk_ids' must be a list of integers"
            )
        if not ids:
            raise ValueError(f"Line {line_num}: 'relevant_chunk_ids' must not be empty")
        sample["relevant_chunk_ids"] = [int(i) for i in ids]
        if "id" not in sample:
            sample["id"] = f"q{uuid.uuid4().hex[:8]}"
        return True
