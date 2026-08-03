"""CLI wrapper for exporting runtime feedback (see agentflow.eval.export_feedback)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow.config.settings import settings
from agentflow.eval.export_feedback import export_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write failure cases to this JSON file.",
    )
    args = parser.parse_args()

    default_path = settings.project_root / "data" / "feedback" / "feedback.jsonl"
    print(f"Feedback file: {default_path}")
    stats = export_feedback(args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.out is not None:
        print(f"Failure cases written to: {args.out}")


if __name__ == "__main__":
    main()
