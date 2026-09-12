"""Summarise a JUnit XML report for CI.

Prints the failing test cases (name + first message line), writes them to
``$GITHUB_STEP_SUMMARY`` when running inside GitHub Actions, and emits workflow
annotations so the failures are visible on the run page without downloading
logs. Output is ASCII-only: the Windows runner's default stdout encoding
(cp1252) cannot encode typographic dashes.

Usage::

    python scripts/ci_failure_summary.py pytest-report.xml
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_MAX_ENTRIES = 40
_MAX_ANNOTATIONS = 10


def collect_failures(report: Path) -> tuple[list[tuple[str, str]], int]:
    """Return ([(test name, first message line)], total tests)."""
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)

    failures: list[tuple[str, str]] = []
    total = 0
    for suite in suites:
        for case in suite.iter("testcase"):
            total += 1
            problems = [c for c in case if c.tag in ("failure", "error")]
            if not problems:
                continue
            name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            message = (problems[0].get("message") or "").strip().splitlines()
            head = message[0][:200] if message else problems[0].tag
            failures.append((name, head))
    return failures, total


def render_summary(failures: list[tuple[str, str]], total: int) -> str:
    lines = [f"### Test failures ({len(failures)} of {total})", ""]
    for name, head in failures[:_MAX_ENTRIES]:
        lines.append(f"- `{name}` - {head}")
    if len(failures) > _MAX_ENTRIES:
        lines.append(f"- ... ({len(failures) - _MAX_ENTRIES} more)")
    return "\n".join(lines)


def emit_annotations(failures: list[tuple[str, str]]) -> None:
    """Surface failures as workflow annotations (visible on the run page)."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for name, head in failures[:_MAX_ANNOTATIONS]:
        message = f"{name} - {head}".replace("\n", " ")[:400]
        print(f"::error title=Test failure::{message}")


def main(argv: list[str]) -> int:
    report = Path(argv[1] if len(argv) > 1 else "pytest-report.xml")
    if not report.exists():
        print(f"No JUnit report at {report}; nothing to summarise.")
        return 0

    failures, total = collect_failures(report)
    if not failures:
        print(f"All {total} tests passed; nothing to report.")
        return 0

    summary = render_summary(failures, total)
    print(summary)
    emit_annotations(failures)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
