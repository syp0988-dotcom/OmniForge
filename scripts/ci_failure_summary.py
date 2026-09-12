"""Summarise a JUnit XML report for CI.

Writes the failing test cases (name + message) to stdout and, when running
inside GitHub Actions, to ``$GITHUB_STEP_SUMMARY`` so the failure is visible
on the run page without downloading logs.

Usage::

    python scripts/ci_failure_summary.py pytest-report.xml
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_MAX_ENTRIES = 40


def collect_failures(report: Path) -> tuple[list[str], int, int]:
    """Return (failure lines, total tests, failed count)."""
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)

    lines: list[str] = []
    total = 0
    failed = 0
    for suite in suites:
        for case in suite.iter("testcase"):
            total += 1
            problems = [c for c in case if c.tag in ("failure", "error")]
            if not problems:
                continue
            failed += 1
            name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            message = (problems[0].get("message") or "").strip().splitlines()
            head = message[0][:200] if message else problems[0].tag
            if len(lines) < _MAX_ENTRIES:
                lines.append(f"- `{name}` - {head}")
            elif len(lines) == _MAX_ENTRIES:
                lines.append(f"- ... ({failed - _MAX_ENTRIES} more)")
    return lines, total, failed


def main(argv: list[str]) -> int:
    report = Path(argv[1] if len(argv) > 1 else "pytest-report.xml")
    if not report.exists():
        print(f"No JUnit report at {report}; nothing to summarise.")
        return 0

    lines, total, failed = collect_failures(report)
    if not failed:
        print(f"All {total} tests passed; nothing to report.")
        return 0

    body = [f"### Test failures ({failed} of {total})", ""] + lines
    print("\n".join(body))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(body) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
