"""Find and optionally delete duplicate files under ``knowledge_files/``.

The upload API now dedupes by content hash, but older uploads may have left
identical copies on disk (e.g. the same PDF uploaded multiple times).  This
script reports those duplicates and can remove the redundant copies:

    python scripts/dedupe_knowledge_files.py            # dry run (report only)
    python scripts/dedupe_knowledge_files.py --delete   # remove duplicates

Only files whose SHA-256 matches another file are candidates; the first
occurrence (by modification time) is always kept.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge_files"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete duplicate copies (default: dry run)",
    )
    parser.add_argument(
        "--dir",
        default=str(KNOWLEDGE_DIR),
        help="directory to scan (default: project knowledge_files/)",
    )
    args = parser.parse_args()

    scan_dir = Path(args.dir).resolve()
    if not scan_dir.is_dir():
        print(f"Directory not found: {scan_dir}", file=sys.stderr)
        return 1

    by_hash: dict[str, list[Path]] = {}
    for path in sorted(scan_dir.iterdir()):
        if not path.is_file():
            continue
        by_hash.setdefault(sha256(path), []).append(path)

    duplicates: list[Path] = []
    for paths in by_hash.values():
        if len(paths) > 1:
            # Keep the oldest file, mark the rest as duplicates.
            paths.sort(key=lambda p: p.stat().st_mtime)
            duplicates.extend(paths[1:])

    if not duplicates:
        print("No duplicate files found.")
        return 0

    print(f"Found {len(duplicates)} duplicate file(s):")
    for path in duplicates:
        print(f"  {path}")

    if not args.delete:
        print("\nRe-run with --delete to remove them.")
        return 0

    for path in duplicates:
        try:
            path.unlink()
            print(f"Deleted {path}")
        except OSError as exc:
            print(f"Failed to delete {path}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
