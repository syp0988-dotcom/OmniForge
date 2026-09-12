"""Shared path-safety helpers for workspace-scoped tools.

Both ``FileSystemTool`` and ``DocxTool`` must refuse to touch anything outside
their workspace.  Keeping the rules in one place means a fix (or a new bypass
pattern) only has to be applied once, and the two tools cannot drift apart.

Rules enforced here:

* NUL bytes are rejected (they truncate paths in C APIs).
* NTFS alternate-data-stream syntax (``file.txt:evil``) is rejected; a colon is
  only accepted as a Windows drive letter (``C:\\...``).
* The resolved path must stay inside the workspace, after symlink resolution —
  so ``../`` traversal and symlinks pointing outside are both refused.
"""

from __future__ import annotations

import re
from pathlib import Path

# Directories that are never acceptable as a tool target.
DANGEROUS_DIRS: set[str] = {
    "/etc", "/var", "/sys", "/proc", "/dev", "/boot", "/bin", "/sbin",
    "/lib", "/lib64", "/usr", "/opt", "/root",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\System32", "C:\\Windows\\System32",
}

# Patterns that look like path traversal attempts.
TRAVERSAL_PATTERNS = re.compile(r"(\.\./|\.\.\\)")


def looks_like_traversal(raw: str) -> bool:
    """Return True when *raw* contains an explicit ``../`` style segment."""
    return bool(TRAVERSAL_PATTERNS.search(raw))


def has_unsafe_path_syntax(raw: str) -> bool:
    """Return True for NUL bytes and alternate-data-stream style paths."""
    if "\x00" in raw:
        return True
    colon = raw.find(":")
    if colon == -1:
        return False
    # Only a drive-letter colon ("C:\...") is allowed.
    return raw.count(":") > 1 or colon != 1 or not raw[0].isalpha()


def resolve_in_workspace(raw: str, workspace: Path) -> Path | None:
    """Resolve *raw* against *workspace*; return ``None`` when it escapes.

    ``workspace`` is expected to be already resolved.
    """
    if not raw:
        return None
    if has_unsafe_path_syntax(raw):
        return None

    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return None
    return resolved


def is_in_dangerous_dir(path: Path) -> str | None:
    """Return the offending system directory prefix, or ``None``."""
    resolved = str(path)
    for dangerous in DANGEROUS_DIRS:
        if resolved.startswith(dangerous):
            return dangerous
    return None
