"""Propose file creations from code blocks found in agent responses."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

LANG_TO_EXT: dict[str, str] = {
    "python": "py",
    "py": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "vue": "vue",
    "html": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml",
    "yml": "yml",
    "bash": "sh",
    "shell": "sh",
    "sh": "sh",
    "go": "go",
    "rust": "rs",
    "rs": "rs",
    "java": "java",
    "kotlin": "kt",
    "sql": "sql",
    "xml": "xml",
    "svg": "svg",
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
    "text": "txt",
    "markdown": "md",
    "md": "md",
    "plaintext": "txt",
    "docker": "Dockerfile",
    "diff": "diff",
    "ini": "ini",
    "toml": "toml",
}

CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
MAX_CONTENT_CHARS = 100_000
MIN_LINES = 1


def propose_files(answer_text: str) -> list[dict[str, str]]:
    """Scan *answer_text* for fenced code blocks and return file proposals.

    Each proposal dict contains:
      suggestion_id, filename, language, content, preview
    """
    if not answer_text:
        return []

    proposals: list[dict[str, str]] = []

    for idx, match in enumerate(CODE_BLOCK_RE.finditer(answer_text)):
        lang = match.group(1).strip().lower() or "text"
        content = match.group(2)

        # Skip empty or very short blocks
        line_count = content.count("\n") + 1
        if line_count < MIN_LINES or not content.strip():
            continue

        # Truncate oversized content
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS]

        # Derive filename
        heading = _find_preceding_heading(answer_text, match.start())
        ext = LANG_TO_EXT.get(lang, lang if lang != "text" else "txt")
        filename = _make_filename(heading, ext, idx)

        preview = content[:200].rstrip()

        proposals.append(
            {
                "suggestion_id": uuid.uuid4().hex[:12],
                "filename": filename,
                "language": lang or "text",
                "content": content,
                "preview": preview + ("..." if len(content) > 200 else ""),
            }
        )

    return proposals


def _slugify(text: str) -> str:
    """Convert a heading to a filesystem-safe slug.

    Preserves Chinese characters and common punctuation,
    only removes characters that are dangerous on any filesystem.
    """
    slug = text.strip()
    # Replace path separators and control characters with hyphens
    slug = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", slug)
    # Collapse runs of whitespace/hyphens into single hyphen
    slug = re.sub(r"[\s\-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:120] or "untitled"


def _find_preceding_heading(text: str, block_start: int) -> str | None:
    """Find the nearest markdown heading (# or ##) before the code block."""
    preceding = text[max(0, block_start - 500) : block_start]
    lines = preceding.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            return stripped.lstrip("#").strip()
    return None


def _make_filename(heading: str | None, ext: str, idx: int) -> str:
    if heading:
        slug = _slugify(heading)
        if slug:
            return f"{slug}.{ext}"
    return f"code-snippet-{idx + 1}.{ext}"
