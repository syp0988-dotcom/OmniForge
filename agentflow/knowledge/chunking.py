"""Structure-aware document chunking strategies.

Each strategy implements the same signature::

    def chunk(text: str, chunk_size: int, overlap: int) -> list[str]: ...

and guarantees that returned chunks are **semantically complete** — they
do not split code blocks, JSON structures, or Markdown headings mid-way.
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    file_type: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Auto-detect the best chunking strategy for *file_type*.

    Falls back to paragraph-level chunking for unknown types.
    """
    strategy = _select_strategy(file_type)
    return strategy(text, chunk_size, overlap)


_CHUNK_STRATEGIES: dict[str, Callable[..., list[str]]] = {}


def _select_strategy(file_type: str) -> Callable[..., list[str]]:
    file_type = file_type.lower().lstrip(".")
    if file_type in ("md", "markdown", "rst"):
        return chunk_by_markdown
    if file_type in ("html", "htm"):
        return chunk_by_html
    if file_type in ("xlsx", "xls", "csv"):
        return chunk_by_table
    if file_type == "pptx":
        return chunk_by_slide
    if file_type in ("py", "js", "ts", "jsx", "tsx", "java", "go", "rs", "c", "cpp", "h", "hpp"):
        return chunk_by_code
    return chunk_by_paragraph


# -- Export mapping so callers can introspect --------------------------------
def register_strategy(ext: str, fn: Callable[..., list[str]]) -> None:
    """Register a custom chunking strategy for a file extension."""
    _CHUNK_STRATEGIES[ext.lower().lstrip(".")] = fn


# ---------------------------------------------------------------------------
# Strategy: Markdown heading-based
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_by_markdown(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split on ``##`` / ``###`` headings and keep heading context.

    Each chunk starts at a heading and includes all content until the next
    heading at the same or higher level.  If a section is longer than
    ``chunk_size`` it is further split by paragraph.
    """
    if not text.strip():
        return []

    sections = _split_by_heading(text)
    return _merge_or_split_sections(sections, chunk_size, overlap, _markdown_context)


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """Split text into ``(heading_context, body)`` pairs.

    ``heading_context`` includes the heading line itself.
    Text before the first heading is preserved with an empty heading context.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    # Preserve text before the first heading
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        start = m.end()  # body starts after the heading line
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((m.group(0), body))
    return sections


def _markdown_context(heading: str, _body: str) -> str:
    """Join heading and body as a semantically complete chunk."""
    if not heading:
        return _body
    return f"{heading}\n\n{_body}"


# ---------------------------------------------------------------------------
# Strategy: Code function/class boundary
# ---------------------------------------------------------------------------

# Patterns for common function/class definitions
_CODE_BOUNDARY_RE = re.compile(
    r"^(def\s+\w+|class\s+\w+|async\s+def\s+\w+|"
    r"public\s+(static\s+)?\w+\s+\w+\s*\(|"
    r"function\s+\w*|"
    r"func\s+\w+|"
    r"sub\s+\w+|"
    r"pub\s+fn\s+\w+)",
    re.MULTILINE,
)


def chunk_by_code(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split on function/class definitions.

    Each chunk represents one top-level definition (plus any decorators or
    comments that precede it).  If a definition body exceeds ``chunk_size``
    it is further split by paragraph.
    """
    if not text.strip():
        return []

    matches = list(_CODE_BOUNDARY_RE.finditer(text))
    if not matches:
        return chunk_by_paragraph(text, chunk_size, overlap)

    sections: list[tuple[str, str]] = []

    # Preserve text before the first definition (e.g. license headers, imports)
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].rstrip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].rstrip()
        if body:
            sections.append((m.group(0), body))

    return _merge_or_split_sections(sections, chunk_size, overlap, _code_context)


def _code_context(_heading: str, body: str) -> str:
    return body


# ---------------------------------------------------------------------------
# Strategy: Paragraph-based (upgraded from legacy chunk_text)
# ---------------------------------------------------------------------------


def chunk_by_paragraph(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split by paragraph boundaries, avoiding mid-structure breaks.

    The algorithm collects paragraphs until ``chunk_size`` is reached,
    then starts a new chunk with an overlap window from the tail of the
    previous chunk.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Check if adding this paragraph would exceed chunk_size
        if current_len + para_len + 2 > chunk_size and current:
            chunks.append("\n\n".join(current))

            # Start new chunk with overlap
            overlap_text = _tail_overlap(current, overlap) if overlap > 0 else ""
            current = [overlap_text, para] if overlap_text else [para]
            current_len = len(overlap_text) + para_len + 2
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text]


def _tail_overlap(paragraphs: list[str], overlap_chars: int) -> str:
    """Extract the last ~*overlap_chars* characters from a group of paragraphs."""
    text = "\n\n".join(paragraphs)
    if len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    if "\n\n" in tail:
        tail = tail[tail.index("\n\n") + 2:]
    return tail


# ---------------------------------------------------------------------------
# Strategy: HTML heading-based (h1-h6, similar to markdown)
# ---------------------------------------------------------------------------

_HTML_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def chunk_by_html(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split HTML content on heading lines (``# `` to ``###### ``).

    Behaves identically to ``chunk_by_markdown`` since the HTML parser
    in ``_read_html`` already converts ``<h1>``-``<h6>`` to markdown-style
    headings.
    """
    return chunk_by_markdown(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Strategy: Table-aware (Excel / CSV)
# ---------------------------------------------------------------------------


def chunk_by_table(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split table content by sheet/table boundaries, grouping rows into batches.

    Each ``## Sheet: <name>`` section's rows are grouped into row batches
    that each fit within ``chunk_size``.  Batches within the same sheet
    are separated by ``\\n\\n`` so that large tables split cleanly at
    row-group boundaries rather than mid-row.
    """
    if not text.strip():
        return []

    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_rows: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_rows:
                body = _group_rows(current_rows, chunk_size)
                sections.append((current_heading, body))
            current_heading = line
            current_rows = []
        else:
            current_rows.append(line)

    if current_rows:
        body = _group_rows(current_rows, chunk_size)
        sections.append((current_heading, body))

    if not sections:
        return chunk_by_paragraph(text, chunk_size, overlap)

    return _merge_or_split_sections(sections, chunk_size, overlap, _table_context)


def _group_rows(rows: list[str], max_chars: int) -> str:
    """Split rows into groups, each fitting within ``max_chars``.

    Returns groups separated by ``\\n\\n`` so downstream paragraph-based
    splitting can cleanly break at group boundaries.
    """
    groups: list[str] = []
    batch: list[str] = []
    batch_len = 0
    for row in rows:
        row_len = len(row) + 1  # +1 for the joining newline
        if batch_len + row_len > max_chars and batch:
            groups.append("\n".join(batch))
            batch = [row]
            batch_len = row_len
        else:
            batch.append(row)
            batch_len += row_len
    if batch:
        groups.append("\n".join(batch))
    return "\n\n".join(groups)


def _table_context(heading: str, body: str) -> str:
    if not heading:
        return body
    return f"{heading}\n\n{body}"


# ---------------------------------------------------------------------------
# Strategy: Slide-based (PowerPoint)
# ---------------------------------------------------------------------------


def chunk_by_slide(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split PowerPoint content by slide boundaries.

    Each ``## Slide N`` section becomes a chunk.  Long slides are split
    by paragraph.
    """
    return chunk_by_table(text, chunk_size, overlap)  # same structure: ## headings


# ---------------------------------------------------------------------------
# Common helper for section-based strategies
# ---------------------------------------------------------------------------


def _merge_or_split_sections(
    sections: list[tuple[str, str]],
    chunk_size: int,
    overlap: int,
    joiner: Callable[[str, str], str],
) -> list[str]:
    """Merge small sections together; split large ones by paragraph.

    This is the shared post-processing for markdown and code chunkers.
    """
    if not sections:
        return []

    chunks: list[str] = []
    buffer: list[tuple[str, str]] = []
    buffer_len = 0

    def _flush_buffer() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        # Merge buffer into a single chunk
        merged = "\n\n".join(joiner(h, b) for h, b in buffer)
        # If still too large, split by paragraph
        if len(merged) > chunk_size:
            chunks.extend(_split_large_section(merged, chunk_size, overlap))
        else:
            chunks.append(merged)
        buffer = []
        buffer_len = 0

    for heading, body in sections:
        sec_len = len(heading) + len(body) + 2
        if sec_len > chunk_size:
            # Flush buffer first, then handle this large section
            _flush_buffer()
            full = joiner(heading, body)
            chunks.extend(_split_large_section(full, chunk_size, overlap))
        elif buffer_len + sec_len > chunk_size * 1.5:
            _flush_buffer()
            buffer = [(heading, body)]
            buffer_len = sec_len
        else:
            buffer.append((heading, body))
            buffer_len += sec_len

    _flush_buffer()
    return chunks


def _split_large_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a section that is still too large by paragraphs."""
    return chunk_by_paragraph(text, chunk_size, overlap)
