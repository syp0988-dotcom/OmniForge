"""Document parsing pipeline supporting PDF, DOCX, TXT, and Markdown.

Upgraded to use structure-aware chunking (see ``agentflow.knowledge.chunking``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from agentflow.knowledge.chunking import chunk_document


def parse_document(
    file_path: str | Path,
    file_type: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Parse a document and return a list of text chunks.

    Uses structure-aware chunking (heading-based for Markdown, definition-
    boundary for code, paragraph-based for others).

    Args:
        file_path: Path to the document file.
        file_type: Optional override (e.g. ``"pdf"``, ``"docx"``, ``"txt"``, ``"md"``).
                   Auto-detected from extension if not provided.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        A list of text chunks from the document.
    """
    path = Path(file_path)
    if file_type is None:
        file_type = path.suffix.lstrip(".").lower()

    raw_text = _read_raw(path, file_type)
    return chunk_document(raw_text, file_type, chunk_size=chunk_size, overlap=chunk_overlap)


def _read_raw(path: Path, file_type: str) -> str:
    """Read the raw text content from a file based on its type."""
    if file_type == "pdf":
        return _read_pdf(path)
    elif file_type == "docx":
        return _read_docx(path)
    elif file_type == "md":
        return _read_markdown(path)
    else:
        return _read_text(path)


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        return _fallback_read(path)


def _read_docx(path: Path) -> str:
    """Extract text from a Word document using python-docx."""
    try:
        from docx import Document

        doc = Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except ImportError:
        return _fallback_read(path)


def _read_markdown(path: Path) -> str:
    """Read a Markdown file, stripping frontmatter."""
    text = path.read_text(encoding="utf-8")
    return _strip_frontmatter(text)


def _read_text(path: Path) -> str:
    """Read a plain text file with encoding auto-detection."""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _fallback_read(path: Path) -> str:
    """Fallback: read file as plain text when a parser library is missing."""
    return _read_text(path)


def _strip_frontmatter(text: str) -> str:
    """Strip YAML/TOML frontmatter delimited by --- or +++."""
    if re.match(r"^(---|\+\+\+)\s*$", text.splitlines()[0] if text else ""):
        parts = re.split(r"^(---|\+\+\+)\s*$", text, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 4:
            return parts[3].strip()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries.

    .. deprecated::
       Use ``agentflow.knowledge.chunking.chunk_by_paragraph`` instead.
    """
    from agentflow.knowledge.chunking import chunk_by_paragraph
    return chunk_by_paragraph(text, chunk_size, overlap)
