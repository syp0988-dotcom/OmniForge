"""API routes: shared store accessors and workspace helpers.

Endpoint implementations live in focused sub-modules (``chat``,
``knowledge``, ``sessions``, ``files_workspace``, ``models``, ``memory``,
``executions``, ``system``) and are mounted by ``agentflow.app.main``.

This module keeps the shared store globals (so tests can swap the store via
``set_store``) and the workspace path helpers used by those routers.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from agentflow.config.settings import settings
from agentflow.database.sqlite import SQLiteStore
from agentflow.knowledge.store import KnowledgeStore
from agentflow.utils.logging import build_logger

logger = build_logger("api")

# -- Lazy-initialised store accessors (decoupled for testability) -------------


async def _read_upload_limited(
    file: UploadFile, max_bytes: int | None = None,
) -> bytes:
    """Read an UploadFile fully while enforcing the configured size limit."""
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {limit} byte upload limit",
            )
    return bytes(content)
# Replace module-level globals with lazy accessors so tests can swap
# implementations by calling set_store(mock_store) before routes are hit.

_store: SQLiteStore | None = None
_knowledge_store: KnowledgeStore | None = None
_workspace_root: Path | None = None


def get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store


def get_knowledge_store() -> KnowledgeStore:
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore(db=get_store())
    return _knowledge_store


def set_store(store: SQLiteStore) -> None:
    """Override the global store (for testing / DI)."""
    global _store
    _store = store


def set_knowledge_store(ks: KnowledgeStore) -> None:
    """Override the global knowledge store (for testing / DI)."""
    global _knowledge_store
    _knowledge_store = ks

# Directory for uploaded document files
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Permanent storage for knowledge base original files (for click-to-preview)
KNOWLEDGE_FILES_DIR = Path(__file__).resolve().parents[2] / "knowledge_files"
KNOWLEDGE_FILES_DIR.mkdir(exist_ok=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when ``path`` is inside ``parent`` after resolution."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _current_workspace_root() -> Path:
    """Return the active workspace root, defaulting to the app project root."""
    return (_workspace_root or Path(__file__).resolve().parents[2]).resolve()


def _resolve_workspace_child(path: str | None, *, default: Path | None = None) -> Path:
    """Resolve a path and require it to stay inside the active workspace root."""
    root = _current_workspace_root()
    target = (default or root) if not path else Path(path).resolve()
    if not _is_relative_to(target, root):
        raise HTTPException(status_code=403, detail="Path is outside the active workspace")
    return target


# -- Agent introspection ---------------------------------------------------




def _set_workspace_root(path: Path) -> None:
    """Update the shared workspace root (used by the workspace router)."""
    global _workspace_root
    _workspace_root = path
