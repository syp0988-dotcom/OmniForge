"""Shared pytest configuration.

Runtime paths are redirected into a session-scoped temporary directory *before*
the application package is imported, so running the suite never touches the
developer's real database, uploaded knowledge sources, generated files or logs.

This closes backlog item P0-CI-1: previously a full run rewrote
``agentflow/database/agentflow.db`` and files under ``knowledge_files/``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_SESSION_RUNTIME = Path(tempfile.mkdtemp(prefix="omniforge-tests-"))

# Set before any test module imports ``agentflow``.  ``setdefault`` keeps a
# developer's explicit override (e.g. a scratch directory) intact.
os.environ.setdefault("DATABASE_PATH", str(_SESSION_RUNTIME / "agentflow.db"))
os.environ.setdefault("OUTPUTS_DIR", str(_SESSION_RUNTIME / "outputs"))
os.environ.setdefault(
    "KNOWLEDGE_FILES_DIR", str(_SESSION_RUNTIME / "knowledge_files"),
)
os.environ.setdefault("LOGS_DIR", str(_SESSION_RUNTIME / "logs"))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_session_runtime():
    """Remove the redirected runtime directory once the session finishes."""
    yield
    shutil.rmtree(_SESSION_RUNTIME, ignore_errors=True)
