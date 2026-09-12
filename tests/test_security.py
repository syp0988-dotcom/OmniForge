"""Tests for deployment security: optional auth token and CORS origins."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agentflow.app.main import app
from agentflow.config.settings import settings
from agentflow.tools.filesystem_tool import FileSystemTool

client = TestClient(app)


def test_health_stays_open_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "sekrit")
    response = client.get("/health")
    assert response.status_code == 200


def test_api_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "sekrit")
    assert client.get("/sessions").status_code == 401
    assert client.get("/sessions", headers={
        "Authorization": "Bearer wrong",
    }).status_code == 401
    assert client.get("/sessions", headers={
        "Authorization": "Bearer sekrit",
    }).status_code == 200


def test_no_auth_when_token_unset():
    settings.auth_token = ""
    assert client.get("/sessions").status_code == 200


def test_custom_cors_origins(monkeypatch):
    monkeypatch.setattr(
        settings, "cors_origins", "https://app.example.com,https://admin.example.com",
    )
    response = client.get("/health", headers={"Origin": "https://app.example.com"})
    assert response.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_default_cors_rejects_external_origin(monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", "")
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.headers.get("access-control-allow-origin") is None


# -- Filesystem path-safety hardening --------------------------------------


def _tool(tmp_path) -> FileSystemTool:
    return FileSystemTool(workspace=str(tmp_path))


def test_path_traversal_patterns_rejected(tmp_path):
    tool = _tool(tmp_path)
    ok, reason = tool.validate(action="read_file", path="../secret.txt")
    assert not ok and "traversal" in reason.lower()
    ok, reason = tool.validate(action="read_file", path="..\\secret.txt")
    assert not ok and "traversal" in reason.lower()


def test_absolute_path_outside_workspace_rejected(tmp_path):
    tool = _tool(tmp_path)
    outside = os.path.join(tmp_path.parent, "outside.txt")
    ok, _ = tool.validate(action="read_file", path=outside)
    assert not ok


def test_destination_path_validated_for_move(tmp_path):
    tool = _tool(tmp_path)
    # Source inside workspace is fine, but the destination escapes it.
    ok, reason = tool.validate(
        action="move_file",
        src="a.txt",
        dst=os.path.join(tmp_path.parent, "escape.txt"),
    )
    assert not ok and "outside" in reason.lower()


def test_valid_relative_path_accepted(tmp_path):
    tool = _tool(tmp_path)
    ok, reason = tool.validate(action="write_file", path="project/app.py")
    assert ok, reason


def test_nul_byte_and_ads_rejected(tmp_path):
    tool = _tool(tmp_path)
    ok, _ = tool.validate(action="read_file", path="a.txt\x00b.txt")
    assert not ok
    ok, _ = tool.validate(action="read_file", path="a.txt:evil")
    assert not ok


# -- DocxTool path-safety (security review H3) -----------------------------


def _docx_tool(tmp_path):
    from agentflow.tools.docx_tool import DocxTool

    return DocxTool(workspace=str(tmp_path))


def test_docx_traversal_rejected(tmp_path):
    tool = _docx_tool(tmp_path)
    ok, reason = tool.validate(action="read", path="../secret.docx")
    assert not ok and "traversal" in reason.lower()


def test_docx_absolute_path_outside_workspace_rejected(tmp_path):
    tool = _docx_tool(tmp_path)
    outside = os.path.join(tmp_path.parent, "outside.docx")
    ok, reason = tool.validate(action="read", path=outside)
    assert not ok and "outside" in reason.lower()


def test_docx_execute_blocks_absolute_path(tmp_path):
    """A direct execute() call must not read a file outside the workspace."""
    outside = tmp_path.parent / "outside.docx"
    outside.write_bytes(b"not a real docx")

    result = _docx_tool(tmp_path).execute(action="read", path=str(outside))

    assert not result.success
    assert "outside" in (result.error or "").lower()


def test_docx_execute_blocks_traversal_write(tmp_path):
    """Creating a document outside the workspace must be refused."""
    result = _docx_tool(tmp_path).execute(
        action="create", path="../escaped.docx", content="hi",
    )
    assert not result.success
    assert not (tmp_path.parent / "escaped.docx").exists()


# -- Workspace switcher confinement (security review H1) -------------------


def test_workspace_set_rejects_system_directory():
    target = "C:\\Windows" if os.name == "nt" else "/etc"
    if not Path(target).is_dir():
        pytest.skip("system directory not present on this platform")
    response = client.post("/workspace/set", json={"path": target})
    assert response.status_code == 403


def test_workspace_set_rejects_path_outside_allowed_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(allowed))

    response = client.post("/workspace/set", json={"path": str(outside)})

    assert response.status_code == 403


def test_workspace_set_allows_configured_root(tmp_path, monkeypatch):
    from agentflow.api import routes

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(allowed))
    previous = routes._workspace_root
    try:
        response = client.post("/workspace/set", json={"path": str(allowed)})
        assert response.status_code == 200, response.text
        assert routes._current_workspace_root() == allowed.resolve()
    finally:
        routes._set_workspace_root(previous)


def test_default_allowed_roots_include_the_temp_directory(monkeypatch):
    """Regression: on Linux pytest's tmp_path is under /tmp, not $HOME.

    Omitting the temp dir here made every test that sets the workspace to a
    temporary directory fail with 403 on the Ubuntu runner.
    """
    import tempfile

    from agentflow.api import routes

    monkeypatch.setattr(settings, "workspace_allowed_roots", "")
    roots = routes.allowed_workspace_roots()

    assert Path(tempfile.gettempdir()).resolve() in roots
    assert routes.is_allowed_workspace_root(
        Path(tempfile.gettempdir()) / "omniforge-scratch"
    ) is True


# -- API input validation (backlog P1-API-1) -------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "/history?limit=-1",
        "/sessions?limit=-1",
        "/memory?limit=0",
        "/memory/search?query=ab&limit=-5",
        "/executions?limit=100000",
    ],
)
def test_negative_or_excessive_limit_is_rejected(url):
    """A negative limit means "no limit" in SQLite, so it must be refused."""
    assert client.get(url).status_code == 422


def test_symlink_escape_blocked(tmp_path):
    """A symlink inside the workspace pointing outside must not be readable.

    The security property is that the read is refused and no byte of the
    target leaks; the exact wording of the refusal is an implementation
    detail and must not be asserted (it differs per platform).
    """
    outside = tmp_path.parent / "secret_outside.txt"
    outside.write_text("top secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    tool = _tool(tmp_path)
    result = tool.execute(action="read_file", path="link.txt")
    assert not result.success, "symlink escape must be blocked"
    assert (result.error or "").strip(), "a blocked read must explain why"
    assert "top secret" not in str(result.result or ""), "target content leaked"
