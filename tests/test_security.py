"""Tests for deployment security: optional auth token and CORS origins."""

from __future__ import annotations

import os

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


def test_symlink_escape_blocked(tmp_path):
    """A symlink inside the workspace pointing outside must not be readable."""
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
    assert "outside" in (result.error or "").lower()
