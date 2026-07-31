"""Tests for deployment security: optional auth token and CORS origins."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentflow.app.main import app
from agentflow.config.settings import settings

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
