"""Tests for the Prometheus-format /metrics endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentflow.app.main import app
from agentflow.utils.metrics import reset

client = TestClient(app)


def test_metrics_reports_http_traffic():
    reset()
    client.get("/health")
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert 'path="/health"' in text
    assert "http_request_duration_seconds_count" in text
    assert "http_request_duration_seconds_sum" in text
