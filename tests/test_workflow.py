from fastapi.testclient import TestClient

from agentflow.app.main import app
from agentflow.config.settings import settings
from agentflow.graph.workflow import build_workflow, run_workflow


def test_workflow_produces_answer() -> None:
    workflow = build_workflow()
    result = run_workflow(workflow, "Analyze AI product manager careers")
    assert "answer" in result
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert isinstance(result["workflow"], list)
    assert isinstance(result.get("search_results", []), list)
    if result.get("search_results"):
        first_result = result["search_results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert any(k in first_result for k in ("summary", "snippet", "content"))


def test_health_endpoint() -> None:
    """``/health`` reports 200 plus a component-level contract.

    The overall status depends on whether credentials are configured, so this
    asserts the contract and the database check rather than a machine-specific
    status value.
    """
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "database" in body["checks"]
    assert body["checks"]["database"]["ok"] is True
    assert "llm_config" in body["checks"]


def test_health_endpoint_ok_when_llm_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["llm_config"]["ok"] is True


def test_health_endpoint_degraded_without_llm_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["llm_config"]["ok"] is False
