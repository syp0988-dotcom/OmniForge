from fastapi.testclient import TestClient

from agentflow.app.main import app
from agentflow.graph.workflow import build_workflow, run_workflow


def test_workflow_produces_answer() -> None:
    workflow = build_workflow()
    result = run_workflow(workflow, "Analyze AI product manager careers")
    assert "answer" in result
    assert isinstance(result["workflow"], list)
    assert result["answer"].startswith("Processed request")
    assert isinstance(result.get("search_results", []), list)
    if result.get("search_results"):
        first_result = result["search_results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "summary" in first_result


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
