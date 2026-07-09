from __future__ import annotations

from agentflow.agents.answer.agent import AnswerAgent
from agentflow.agents.goal_analyzer.agent import _apply_source_mode
from agentflow.graph.workflow import _route_after_goal_analyzer
from agentflow.models.chat import ChatRequest


def test_chat_request_source_mode_defaults_to_auto():
    req = ChatRequest(message="你好")

    assert req.source_mode == "auto"


def test_knowledge_mode_forces_local_for_informational_goal():
    goal = {
        "goal": "解释一下内部规范",
        "goal_type": "other",
        "knowledge_source": "general",
        "expected_outputs": ["answer"],
    }

    result = _apply_source_mode(goal, "knowledge")

    assert result["goal_type"] == "question"
    assert result["knowledge_source"] == "local"
    assert result["source_mode"] == "knowledge"


def test_web_mode_forces_search_for_informational_goal():
    goal = {
        "goal": "今天有什么 AI 新闻",
        "goal_type": "question",
        "knowledge_source": "local",
        "expected_outputs": ["answer"],
    }

    result = _apply_source_mode(goal, "web")

    assert result["goal_type"] == "search"
    assert result["knowledge_source"] == "general"
    assert result["source_mode"] == "web"


def test_source_mode_does_not_reclassify_project_goal():
    goal = {
        "goal": "创建一个 Python 项目",
        "goal_type": "project",
        "knowledge_source": "hybrid",
        "expected_outputs": ["project"],
    }

    result = _apply_source_mode(goal, "web")

    assert result["goal_type"] == "project"
    assert result["knowledge_source"] == "hybrid"
    assert result["source_mode"] == "web"


def test_manual_knowledge_mode_forces_knowledge_route_even_for_project_goal():
    route = _route_after_goal_analyzer({
        "source_mode": "knowledge",
        "goal_analysis": {
            "goal": "use the knowledge base before creating a project",
            "goal_type": "project",
            "knowledge_source": "hybrid",
            "source_mode": "knowledge",
        },
    })

    assert route == "knowledge"


def test_manual_knowledge_mode_reports_empty_retrieval_without_llm():
    state = {
        "source_mode": "knowledge",
        "goal_analysis": {
            "goal": "what does the knowledge base say about deployment",
            "goal_type": "question",
            "knowledge_source": "local",
            "source_mode": "knowledge",
        },
        "knowledge_results": [],
    }

    result = AnswerAgent().run(state)

    assert "没有找到" in result["answer"]
    assert "知识库" in result["answer"]
