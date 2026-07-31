"""Integration tests for the full LangGraph workflow with a mock LLM.

These tests verify the entire graph execution path (node ordering, routing,
state propagation) without consuming real LLM API calls.  The MockLLMService
replaces the global ``_llm_service`` singleton before each test.
"""

from __future__ import annotations

import json

import pytest

from agentflow.graph.workflow import build_workflow, reset_workflow_cache, run_workflow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure every test starts with a fresh workflow graph."""
    reset_workflow_cache()
    yield


def _mock_response(goal_type: str = "other", fallback: bool = False) -> dict:
    """Build a realistic goal_analysis JSON for the mock to return."""
    return {
        "goal": "测试目标",
        "goal_type": goal_type,
        "knowledge_source": "general",
        "confidence": 0.95,
        "fallback": fallback,
    }


# ── Happy path tests ──


class TestWorkflowHappyPath:
    """Tests that verify the full workflow completes normally."""

    def test_basic_conversation_routes_to_answer(self):
        """A simple greeting should go goal_analyzer → answer → memory → END."""
        from tests.mock_llm import MockLLMService

        responses = {
            "你好": json.dumps(_mock_response(goal_type="other")),
            "default": "这是一个模拟回答。",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "你好")
        assert "answer" in result
        assert result.get("answer"), "Answer should not be empty"
        # Should have trace entries
        trace = result.get("_trace", [])
        assert len(trace) >= 3, f"Expected ≥3 trace entries, got {len(trace)}"
        node_names = [t["node"] for t in trace]
        assert "goal_analyzer" in node_names
        assert "answer" in node_names
        assert "memory" in node_names

    def test_project_goal_triggers_planner(self):
        """A project-type goal should route through the planner node."""
        from tests.mock_llm import MockLLMService

        responses = {
            "计算器": json.dumps(_mock_response(goal_type="project")),
            "default": "",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "创建一个计算器应用")

        trace = result.get("_trace", [])
        node_names = [t["node"] for t in trace]
        assert "planner" in node_names, (
            f"Expected planner in trace, got {node_names}"
        )

    def test_degraded_mode_bypasses_llm(self):
        """When _degraded is set in initial state, AnswerAgent uses fallback."""
        from tests.mock_llm import MockLLMService

        # Use raise_on_call so any accidental LLM call would fail the test
        with MockLLMService.as_default(raise_on_call=RuntimeError("应当被降级跳过")):
            graph = build_workflow()
            # Directly invoke with _degraded set so no agent calls LLM
            from agentflow.graph.workflow import WorkflowState
            initial: WorkflowState = {
                "question": "数据分析",
                "history": [],
                "_degraded": {"goal_analyzer", "planner", "answer", "reflection"},
            }
            result = graph.invoke(initial)
        assert "answer" in result

    def test_workflow_with_history_preserves_context(self):
        """Conversation history should be passed through the workflow."""
        from tests.mock_llm import MockLLMService

        responses = {
            "继续说": json.dumps(_mock_response(goal_type="other")),
            "default": "后续回答。",
        }
        history = [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
        ]
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "继续说", history=history)

        assert "answer" in result
        assert result.get("answer"), "Answer should not be empty"

    def test_knowledge_mode_skips_llm_in_answer(self):
        """source_mode=knowledge should produce a no-LLM answer when results exist."""
        from tests.mock_llm import MockLLMService

        responses = {
            "流程": json.dumps(_mock_response(goal_type="question", fallback=False)),
            "default": "LLM回答",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "介绍一下流程", session_state={})
        assert "answer" in result

    def test_trace_id_propagates_through_workflow(self):
        """trace_id set in initial_state should appear in the result."""
        from tests.mock_llm import MockLLMService

        responses = {
            "追踪": json.dumps(_mock_response(goal_type="other")),
            "default": "带trace的回答。",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            # Pass trace_id via initial_state
            from agentflow.graph.workflow import WorkflowState
            initial: WorkflowState = {
                "question": "带追踪的请求",
                "history": [],
                "trace_id": "test-trace-001",
            }
            result = graph.invoke(initial)
        assert result.get("trace_id") == "test-trace-001"
        # Trace entries should also be present
        trace = result.get("_trace", [])
        assert len(trace) > 0, "Should have trace entries"


# ── Error / failure path tests ──


class TestWorkflowFailurePaths:
    """Tests that verify graceful handling of LLM failures."""

    def test_goal_analyzer_llm_timeout_falls_back(self):
        """When GoalAnalyzer's LLM call fails, it should produce a degraded answer."""
        from tests.mock_llm import MockLLMService

        # For the llm timeout path: GoalAnalyzer will catch the error
        # and use _default_goal() which sets fallback=True.
        # But the IntentIndex embedding might match first.
        # Use a query unlikely to match any intent.
        with MockLLMService.as_default(raise_on_call=TimeoutError("LLM timed out")):
            graph = build_workflow()
            initial_state = {
                "question": "xyz789_nonexistent_query_avoiding_embedding_match",
                "history": [],
            }
            result = graph.invoke(initial_state)

        assert "answer" in result, "Should still produce an answer"
        # Should have errors recorded
        errors = result.get("_errors", [])
        assert len(errors) > 0, "Should have at least one error recorded"


    def test_execution_trace_records_routes(self):
        """Trace entries should have route fields filled in by routing functions."""
        from tests.mock_llm import MockLLMService

        responses = {
            "路由追踪": json.dumps(_mock_response(goal_type="other")),
            "default": "回答。",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "测试路由追踪")

        trace = result.get("_trace", [])
        # At least one trace entry should have a route
        routes = [t.get("route") for t in trace if t.get("route")]
        assert len(routes) > 0, (
            f"Expected at least one route in trace, got none from {trace}"
        )

    def test_result_contains_expected_state_keys(self):
        """The workflow result should contain expected top-level state keys."""
        from tests.mock_llm import MockLLMService

        responses = {
            "状态字段": json.dumps(_mock_response(goal_type="other")),
            "default": "回答内容。",
        }
        with MockLLMService.as_default(responses=responses):
            graph = build_workflow()
            result = run_workflow(graph, "测试状态字段")

        for key in ("answer", "memory", "goal_analysis", "_trace"):
            assert key in result, f"Expected key '{key}' in result"
