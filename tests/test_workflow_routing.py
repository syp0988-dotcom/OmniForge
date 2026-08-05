"""Tests for the workflow routing functions."""

from agentflow.graph.workflow import (
    _route_after_executor,
    _route_after_goal_analyzer,
    _route_after_planner,
    _route_after_reflector,
)


def _goal_state(goal_type="question", knowledge_source="general") -> dict:
    return {
        "goal_analysis": {
            "goal_type": goal_type,
            "knowledge_source": knowledge_source,
            "source_mode": "auto",
        },
        "source_mode": "auto",
    }


# -- goal analyzer routing ---------------------------------------------------


def test_direct_answer_types_skip_planner():
    for goal_type in ("other", "translation", "editing"):
        assert _route_after_goal_analyzer(_goal_state(goal_type)) == "answer"


def test_question_general_routes_to_answer():
    assert _route_after_goal_analyzer(_goal_state("question", "general")) == "answer"


def test_question_local_routes_to_knowledge():
    assert _route_after_goal_analyzer(_goal_state("question", "local")) == "knowledge"


def test_project_always_routes_to_planner():
    assert _route_after_goal_analyzer(_goal_state("project", "general")) == "planner"


def test_source_mode_knowledge_forces_retrieval():
    state = _goal_state("other", "general")
    state["source_mode"] = "knowledge"
    assert _route_after_goal_analyzer(state) == "knowledge"


# -- planner routing ---------------------------------------------------------


def test_plan_completed_routes_to_answer():
    state = {"plan": {"goal_completed": True}, "task_queue": []}
    assert _route_after_planner(state) == "answer"


def test_plan_direct_answer_routes_to_answer():
    state = {"plan": {"direct_answer": True}, "task_queue": []}
    assert _route_after_planner(state) == "answer"


def test_plan_with_todo_routes_to_executor():
    state = {
        "plan": {"goal_completed": False},
        "task_queue": [{"task_id": "a", "tool": "filesystem", "status": "todo", "priority": 90}],
    }
    assert _route_after_planner(state) == "tool_executor"


def test_plan_no_todo_incomplete_routes_to_reflector():
    state = {"plan": {"goal_completed": False}, "task_queue": []}
    assert _route_after_planner(state) == "reflector"


# -- executor routing --------------------------------------------------------


def test_executor_failed_task_routes_to_reflector():
    state = {
        "tool_results": [{"success": False, "error": "boom"}],
        "task_queue": [],
    }
    assert _route_after_executor(state) == "reflector"


def test_executor_success_with_next_todo_routes_to_node():
    state = {
        "tool_results": [{"success": True}],
        "task_queue": [{"task_id": "a", "tool": "filesystem", "status": "todo", "priority": 80}],
    }
    assert _route_after_executor(state) == "tool_executor"


def test_executor_success_no_todo_routes_to_reflector():
    state = {"tool_results": [{"success": True}], "task_queue": []}
    assert _route_after_executor(state) == "reflector"


def test_executor_search_task_routes_to_query_rewriter():
    state = {
        "tool_results": [{"success": True}],
        "task_queue": [{"task_id": "s", "tool": "search", "status": "todo", "priority": 90}],
    }
    assert _route_after_executor(state) == "query_rewriter"


# -- reflector routing -------------------------------------------------------


def test_reflector_done_routes_to_answer():
    assert _route_after_reflector({"_reflection_result": "done"}) == "answer"


def test_reflector_replan_under_limit_routes_to_planner():
    assert _route_after_reflector({"_reflection_result": "replan", "_replan_count": 1}) == "planner"


def test_reflector_replan_over_limit_forces_answer():
    assert _route_after_reflector({"_reflection_result": "replan", "_replan_count": 3}) == "answer"


def test_reflector_retry_routes_to_executor():
    assert _route_after_reflector({"_reflection_result": "retry"}) == "tool_executor"


def test_reflector_next_with_todo_routes_to_node():
    state = {
        "_reflection_result": "next",
        "task_queue": [{"task_id": "a", "tool": "python", "status": "todo", "priority": 90}],
    }
    assert _route_after_reflector(state) == "python"


def test_reflector_stuck_forces_answer():
    state = {
        "_reflection_result": "next",
        "task_queue": [],
        "_stuck_rounds": 3,
        "_planner_cycle_count": 1,
    }
    assert _route_after_reflector(state) == "answer"
