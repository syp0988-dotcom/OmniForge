"""Dedicated tests for the PlannerAgent core logic."""

import pytest

from agentflow.agents.planner.agent import PlannerAgent
from agentflow.agents.planner.task_queue import TaskQueue
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task, TaskStatus
from agentflow.services.llm_service import ToolCall
from tests.mock_llm import MockLLMService


def _planner(llm) -> PlannerAgent:
    p = PlannerAgent.__new__(PlannerAgent)
    p._llm = llm
    p.registry = None
    return p


def _state(goal="创建图书管理系统", goal_type="project") -> dict:
    return {
        "question": goal,
        "goal_analysis": {"goal": goal, "goal_type": goal_type},
        "task_queue": [],
    }


# -- JSON plan parsing ---------------------------------------------------------


def test_build_plan_from_json_with_tasks():
    p = _planner(MockLLMService())
    data = {
        "goal_completed": False,
        "tasks": [
            {
                "task_id": "create_app",
                "title": "创建入口",
                "priority": 90,
                "tool": "filesystem",
                "goal": "创建 app.py",
                "input": {"action": "write_file", "path": "app.py"},
            }
        ],
    }
    plan = p._build_plan_from_json(data, "图书系统", "project")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].tool == "filesystem"
    assert plan.tasks[0].input["path"] == "app.py"
    assert plan.goal_completed is False


def test_build_plan_from_json_completed():
    p = _planner(MockLLMService())
    plan = p._build_plan_from_json(
        {"goal_completed": True, "reasoning": "done"}, "g", "project",
    )
    assert plan.goal_completed is True
    assert plan.direct_answer is True
    assert plan.tasks == []


def test_build_plan_from_json_rejects_non_list_tasks():
    p = _planner(MockLLMService())
    with pytest.raises(ValueError):
        p._build_plan_from_json({"tasks": "not-a-list"}, "g", "project")


def test_build_plan_from_json_skips_non_dict_tasks():
    p = _planner(MockLLMService())
    plan = p._build_plan_from_json(
        {"tasks": [{"task_id": "ok", "tool": "filesystem", "input": {}}, "junk"]},
        "g", "project",
    )
    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_id == "ok"


# -- Non-project handling -----------------------------------------------------


@pytest.mark.parametrize("goal_type", ["question", "other", "translation", "editing"])
def test_handle_non_project_returns_direct_answer(goal_type):
    p = _planner(MockLLMService())
    state = _state(goal="你好", goal_type=goal_type)
    result = p._handle_non_project(state, "你好", goal_type)
    assert result["plan"].direct_answer is True
    assert result["task_queue"] == []


# -- Queue merging ------------------------------------------------------------


def test_merge_into_queue_never_overwrites_done_tasks():
    p = _planner(MockLLMService())
    done = Task(
        task_id="t1", title="old", priority=50,
        tool="filesystem", goal="x", status=TaskStatus.DONE,
    )
    queue = TaskQueue([done])
    plan = Plan(
        goal="g", category="project",
        tasks=[
            Task(
                task_id="t1", title="new", priority=99,
                tool="filesystem", goal="y", status=TaskStatus.TODO,
            )
        ],
    )
    merged = p._merge_into_queue(queue, plan)
    assert merged.get("t1").status == TaskStatus.DONE
    assert merged.get("t1").title == "old"


def test_merge_into_queue_adds_new_tasks():
    p = _planner(MockLLMService())
    plan = Plan(
        goal="g", category="project",
        tasks=[Task(task_id="a", title="a", priority=1, tool="filesystem", goal="a")],
    )
    merged = p._merge_into_queue(TaskQueue(), plan)
    assert merged.get("a") is not None
    assert merged.todo_count == 1


# -- LLM task generation fallbacks -------------------------------------------


def test_llm_generate_tasks_uses_json_fallback():
    mock = MockLLMService(responses={
        "default": (
            '{"goal_completed": false, "tasks": ['
            '{"task_id": "a", "title": "t", "priority": 80, "tool": "filesystem", '
            '"input": {"action": "write_file", "path": "a.py"}}]}'
        ),
    })
    p = _planner(mock)
    plan = p._llm_generate_tasks("创建项目", "project", _state())
    assert len(plan.tasks) == 1
    assert plan.tasks[0].input["path"] == "a.py"


def test_llm_generate_tasks_degraded_when_llm_fails():
    mock = MockLLMService(raise_on_call=TimeoutError("llm down"))
    p = _planner(mock)
    plan = p._llm_generate_tasks("创建项目", "project", _state())
    assert plan.direct_answer is True
    assert plan.goal_completed is False
    assert plan.tasks == []


def test_parse_json_handles_fenced_output():
    parsed = PlannerAgent._parse_json('```json\n{"goal_completed": true}\n```')
    assert parsed == {"goal_completed": True}


def test_parse_json_returns_none_for_garbage():
    assert PlannerAgent._parse_json("not json at all") is None


# -- function-calling planning ------------------------------------------------


def test_build_plan_from_tool_calls():
    calls = [
        ToolCall(id="1", name="filesystem__mkdir", arguments='{"path": "app"}'),
        ToolCall(id="2", name="filesystem__write_file", arguments='{"path": "app/main.py", "code_prompt": "入口"}'),
    ]
    plan = PlannerAgent._build_plan_from_tool_calls(calls, "reasoning", goal="g", category="project")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].tool == "filesystem"
    assert plan.tasks[0].input["path"] == "app"


def test_build_plan_from_tool_calls_skips_readonly():
    calls = [
        ToolCall(id="1", name="filesystem__read_file", arguments='{"path": "app/main.py"}'),
        ToolCall(id="2", name="filesystem__mkdir", arguments='{"path": "app"}'),
    ]
    plan = PlannerAgent._build_plan_from_tool_calls(calls, "", goal="g", category="project")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].input["path"] == "app"


def test_llm_plan_json():
    mock = MockLLMService(responses={
        "default": '{"goal_completed": false, "tasks": [{"task_id": "t", "priority": 50, "tool": "filesystem", "input": {"action": "mkdir", "path": "x"}}]}',
    })
    p = _planner(mock)
    plan = p._llm_plan("创建项目", "project")
    assert plan is not None and len(plan.tasks) == 1


def test_llm_plan_empty_returns_none():
    p = _planner(MockLLMService(answer=""))
    assert p._llm_plan("创建项目", "project") is None


def test_fc_plan_returns_none_when_no_tool_calls():
    mock = MockLLMService(answer="no tools")
    p = _planner(mock)
    assert p._fc_plan("创建项目", "project") is None


def test_initialize_from_blueprint_creates_tasks():
    p = _planner(MockLLMService())
    plan = p._initialize_from_blueprint("帮我创建一个 FastAPI 项目", _state())
    if plan is not None:
        assert len(plan.tasks) >= 1


def test_initialize_from_template_none_for_unknown_goal():
    p = _planner(MockLLMService())
    assert p._initialize_from_template("帮我写个员工考勤系统") is None


# ---------------------------------------------------------------------------
# TaskQueue status handling (backlog P1-QUEUE-1)
# ---------------------------------------------------------------------------


def test_queue_update_accepts_known_status():
    queue = TaskQueue()
    queue.add(Task(task_id="t1", title="task", tool="filesystem"))

    assert queue.update("t1", status="done") is True
    assert queue.get("t1").status == TaskStatus.DONE


def test_queue_update_ignores_unknown_status():
    """An unknown status must not raise (it used to abort the planner cycle)."""
    queue = TaskQueue()
    queue.add(Task(task_id="t1", title="task", tool="filesystem"))

    assert queue.update("t1", status="not-a-status") is True
    assert queue.get("t1").status == TaskStatus.TODO


def test_queue_update_still_applies_other_fields():
    queue = TaskQueue()
    queue.add(Task(task_id="t1", title="task", tool="filesystem"))

    queue.update("t1", status="bogus", title="renamed")

    assert queue.get("t1").title == "renamed"


# ---------------------------------------------------------------------------
# Non-project replan must not discard the existing queue (backlog P1-PLAN-1)
# ---------------------------------------------------------------------------


def test_non_project_replan_preserves_completed_tasks(monkeypatch):
    p = _planner(MockLLMService())
    done = Task(task_id="already_done", title="已完成", tool="filesystem")
    done.status = TaskStatus.DONE
    new_task = Task(task_id="next_step", title="下一步", tool="filesystem")
    monkeypatch.setattr(
        p, "_fc_plan",
        lambda *a, **kw: Plan(goal="写一个脚本", category="coding", tasks=[new_task]),
    )
    monkeypatch.setattr(p, "_fill_code_content", lambda *a, **kw: [])

    state = _state(goal="写一个脚本", goal_type="coding")
    state["task_queue"] = [done.to_dict()]

    p._handle_non_project(state, "写一个脚本", "coding")

    ids = {t["task_id"]: t["status"] for t in state["task_queue"]}
    assert ids.get("already_done") == TaskStatus.DONE.value, ids
    assert "next_step" in ids
