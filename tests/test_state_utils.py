"""Tests for the shared workflow-state accessors."""

from agentflow.graph.state_utils import (
    get_goal,
    get_goal_analysis,
    get_goal_type,
    get_knowledge_source,
    get_plan_field,
    get_plan_tasks,
    get_source_mode,
    is_plan_completed,
    plan_flag,
)


class _Plan:
    """Minimal object-shaped plan used to exercise dict/object fallbacks."""

    def __init__(self, **kwargs) -> None:
        self.goal_completed = kwargs.get("goal_completed", False)
        self.direct_answer = kwargs.get("direct_answer", False)
        self.intent = kwargs.get("intent", "")
        self.tasks = kwargs.get("tasks", [])


def test_goal_accessors_from_goal_analysis():
    state = {
        "question": "raw question",
        "goal_analysis": {
            "goal": "帮我创建项目",
            "goal_type": "project",
            "knowledge_source": "local",
            "source_mode": "knowledge",
        },
    }
    assert get_goal(state) == "帮我创建项目"
    assert get_goal_type(state) == "project"
    assert get_knowledge_source(state) == "local"
    assert get_source_mode(state) == "knowledge"
    assert get_goal_analysis(state) == state["goal_analysis"]


def test_goal_accessors_fall_back_to_legacy_fields():
    state = {"question": "raw question", "category": "coding"}
    assert get_goal(state) == "raw question"
    assert get_goal_type(state) == "coding"
    assert get_knowledge_source(state) == "hybrid"
    assert get_source_mode(state) == "auto"
    assert get_goal_analysis(state) == {}


def test_goal_accessors_handle_non_dict_goal_analysis():
    state = {"question": "q", "goal_analysis": "not-a-dict"}
    assert get_goal(state) == "q"
    assert get_goal_type(state) == "other"


def test_plan_flag_and_completion_dict():
    state = {"plan": {"goal_completed": False, "direct_answer": True}}
    assert is_plan_completed(state)
    assert not plan_flag(state["plan"], "goal_completed")
    assert plan_flag(state["plan"], "direct_answer")


def test_plan_flag_and_completion_object():
    state = {"plan": _Plan(goal_completed=True)}
    assert is_plan_completed(state)
    state = {"plan": _Plan()}
    assert not is_plan_completed(state)


def test_plan_flag_none():
    assert not plan_flag(None, "goal_completed")
    assert not is_plan_completed({})


def test_get_plan_field_and_tasks():
    state = {"plan": {"intent": "project", "tasks": [{"task_id": "a"}]}}
    assert get_plan_field(state, "intent") == "project"
    assert get_plan_tasks(state) == [{"task_id": "a"}]
    assert get_plan_field(state, "missing", "fallback") == "fallback"


def test_get_plan_tasks_normalizes_objects():
    from agentflow.graph.task import Task, TaskStatus

    task = Task(
        task_id="t1",
        title="task",
        priority=10,
        tool="filesystem",
        goal="write",
        input={"action": "write_file"},
        status=TaskStatus.DONE,
    )
    state = {"plan": _Plan(tasks=[task])}
    tasks = get_plan_tasks(state)
    assert tasks == [task.to_dict()]
    assert tasks[0]["status"] == "done"
