"""Dedicated tests for the ReflectionAgent evaluation logic."""

from agentflow.agents.reflection.agent import ReflectionAgent
from agentflow.agents.reflection.agent import _generate_stuck_tasks
from agentflow.agents.planner.task_queue import TaskQueue
from agentflow.graph.task import TaskStatus


def _queue(tasks: list[dict]) -> TaskQueue:
    return TaskQueue.from_dict_list(tasks)


def _task(task_id="t1", status="todo", priority=50, tool="filesystem") -> dict:
    return {
        "task_id": task_id,
        "title": task_id,
        "priority": priority,
        "tool": tool,
        "goal": f"创建 {task_id}",
        "status": status,
        "input": {"action": "write_file", "path": f"{task_id}.py"},
    }


def _agent() -> ReflectionAgent:
    return ReflectionAgent.__new__(ReflectionAgent)


# -- run(): empty queue --------------------------------------------------------


def test_run_empty_queue_returns_done():
    agent = _agent()
    state = {
        "question": "你好",
        "goal_analysis": {"goal": "你好", "goal_type": "other"},
        "task_queue": [],
        "tool_results": [],
    }
    result = agent.run(state)
    assert result["_reflection_result"] == "done"


# -- rule evaluation ----------------------------------------------------------


def test_rule_evaluation_all_success_no_replan():
    agent = _agent()
    queue = _queue([_task("a", status="done"), _task("b", status="done")])
    results = [
        {"success": True, "action": "创建 a", "result": {"path": "a.py"}},
        {"success": True, "action": "创建 b", "result": {"path": "b.py"}},
    ]
    eval_result = agent._rule_evaluation(queue, results, "创建项目", "project")
    assert eval_result["goal_completed"] is False or eval_result["goal_completed"] is True
    assert eval_result.get("need_replan") is False


def test_rule_evaluation_failure_sets_replan():
    agent = _agent()
    queue = _queue([_task("a", status="failed")])
    results = [{"success": False, "action": "创建 a", "error": "boom"}]
    eval_result = agent._rule_evaluation(queue, results, "创建项目", "project")
    assert eval_result.get("need_replan") is True
    # A FAILED task must be reported as a task update.
    assert any(
        u["task_id"] == "a" and u["status"] == "FAILED"
        for u in eval_result.get("task_updates", [])
    )


def test_rule_evaluation_no_results_not_complete():
    agent = _agent()
    queue = _queue([_task("a", status="todo")])
    eval_result = agent._rule_evaluation(queue, [], "创建项目", "project")
    assert eval_result.get("goal_completed") is False


def test_rule_evaluation_maps_each_result_to_its_own_task():
    """Substring matching used to collapse every result onto the first task."""
    agent = _agent()
    queue = _queue([
        _task("write_file_a", status="done"),
        _task("write_file_b", status="done"),
    ])
    # ``write_file`` is a substring of both goals ("创建 write_file_a" / _b).
    results = [
        {"success": True, "action": "write_file", "result": {"path": "a.py"}},
        {"success": True, "action": "write_file", "result": {"path": "b.py"}},
    ]

    eval_result = agent._rule_evaluation(queue, results, "创建项目", "project")

    updated = {u["task_id"] for u in eval_result.get("task_updates", [])}
    assert len(updated) == 2, f"both tasks must be updated, got {updated}"


def test_rule_evaluation_prefers_exact_task_id():
    agent = _agent()
    queue = _queue([
        _task("first", status="done"),
        _task("second", status="done"),
    ])
    results = [
        {"success": False, "task_id": "second", "action": "创建 first",
         "error": "boom"},
    ]

    eval_result = agent._rule_evaluation(queue, results, "创建项目", "project")

    updates = {u["task_id"]: u["status"] for u in eval_result.get("task_updates", [])}
    assert updates.get("second") == "FAILED"


# -- applying reflection updates ----------------------------------------------


def test_apply_reflection_updates_queue():
    agent = _agent()
    queue = _queue([_task("a", status="todo"), _task("b", status="todo")])
    reflection = {
        "task_updates": [{"task_id": "a", "status": "DONE"}],
        "new_tasks": [{
            "task_id": "c", "title": "c", "priority": 90,
            "tool": "filesystem",
            "input": {"action": "write_file", "path": "c.py", "content": "x"},
        }],
        "remove_tasks": ["b"],
    }
    agent._apply_reflection(queue, reflection)
    assert queue.get("a").status == TaskStatus.DONE
    assert queue.get("b") is None
    assert queue.get("c") is not None


def test_apply_reflection_skips_write_without_content():
    agent = _agent()
    queue = _queue([])
    reflection = {
        "new_tasks": [{
            "task_id": "nocontent", "title": "x", "priority": 50,
            "tool": "filesystem",
            "input": {"action": "write_file", "path": "x.py"},
        }],
    }
    agent._apply_reflection(queue, reflection)
    assert queue.get("nocontent") is None


# -- evaluate: rule-first, LLM only for failures/stuck -----------------------


def test_evaluate_skips_llm_when_rule_complete():
    class _CountingLLM:
        calls = 0

        def complete(self, *a, **k):
            type(self).calls += 1
            return ""

    agent = _agent()
    agent._llm = _CountingLLM()
    queue = _queue([_task("a", status="done")])
    results = [{"success": True, "action": "创建 a"}]
    # For non-project goals the rule path decides directly; LLM should not fire.
    eval_result = agent._evaluate(
        "你好", "other", queue, results,
        degraded_set=set(),
    )
    assert isinstance(eval_result, dict)
    assert _CountingLLM.calls == 0


# -- LLM evaluation paths -----------------------------------------------------


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0) if self.responses else ""


def test_llm_evaluate_valid_json():
    agent = _agent()
    agent._llm = _FakeLLM(['{"goal_completed": true, "task_updates": [], "new_tasks": [], "remove_tasks": [], "reason": "done"}'])
    queue = _queue([_task("a", status="failed")])
    results = [{"success": False, "action": "创建 a", "error": "boom"}]
    out = agent._llm_evaluate("创建项目", "project", queue, results)
    assert out is not None
    assert out["goal_completed"] is True


def test_llm_evaluate_invalid_then_retries_once():
    agent = _agent()
    agent._llm = _FakeLLM([
        "not json at all",
        '{"goal_completed": false, "new_tasks": [{"task_id": "b", "tool": "filesystem", "input": {"action": "mkdir", "path": "b"}}]}',
    ])
    queue = _queue([_task("a", status="failed")])
    results = [{"success": False, "action": "创建 a", "error": "boom"}]
    out = agent._llm_evaluate("创建项目", "project", queue, results)
    assert agent._llm.calls == 2
    assert out is not None
    assert any(t["task_id"] == "b" for t in out["new_tasks"])


def test_generate_stuck_tasks_returns_configs():
    out = _generate_stuck_tasks("创建项目", "项目", [])
    assert isinstance(out, list)


def test_validate_reflection_rejects_malformed():
    assert ReflectionAgent._validate_reflection("junk") is None
    assert ReflectionAgent._validate_reflection(None) is None
