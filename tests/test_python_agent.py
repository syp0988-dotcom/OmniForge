"""Tests for the PythonExecutor agent node.

Regression coverage for backlog P1-PY-1 plus the removed inverted
``ALLOW_UNSAFE_PYTHON_TOOL`` gate (the node used to refuse to run whenever the
flag was False, i.e. by default).
"""

from __future__ import annotations

from agentflow.agents.python.agent import PythonAgent


def _state(question: str, task_queue: list[dict] | None = None) -> dict:
    return {"question": question, "task_queue": task_queue or []}


def test_executes_fenced_code_and_marks_task_done():
    executor = PythonAgent()
    queue = [{"task_id": "t1", "tool": "python", "status": "todo", "goal": "run it"}]

    state = executor.run(
        _state("请运行这段代码\n```python\nprint(1 + 1)\n```", queue)
    )

    assert state["python_result"]["status"] == "ok"
    assert state["python_result"]["stdout"].strip() == "2"
    assert queue[0]["status"] == "done"
    assert state["tool_results"][0]["success"] is True


def test_task_without_code_is_not_marked_failed():
    """A python task with nothing to run is a no-op, not a failure."""
    executor = PythonAgent()
    queue = [{"task_id": "t1", "tool": "python", "status": "todo", "goal": "run"}]

    state = executor.run(_state("帮我看看这个项目", queue))

    assert state["python_result"]["status"] == "no_code"
    assert queue[0]["status"] == "done"
    assert state["tool_results"][0]["success"] is True


def test_plain_sentence_is_not_treated_as_code():
    """An ordinary Chinese sentence parses as a lone identifier — not code."""
    executor = PythonAgent()
    queue = [{"task_id": "t1", "tool": "python", "status": "todo", "goal": "chat"}]

    state = executor.run(_state("帮我看看这个项目", queue))

    assert state["python_result"]["status"] == "no_code"
    assert queue[0]["status"] == "done"


def test_code_attached_to_the_task_is_used():
    """The planner may put the code in the task input; it must be executed."""
    executor = PythonAgent()
    queue = [{
        "task_id": "t1",
        "tool": "python",
        "status": "todo",
        "goal": "run the attached snippet",
        "input": {"action": "execute", "code": "print('from task input')"},
    }]

    state = executor.run(_state("执行计划中的代码", queue))

    assert state["python_result"]["status"] == "ok"
    assert "from task input" in state["python_result"]["stdout"]
    assert queue[0]["status"] == "done"


def test_blocked_code_fails_the_task():
    executor = PythonAgent()
    queue = [{"task_id": "t1", "tool": "python", "status": "todo", "goal": "run"}]

    state = executor.run(_state("```python\nimport os\n```", queue))

    assert state["python_result"]["status"] == "error"
    assert "blocked" in state["python_result"]["stderr"].lower()
    assert queue[0]["status"] == "failed"
