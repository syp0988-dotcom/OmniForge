from __future__ import annotations

from agentflow.agents.reflection.agent import ReflectionAgent, _generate_stuck_tasks
from agentflow.graph.workflow import _route_after_reflector


def test_stuck_file_creation_generates_language_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tasks = _generate_stuck_tasks(
        "创建两个文件，一个 python 脚本，一个 java 脚本",
        "",
        [],
    )

    paths = [task["input"]["path"] for task in tasks]
    assert paths == [
        "generated_files/main.py",
        "generated_files/Main.java",
    ]
    assert all(task["tool"] == "filesystem" for task in tasks)
    assert all(task["input"]["action"] == "write_file" for task in tasks)
    assert "Hello from" in tasks[0]["input"]["content"]
    assert "public class Main" in tasks[1]["input"]["content"]


def test_empty_project_queue_gets_file_creation_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = ReflectionAgent()

    state = agent.run({
        "question": "创建两个文件，一个 python 脚本，一个 java 脚本",
        "goal_analysis": {
            "goal": "创建两个文件，一个 python 脚本，一个 java 脚本",
            "goal_type": "project",
        },
        "task_queue": [],
        "tool_results": [],
    })

    assert state["_reflection_result"] == "next"
    queue = state["task_queue"]
    assert len(queue) == 2
    assert {task["status"] for task in queue} == {"todo"}
    paths = {task["input"]["path"] for task in queue}
    assert len(paths) == 2
    assert all(p.startswith("generated_files/") for p in paths)
    assert any(p.endswith("main.py") for p in paths)
    assert any(p.endswith("Main.java") for p in paths)


def test_reflector_routes_to_answer_after_stuck_round_limit():
    route = _route_after_reflector({
        "_reflection_result": "next",
        "_stuck_rounds": 3,
        "_planner_cycle_count": 0,
        "task_queue": [],
    })

    assert route == "answer"
