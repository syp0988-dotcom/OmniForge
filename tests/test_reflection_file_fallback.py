from __future__ import annotations

from agentflow.agents.reflection.agent import ReflectionAgent, _generate_stuck_tasks
from agentflow.graph.workflow import _route_after_reflector


def test_stuck_file_creation_generates_python_and_java_snake_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tasks = _generate_stuck_tasks(
        "创建两个文件，一个 python 贪吃蛇，一个 java 贪吃蛇",
        "",
        [],
    )

    paths = [task["input"]["path"] for task in tasks]
    assert paths == [
        "generated_files/snake_games/python_snake.py",
        "generated_files/snake_games/SnakeGame.java",
    ]
    assert all(task["tool"] == "filesystem" for task in tasks)
    assert all(task["input"]["action"] == "write_file" for task in tasks)
    assert "tkinter" in tasks[0]["input"]["content"]
    assert "javax.swing" in tasks[1]["input"]["content"]


def test_empty_project_queue_gets_file_creation_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = ReflectionAgent()

    state = agent.run({
        "question": "创建两个文件，一个 python 贪吃蛇，一个 java 贪吃蛇",
        "goal_analysis": {
            "goal": "创建两个文件，一个 python 贪吃蛇，一个 java 贪吃蛇",
            "goal_type": "project",
        },
        "task_queue": [],
        "tool_results": [],
    })

    assert state["_reflection_result"] == "next"
    queue = state["task_queue"]
    assert len(queue) == 2
    assert {task["status"] for task in queue} == {"todo"}
    assert {task["input"]["path"] for task in queue} == {
        "generated_files/snake_games/python_snake.py",
        "generated_files/snake_games/SnakeGame.java",
    }


def test_reflector_routes_to_answer_after_stuck_round_limit():
    route = _route_after_reflector({
        "_reflection_result": "next",
        "_stuck_rounds": 3,
        "_planner_cycle_count": 0,
        "task_queue": [],
    })

    assert route == "answer"
