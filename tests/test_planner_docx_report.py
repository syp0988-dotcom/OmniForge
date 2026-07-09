from __future__ import annotations

from agentflow.agents.planner.agent import PlannerAgent


def test_docx_report_goal_uses_docx_create_before_direct_answer():
    planner = PlannerAgent()

    state = {
        "question": "整理成docx报告",
        "goal_analysis": {
            "goal": "整理成docx报告",
            "goal_type": "question",
            "knowledge_source": "general",
        },
        "history": [
            {"role": "user", "content": "介绍 OmniForge 亮点"},
            {"role": "assistant", "content": "OmniForge 的亮点包括多 Agent 工作流、知识库检索和工具执行。"},
        ],
    }

    result = planner.run(state)

    queue = result["task_queue"]
    assert len(queue) == 1
    task = queue[0]
    assert task["tool"] == "docx"
    assert task["input"]["action"] == "create"
    assert task["input"]["path"] == "OmniForge报告.docx"
    assert "多 Agent 工作流" in task["input"]["content"]
    assert result["plan"].direct_answer is False


def test_python_java_snake_goal_creates_two_files():
    planner = PlannerAgent()

    result = planner.run({
        "question": "创建两个文件一个python贪吃蛇一个java贪吃蛇",
        "goal_analysis": {
            "goal": "创建两个文件一个python贪吃蛇一个java贪吃蛇",
            "goal_type": "project",
            "knowledge_source": "general",
        },
    })

    queue = result["task_queue"]
    paths = [task["input"]["path"] for task in queue]

    assert paths == ["snake_game/python_snake.py", "snake_game/JavaSnake.java"]
    assert all(task["tool"] == "filesystem" for task in queue)
    assert all(task["input"]["action"] == "write_file" for task in queue)
    assert "Python Snake demo" in queue[0]["input"]["content"]
    assert "Java Snake demo" in queue[1]["input"]["content"]
