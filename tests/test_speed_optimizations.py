from __future__ import annotations

from agentflow.agents.reflection.agent import ReflectionAgent
from agentflow.graph.executor import Executor
from agentflow.graph.workflow import _make_tool_executor_node
from agentflow.tools.filesystem_tool import FileSystemTool


def test_reflection_rule_completed_project_skips_llm(monkeypatch):
    agent = ReflectionAgent()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("reflection LLM should be skipped for done project tasks")

    monkeypatch.setattr(agent, "_llm_evaluate", fail_if_called)

    state = agent.run({
        "question": "创建两个文件",
        "goal_analysis": {
            "goal": "创建两个文件",
            "goal_type": "project",
        },
        "task_queue": [
            {
                "task_id": "create_a",
                "title": "创建 a.txt",
                "priority": 90,
                "tool": "filesystem",
                "goal": "write_file",
                "status": "done",
                "input": {"action": "write_file", "path": "a.txt", "content": "a"},
            },
            {
                "task_id": "create_b",
                "title": "创建 b.txt",
                "priority": 90,
                "tool": "filesystem",
                "goal": "write_file",
                "status": "done",
                "input": {"action": "write_file", "path": "b.txt", "content": "b"},
            },
        ],
        "tool_results": [
            {"success": True, "tool": "filesystem", "action": "write_file", "result": {"path": "a.txt"}},
            {"success": True, "tool": "filesystem", "action": "write_file", "result": {"path": "b.txt"}},
        ],
    })

    assert state["_reflection_result"] == "done"


def test_tool_executor_parallelizes_independent_file_writes(tmp_path):
    executor = Executor()
    executor.registry.register(FileSystemTool(workspace=str(tmp_path)))
    node = _make_tool_executor_node(executor)

    state = {
        "task_queue": [
            {
                "task_id": "write_a",
                "title": "写 a.txt",
                "priority": 90,
                "tool": "filesystem",
                "goal": "write_file",
                "status": "todo",
                "input": {"action": "write_file", "path": "a.txt", "content": "alpha"},
            },
            {
                "task_id": "write_b",
                "title": "写 b.txt",
                "priority": 90,
                "tool": "filesystem",
                "goal": "write_file",
                "status": "todo",
                "input": {"action": "write_file", "path": "b.txt", "content": "beta"},
            },
        ]
    }

    result = node(state)

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "beta"
    assert [task["status"] for task in result["task_queue"]] == ["done", "done"]
    assert len(result["tool_results"]) == 2
