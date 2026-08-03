"""Tests for ReflectionAgent structured-output validation."""

import pytest
from pydantic import ValidationError

from agentflow.agents.reflection.agent import ReflectionAgent
from agentflow.agents.reflection.schemas import NewTask, ReflectionOutput, TaskUpdate


def test_valid_output_roundtrip():
    payload = {
        "goal_completed": False,
        "task_updates": [{"task_id": "a", "status": "DONE"}],
        "new_tasks": [
            {
                "task_id": "b",
                "title": "创建 requirements.txt",
                "priority": 90,
                "tool": "filesystem",
                "input": {"action": "write_file", "content": "flask\n"},
            }
        ],
        "remove_tasks": ["c"],
        "reason": "缺少 requirements.txt",
    }
    result = ReflectionAgent._validate_reflection(payload)
    assert result is not None
    assert result["goal_completed"] is False
    assert result["task_updates"][0]["task_id"] == "a"
    assert result["new_tasks"][0]["tool"] == "filesystem"
    assert result["remove_tasks"] == ["c"]


def test_garbage_payload_rejected():
    assert ReflectionAgent._validate_reflection(None) is None
    assert ReflectionAgent._validate_reflection("not a dict") is None
    assert ReflectionAgent._validate_reflection(["list"]) is None
    assert ReflectionAgent._validate_reflection({}) is not None  # all defaults


def test_invalid_entry_types_rejected():
    payload = {
        "task_updates": [{"task_id": "a", "priority": "not-an-int"}],
        "new_tasks": [{"task_id": 123}],  # task_id must be str
    }
    # The whole payload is rejected (-> corrective retry -> rule fallback)
    # rather than letting malformed entries mutate the queue.
    assert ReflectionAgent._validate_reflection(payload) is None


def test_schema_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        NewTask(task_id="x", priority=101)


def test_schema_defaults():
    t = TaskUpdate(task_id="only-id")
    assert t.status is None
    assert t.priority is None
    n = NewTask(task_id="n")
    assert n.tool == "filesystem"
    assert n.priority == 50
    assert n.input == {}


def test_to_dict_shape():
    model = ReflectionOutput(goal_completed=True, reason="done")
    d = model.to_dict()
    assert d["goal_completed"] is True
    assert d["task_updates"] == []
    assert d["new_tasks"] == []
    assert d["need_replan"] is False
    assert d["reason"] == "done"
