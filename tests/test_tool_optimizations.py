"""Tests for tool-calling improvements: schema filtering, token budgets,
one-shot repair, safe parallel selection, and stricter schemas."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentflow.config.settings import settings
from agentflow.graph.workflow import _repair_tool_task, _select_parallel_tasks
from agentflow.services.llm_service import LLMService
from agentflow.tools.browser_tool import BrowserTool
from agentflow.tools.filesystem_tool import FileSystemTool
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.result import ToolResult
from tests.mock_llm import MockLLMService


def test_registry_hides_interface_only_tools_from_planner():
    registry = ToolRegistry()
    registry.register(BrowserTool())  # interface_only placeholder
    registry.register(FileSystemTool())

    names = [fn["function"]["name"] for fn in registry.get_all_tool_schemas()]
    assert any(n.startswith("filesystem__") for n in names)
    assert not any(n.startswith("browser__") for n in names)

    actions_text = registry.get_tool_actions_text()
    assert "filesystem" in actions_text
    assert "browser" not in actions_text

    caps = registry.get_all_capabilities()
    assert any(c.startswith("filesystem.") for c in caps)
    assert not any(c.startswith("browser.") for c in caps)


def test_tool_schemas_reject_extra_properties():
    schema = FileSystemTool().tool_schemas()[0]
    assert schema["function"]["parameters"]["additionalProperties"] is False


def test_llm_service_honours_per_call_max_tokens():
    db = MagicMock()
    db.get_active_model.return_value = None
    service = LLMService(db=db)
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ok"))
    ]
    service._client = fake
    service._model_name = "test-model"

    service.complete(messages=[{"role": "user", "content": "hi"}], max_tokens=4321)
    call_kwargs = fake.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 4321


def test_repair_fixes_failed_tool_arguments_once():
    llm = MockLLMService(responses={"default": '{"input": {"path": "fixed.txt"}}'})
    task = {
        "tool": "filesystem",
        "action": "write_file",
        "input": {"path": "broken/../path"},
    }
    result = ToolResult.fail(
        tool="filesystem", action="write_file",
        error="Path traversal blocked",
    )

    repaired = _repair_tool_task(task, result, llm=llm)
    assert repaired is not None
    assert repaired["input"]["path"] == "fixed.txt"
    assert repaired["_repair_count"] == 1

    # One-shot guard: a second attempt must not spend another LLM call.
    assert _repair_tool_task(repaired, result, llm=llm) is None


def test_repair_skips_unrepairable_tools():
    llm = MockLLMService(responses={"default": "{}"})
    task = {"tool": "python", "input": {"code": "print(1)"}}
    result = ToolResult.fail(tool="python", action="execute", error="boom")
    assert _repair_tool_task(task, result, llm=llm) is None


def test_repair_respects_config_flag(monkeypatch):
    monkeypatch.setattr(settings, "tool_call_repair_enabled", False)
    task = {"tool": "filesystem", "input": {"path": "x"}}
    result = ToolResult.fail(tool="filesystem", action="write_file", error="boom")
    assert _repair_tool_task(task, result, llm=MockLLMService(responses={"default": "{}"})) is None


def test_parallel_selection_rules():
    queue = [
        {"tool": "filesystem", "status": "todo", "priority": 90,
         "input": {"action": "write_file", "path": "a.py"}},
        {"tool": "filesystem", "status": "todo", "priority": 90,
         "input": {"action": "write_file", "path": "b.py"}},
        {"tool": "filesystem", "status": "todo", "priority": 90,
         "input": {"action": "write_file", "path": "a.py"}},  # dup path → skip
        {"tool": "search", "status": "todo", "priority": 90,
         "input": {"query": "news"}},
        {"tool": "python", "status": "todo", "priority": 90,
         "input": {"code": "print(1)"}},
        {"tool": "filesystem", "status": "done", "priority": 90,
         "input": {"action": "write_file", "path": "c.py"}},
    ]
    selected = _select_parallel_tasks(queue)
    paths = [t["input"].get("path") for t in selected if t["tool"] == "filesystem"]
    assert paths == ["a.py", "b.py"]
    assert any(t["tool"] == "search" for t in selected)
    assert not any(t["tool"] == "python" for t in selected)
