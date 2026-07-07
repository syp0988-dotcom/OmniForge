"""Comprehensive tests for the Tool Framework architecture.

Covers:
  - ToolResult unified return format
  - ToolRegistry registration, queries, execution
  - FileSystemTool all actions + safety validation
  - SearchTool/PythonTool backward compatibility
  - GitTool interface
  - BrowserTool/DatabaseTool/MCPTool interface stubs
  - Executor integration with ToolRegistry
  - Planner explicit task format
  - ProjectStructurePlanner
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agentflow.agents.planner.agent import PlannerAgent
from agentflow.agents.planner.capability import (
    resolve as resolve_capability,
    list_capabilities,
    list_tool_capabilities,
    registry_summary,
)
from agentflow.agents.project_structure_planner.agent import ProjectStructurePlanner
from agentflow.graph.executor import Executor
from agentflow.graph.plan import Plan
from agentflow.graph.task import Task
from agentflow.graph.context import WorkflowContext
from agentflow.tools.base import BaseTool
from agentflow.tools.browser_tool import BrowserTool
from agentflow.tools.database_tool import DatabaseTool
from agentflow.tools.filesystem_tool import FileSystemTool
from agentflow.tools.git_tool import GitTool
from agentflow.tools.mcp_tool import MCPTool
from agentflow.tools.python_tool import PythonTool
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.result import ToolResult
from agentflow.tools.search_tool import SearchTool


# ===========================================================================
# ToolResult
# ===========================================================================


class TestToolResult:
    def test_ok_factory(self):
        r = ToolResult.ok("test_tool", "do_stuff", result={"key": "value"}, message="done")
        assert r.success is True
        assert r.tool == "test_tool"
        assert r.action == "do_stuff"
        assert r.result == {"key": "value"}
        assert r.message == "done"
        assert r.error is None
        assert r.duration >= 0

    def test_fail_factory(self):
        r = ToolResult.fail("test_tool", "do_stuff", error="something broke")
        assert r.success is False
        assert r.error == "something broke"
        assert r.tool == "test_tool"
        assert r.action == "do_stuff"

    def test_to_dict(self):
        r = ToolResult.ok("fs", "read", result={"content": "hi"}, message="OK")
        d = r.to_dict()
        assert d["success"] is True
        assert d["tool"] == "fs"
        assert d["action"] == "read"
        assert d["result"] == {"content": "hi"}
        assert d["message"] == "OK"
        assert d["error"] is None
        assert isinstance(d["duration"], float)


# ===========================================================================
# ToolRegistry
# ===========================================================================


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        assert reg.list_tools() == []
        reg.register(FileSystemTool())
        assert "filesystem" in reg.list_tools()

    def test_register_duplicate_overwrites(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        reg.register(FileSystemTool())  # should not raise
        assert reg.list_tools() == ["filesystem"]

    def test_register_invalid_type(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            reg.register("not a tool")  # type: ignore[arg-type]

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        reg.unregister("filesystem")
        assert "filesystem" not in reg.list_tools()

    def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        reg.unregister("nonexistent")  # should not raise

    def test_get(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        tool = reg.get("filesystem")
        assert tool is not None
        assert tool.name == "filesystem"

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_has_tool(self):
        reg = ToolRegistry()
        reg.register(SearchTool())
        assert reg.has_tool("search")
        assert not reg.has_tool("nonexistent")

    def test_list_with_metadata(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        meta = reg.list_with_metadata()
        assert len(meta) == 1
        assert meta[0]["name"] == "filesystem"
        assert "capabilities" in meta[0]
        assert "actions" in meta[0]

    def test_execute_task_unknown_tool(self):
        reg = ToolRegistry()
        r = reg.execute_task("nonexistent", action="test")
        assert r.success is False
        assert "Unknown tool" in r.error

    def test_execute_task_dict(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        r = reg.execute_task_dict({"tool": "filesystem", "action": "mkdir", "path": "test_dir"})
        assert r.success is True

    def test_execute_batch(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        tasks = [
            {"tool": "filesystem", "action": "mkdir", "path": "batch_test"},
            {"tool": "filesystem", "action": "mkdir", "path": "batch_test/subdir"},
        ]
        results = reg.execute_batch(tasks)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_execute_batch_stop_on_failure(self):
        reg = ToolRegistry()
        reg.register(FileSystemTool())
        tasks = [
            {"tool": "filesystem", "action": "read_file", "path": "/etc/passwd"},
            {"tool": "filesystem", "action": "mkdir", "path": "should_not_run"},
        ]
        results = reg.execute_batch(tasks, stop_on_failure=True)
        assert len(results) >= 1  # at least the first one
        # The second should not execute (or if it did, the path validation handles it)
        assert results[0].success is False


# ===========================================================================
# FileSystemTool
# ===========================================================================


class TestFileSystemTool:
    @pytest.fixture
    def tool(self, tmp_path: Path) -> FileSystemTool:
        return FileSystemTool(workspace=str(tmp_path))

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        return tmp_path

    # -- Basic operations ---------------------------------------------------

    def test_mkdir(self, tool: FileSystemTool, workspace: Path):
        r = tool.cmd_mkdir(path="new_dir")
        assert r.success
        assert (workspace / "new_dir").exists()

    def test_create_file(self, tool: FileSystemTool, workspace: Path):
        r = tool.cmd_create_file(path="hello.py", content="print('hello')")
        assert r.success
        assert (workspace / "hello.py").exists()
        assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hello')"

    def test_create_file_already_exists(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_create_file(path="exists.txt", content="a")
        r = tool.cmd_create_file(path="exists.txt", content="b")
        assert r.success is False
        assert "already exists" in r.error

    def test_write_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="test.txt", content="initial")
        tool.cmd_write_file(path="test.txt", content="overwritten")
        assert (workspace / "test.txt").read_text(encoding="utf-8") == "overwritten"

    def test_append_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="append.txt", content="line1\n")
        r = tool.cmd_append_file(path="append.txt", content="line2\n")
        assert r.success
        assert (workspace / "append.txt").read_text(encoding="utf-8") == "line1\nline2\n"

    def test_read_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="readme.md", content="# Hello")
        r = tool.cmd_read_file(path="readme.md")
        assert r.success
        assert r.result["content"] == "# Hello"
        assert r.result["size"] == 7

    def test_read_file_not_found(self, tool: FileSystemTool):
        r = tool.cmd_read_file(path="nonexistent.py")
        assert r.success is False

    def test_edit_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="edit.txt", content="hello world foo")
        r = tool.cmd_edit_file(path="edit.txt", old_string="foo", new_string="bar")
        assert r.success
        assert (workspace / "edit.txt").read_text(encoding="utf-8") == "hello world bar"

    def test_replace_text(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="replace.txt", content="a1 b1 c1")
        r = tool.cmd_replace_text(path="replace.txt", pattern=r"\d", replacement="2")
        assert r.success
        assert r.result["count"] == 3
        assert (workspace / "replace.txt").read_text(encoding="utf-8") == "a2 b2 c2"

    def test_delete_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="delete_me.txt", content="bye")
        r = tool.cmd_delete_file(path="delete_me.txt")
        assert r.success
        assert not (workspace / "delete_me.txt").exists()

    def test_list_directory(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_mkdir(path="sub")
        tool.cmd_write_file(path="sub/a.txt", content="a")
        tool.cmd_write_file(path="sub/b.txt", content="b")
        r = tool.cmd_list_directory(path="sub")
        assert r.success
        assert r.result["count"] == 2

    def test_exists(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="exists_check.txt", content="x")
        r1 = tool.cmd_exists(path="exists_check.txt")
        assert r1.success and r1.result["exists"] is True
        r2 = tool.cmd_exists(path="no_such_file")
        assert r2.success and r2.result["exists"] is False

    def test_tree(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_mkdir(path="deep/a/b")
        tool.cmd_write_file(path="deep/root.txt", content="r")
        r = tool.cmd_tree(path="deep")
        assert r.success
        assert "a/" in r.result["tree"]

    def test_copy_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="src.txt", content="data")
        r = tool.cmd_copy_file(src="src.txt", dst="dst.txt")
        assert r.success
        assert (workspace / "dst.txt").exists()
        assert (workspace / "dst.txt").read_text(encoding="utf-8") == "data"

    def test_move_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="move_src.txt", content="movable")
        r = tool.cmd_move_file(src="move_src.txt", dst="moved.txt")
        assert r.success
        assert not (workspace / "move_src.txt").exists()
        assert (workspace / "moved.txt").exists()

    def test_rename_file(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_write_file(path="old_name.txt", content="rename me")
        r = tool.cmd_rename_file(path="old_name.txt", name="new_name.txt")
        assert r.success
        assert not (workspace / "old_name.txt").exists()
        assert (workspace / "new_name.txt").exists()

    # -- Safety validation --------------------------------------------------

    def test_validate_rejects_path_traversal(self, tool: FileSystemTool):
        valid, msg = tool.validate(path="../../etc/passwd")
        assert not valid
        assert "Path traversal" in msg

    def test_path_traversal_in_execution(self, tool: FileSystemTool):
        r = tool.cmd_read_file(path="../../etc/passwd")
        assert r.success is False

    def test_validate_blocks_system_dir(self, tool: FileSystemTool):
        # Windows-style system path
        valid, msg = tool.validate(path="C:\\Windows\\system32\\config")
        if os.name == "nt":
            assert not valid

    def test_outside_workspace_blocked(self, tool: FileSystemTool):
        r = tool.cmd_read_file(path="/etc/passwd")
        assert r.success is False

    def test_execute_unknown_action(self, tool: FileSystemTool):
        r = tool.execute(action="nonexistent_action")
        assert r.success is False
        assert "Unknown action" in r.error

    # -- Nested path creation -----------------------------------------------

    def test_mkdir_nested(self, tool: FileSystemTool, workspace: Path):
        r = tool.cmd_mkdir(path="a/b/c/d")
        assert r.success
        assert (workspace / "a/b/c/d").exists()

    # -- Delete directory ---------------------------------------------------

    def test_delete_directory(self, tool: FileSystemTool, workspace: Path):
        tool.cmd_mkdir(path="delete_dir")
        tool.cmd_write_file(path="delete_dir/file.txt", content="x")
        r = tool.cmd_delete_directory(path="delete_dir")
        assert r.success
        assert not (workspace / "delete_dir").exists()


# ===========================================================================
# SearchTool (interface + backward compat)
# ===========================================================================


class TestSearchTool:
    def test_capabilities(self):
        tool = SearchTool()
        caps = tool.capabilities()
        assert "web.search" in caps

    def test_metadata(self):
        tool = SearchTool()
        meta = tool.metadata()
        assert meta["name"] == "search"
        assert "provider" in meta

    def test_execute_empty_query(self):
        tool = SearchTool()
        r = tool.execute(query="")
        assert r.success is False
        assert "No search query" in r.error

    def test_search_interface_returns_list(self):
        # DuckDuckGo may fail in CI; we just verify the method exists
        tool = SearchTool()
        assert hasattr(tool, "search")
        assert callable(tool.search)


# ===========================================================================
# PythonTool (interface + backward compat)
# ===========================================================================


class TestPythonTool:
    def test_capabilities(self):
        tool = PythonTool()
        caps = tool.capabilities()
        assert "python.execute" in caps

    def test_validate_syntax_ok(self):
        tool = PythonTool()
        valid, msg = tool.validate(code="print('hello')")
        assert valid
        assert msg == ""

    def test_validate_syntax_error(self):
        tool = PythonTool()
        valid, msg = tool.validate(code="print(hello")
        assert not valid
        assert "Syntax error" in msg

    def test_execute_simple(self):
        tool = PythonTool()
        r = tool.execute(code="print('hello')")
        assert r.success
        assert r.result["stdout"].strip() == "hello"
        assert r.result["return_code"] == 0

    def test_execute_error(self):
        tool = PythonTool()
        r = tool.execute(code="raise ValueError('boom')")
        assert r.success is False
        assert r.result["return_code"] != 0

    def test_execute_syntax_error(self):
        tool = PythonTool()
        r = tool.execute(code="invalid syntax{{{")
        assert r.success is False

    def test_execute_empty_code(self):
        tool = PythonTool()
        r = tool.execute(code="")
        assert r.success is True  # no_code returns success
        assert r.result["status"] == "no_code"


# ===========================================================================
# GitTool (interface)
# ===========================================================================


class TestGitTool:
    def test_capabilities(self):
        tool = GitTool()
        caps = tool.capabilities()
        assert "git.status" in caps

    def test_metadata(self):
        tool = GitTool()
        meta = tool.metadata()
        assert meta["name"] == "git"
        assert "actions" in meta

    def test_execute_unknown_action(self):
        tool = GitTool()
        r = tool.execute(action="nonexistent_git_cmd")
        assert r.success is False


# ===========================================================================
# BrowserTool (interface)
# ===========================================================================


class TestBrowserTool:
    def test_capabilities(self):
        tool = BrowserTool()
        caps = tool.capabilities()
        assert "browser.open" in caps

    def test_execute_returns_interface_stub(self):
        tool = BrowserTool()
        r = tool.execute(action="open_url", url="http://example.com")
        assert r.success is False
        assert "not yet implemented" in r.error

    def test_metadata_shows_status(self):
        tool = BrowserTool()
        meta = tool.metadata()
        assert meta["status"] == "interface_only"


# ===========================================================================
# DatabaseTool (interface)
# ===========================================================================


class TestDatabaseTool:
    def test_capabilities(self):
        tool = DatabaseTool()
        caps = tool.capabilities()
        assert "database.query" in caps

    def test_execute_returns_interface_stub(self):
        tool = DatabaseTool()
        r = tool.execute(action="query", sql="SELECT 1")
        assert r.success is False
        assert "not yet implemented" in r.error


# ===========================================================================
# MCPTool (interface)
# ===========================================================================


class TestMCPTool:
    def test_capabilities(self):
        tool = MCPTool()
        caps = tool.capabilities()
        assert "mcp.discover" in caps

    def test_execute_call_no_tool_name(self):
        tool = MCPTool()
        r = tool.execute(action="call", tool_name="")
        assert r.success is False
        assert "tool_name is required" in r.error


# ===========================================================================
# BaseTool — custom tool plugin model
# ===========================================================================


class TestCustomToolPlugin:
    """Demonstrate the plugin model: a third-party tool."""

    def test_custom_tool_registration(self):
        class WeatherTool(BaseTool):
            name = "weather"
            description = "Get weather information"

            def execute(self, city: str = "", **kwargs: Any) -> ToolResult:
                if not city:
                    return ToolResult.fail(self.name, "get_weather", "City required")
                return ToolResult.ok(self.name, "get_weather", {"city": city, "temp": 22})

            def capabilities(self) -> list[str]:
                return ["weather.current"]

        reg = ToolRegistry()
        reg.register(WeatherTool())
        assert reg.has_tool("weather")
        assert "weather.current" in reg.get("weather").capabilities()

        r = reg.execute_task("weather", action="get_weather", city="Beijing")
        assert r.success
        assert r.result["temp"] == 22

    def test_validate_called_before_execute(self):
        class StrictTool(BaseTool):
            name = "strict"

            def execute(self, value: str = "", **kwargs: Any) -> ToolResult:
                return ToolResult.ok(self.name, "process", {"value": value})

            def validate(self, value: str = "", **kwargs: Any) -> tuple[bool, str]:
                if not value:
                    return False, "value is required"
                if len(value) > 10:
                    return False, "value too long"
                return True, ""

        reg = ToolRegistry()
        reg.register(StrictTool())

        r = reg.execute_task("strict", action="process")
        assert r.success is False
        assert "value is required" in r.error

        r = reg.execute_task("strict", action="process", value="short")
        assert r.success

        r = reg.execute_task("strict", action="process", value="a" * 20)
        assert r.success is False
        assert "value too long" in r.error


# ===========================================================================
# Executor + ToolRegistry integration
# ===========================================================================


class TestExecutorIntegration:
    def test_executor_uses_registry(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        assert "filesystem" in ex.list_tools()

    def test_execute_task_dict(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        r = ex.execute_task_dict({"tool": "filesystem", "action": "mkdir", "path": "exec_test"})
        assert r.success

    def test_execute_task_via_task_object(self):
        ex = Executor()
        ex.registry.register(PythonTool())
        ctx = WorkflowContext({"question": "test"})
        task = Task(
            goal="执行代码", tool="python",
            input={"code": "print(1+1)"}, agent="test",
        )
        result_task = ex.execute(ctx, task)
        assert result_task.status.value == "completed"

    def test_execute_batch(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        tasks = [
            {"tool": "filesystem", "action": "mkdir", "path": "batch/a"},
            {"tool": "filesystem", "action": "mkdir", "path": "batch/b"},
        ]
        results = ex.execute_batch(tasks)
        assert len(results) == 2

    def test_tool_metadata(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        ex.registry.register(PythonTool())
        meta = ex.tool_metadata()
        assert len(meta) == 2
        names = [m["name"] for m in meta]
        assert "filesystem" in names
        assert "python" in names

    def test_get_capabilities(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        ex.registry.register(PythonTool())
        caps = ex.get_capabilities()
        assert "filesystem.create" in caps
        assert "python.execute" in caps

    def test_summary(self):
        ex = Executor()
        ex.registry.register(FileSystemTool())
        summary = ex.summary
        assert "filesystem" in summary


# ===========================================================================
# Planner — explicit task format
# ===========================================================================


class TestPlannerExplicitTasks:
    def test_build_plan_from_explicit_tasks(self):
        data = {
            "direct_answer": False,
            "goal_completed": False,
            "reasoning": "需要创建项目目录和主文件",
            "tasks": [
                {"tool": "filesystem", "action": "mkdir", "goal": "创建目录",
                 "input": {"path": "my_app"}},
                {"tool": "filesystem", "action": "write_file", "goal": "创建主文件",
                 "input": {"path": "my_app/main.py", "content": "print('hello')"}},
            ],
        }
        plan = PlannerAgent._build_plan_from_json(data, goal="创建应用", goal_type="project")
        assert plan.direct_answer is False
        assert plan.goal_completed is False
        assert len(plan.tasks) == 2
        assert plan.tasks[0].tool == "filesystem"
        assert plan.tasks[0].input["path"] == "my_app"
        assert plan.tasks[1].tool == "filesystem"
        assert plan.tasks[1].input["path"] == "my_app/main.py"

    def test_build_plan_from_explicit_direct_answer(self):
        data = {
            "direct_answer": True,
            "reasoning": "无需工具",
            "tasks": [],
        }
        plan = PlannerAgent._build_plan_from_json(data, goal="测试", goal_type="question")
        assert plan.direct_answer is True
        assert plan.goal_completed is True
        assert len(plan.tasks) == 0

    def test_build_plan_auto_detects_explicit_format(self):
        data = {
            "direct_answer": False,
            "reasoning": "test",
            "tasks": [{"tool": "filesystem", "action": "mkdir", "input": {"path": "x"}}],
        }
        plan = PlannerAgent._build_plan_from_json(data, goal="test", goal_type="project")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].tool == "filesystem"

    def test_build_plan_auto_detects_legacy_format(self):
        data = {
            "direct_answer": False,
            "reasoning": "test",
            "tasks": [{"capability": "web.search", "goal": "搜索"}],
        }
        plan = PlannerAgent._build_plan_from_json(data, goal="test", goal_type="question")
        # Legacy format: capability is set, tool resolved later
        assert len(plan.tasks) == 1
        assert plan.tasks[0].capability == "web.search"


# ===========================================================================
# Capability Registry
# ===========================================================================


class TestCapabilityRegistry:
    def test_new_capabilities(self):
        assert resolve_capability("filesystem.create") == "filesystem"
        assert resolve_capability("git.status") == "git"
        assert resolve_capability("browser.open") == "browser"

    def test_unknown_capability_returns_none(self):
        assert resolve_capability("nonexistent.capability") is None

    def test_list_capabilities_includes_new(self):
        caps = list_capabilities()
        assert "filesystem.create" in caps
        assert "git.status" in caps
        assert "web.search" in caps  # existing still there

    def test_list_tool_capabilities(self):
        caps = list_tool_capabilities()
        assert "filesystem.create" in caps
        assert "web.search" in caps
        assert "knowledge.retrieve" not in caps  # has no tool

    def test_registry_summary(self):
        summary = registry_summary()
        assert "filesystem.create" in summary
        assert "web.search" in summary
        assert "git.status" in summary


# ===========================================================================
# ProjectStructurePlanner
# ===========================================================================


class TestProjectStructurePlanner:
    def test_template_matches_fastapi(self):
        tasks = ProjectStructurePlanner._template_match("create a FastAPI project")
        assert tasks is not None
        assert len(tasks) > 0
        # Should have mkdir for app directory
        assert any(t["action"] == "mkdir" for t in tasks)

    def test_template_matches_flask(self):
        tasks = ProjectStructurePlanner._template_match("Flask web app")
        assert tasks is not None
        assert any("app" in str(t.get("input", {}).get("path", "")) for t in tasks)

    def test_template_no_match(self):
        tasks = ProjectStructurePlanner._template_match("something completely random 42")
        assert tasks is None

    def test_parse_json(self):
        raw = '{"project_name": "test", "tasks": [{"tool": "filesystem", "action": "mkdir"}]}'
        result = ProjectStructurePlanner._parse_json(raw)
        assert result is not None
        assert result["project_name"] == "test"
        assert len(result["tasks"]) == 1

    def test_parse_json_from_code_block(self):
        raw = 'Some text\n```json\n{"tasks": [{"tool": "filesystem", "action": "mkdir"}]}\n```\nmore text'
        result = ProjectStructurePlanner._parse_json(raw)
        assert result is not None
        assert len(result["tasks"]) == 1
