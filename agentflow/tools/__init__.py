"""Tool Framework — pluggable, unified tool abstraction layer.

Every tool in the system implements ``BaseTool`` and returns ``ToolResult``.
The ``ToolRegistry`` is the central plugin manager::

    registry = ToolRegistry()
    registry.register(FileSystemTool())
    registry.register(SearchTool())

    result = registry.execute_task("filesystem", action="read_file", path="main.py")
"""

from __future__ import annotations

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

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "FileSystemTool",
    "SearchTool",
    "PythonTool",
    "GitTool",
    "BrowserTool",
    "DatabaseTool",
    "MCPTool",
]
