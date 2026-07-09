"""MCPTool — Model Context Protocol tool adapter (interface / placeholder).

MCP provides a standard protocol for connecting LLMs to external tools,
data sources, and services.  This tool adapter enables the Agent Framework
to discover and invoke MCP servers — for example:

  - Filesystem MCP server (file operations via MCP)
  - GitHub MCP server (repository management)
  - Playwright MCP server (browser automation)
  - Slack MCP server (messaging)
  - Notion MCP server (knowledge management)

The current implementation is an **interface placeholder**.  When an MCP
client library is integrated, this tool discovers available MCP tools at
runtime and delegates ``execute()`` to the appropriate MCP server.

Reference: https://modelcontextprotocol.io
"""

from __future__ import annotations

from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult


class MCPTool(BaseTool):
    """Model Context Protocol (MCP) tool adapter (interface placeholder).

    Discovers and invokes tools exposed by connected MCP servers.
    """

    name = "mcp"
    description = (
        "Model Context Protocol adapter — connects to MCP servers "
        "(Filesystem, GitHub, Playwright, Slack, Notion, …) via a unified interface"
    )

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url
        # In a real implementation, this would hold an MCP client session
        # and a list of discovered tools from the server.
        self._discovered_tools: list[dict[str, Any]] = []

    def actions(self) -> dict[str, dict]:
        return {
            "discover": {
                "description": "[接口预留] 发现 MCP 服务器的可用工具",
                "parameters": {},
                "required": [],
            },
            "call": {
                "description": "[接口预留] 调用 MCP 服务器的指定工具",
                "parameters": {
                    "tool_name": {"type": "string", "description": "MCP 工具名称"},
                    "arguments": {"type": "object", "description": "工具参数字典"},
                },
                "required": ["tool_name"],
            },
            "list_tools": {
                "description": "[接口预留] 列出所有已发现的 MCP 工具",
                "parameters": {},
                "required": [],
            },
        }

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["server_url"] = self.server_url
        base["status"] = "interface_only"
        base["message"] = (
            "This is an interface placeholder. "
            "Integrate an MCP client library (e.g. ``mcp`` Python package) "
            "to connect to MCP servers and discover tools at runtime."
        )
        base["discovered_tools"] = self._discovered_tools
        return base

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        handler = _ACTION_MAP.get(action)
        if handler is None:
            return ToolResult.fail(
                self.name, action or "execute",
                f"Unknown MCP action '{action}'. "
                f"Available: {', '.join(sorted(_ACTION_MAP))}",
            )
        return handler(self, **kwargs)

    # ==================================================================
    # Interface stubs — implement with real MCP client
    # ==================================================================

    def cmd_discover(self, **kwargs: Any) -> ToolResult:
        """Discover tools from the connected MCP server."""
        return ToolResult.fail(
            self.name, "discover",
            "MCPTool not yet implemented — integrate an MCP client library",
        )

    def cmd_call(self, tool_name: str = "", arguments: dict | None = None, **kwargs: Any) -> ToolResult:
        """Call a specific tool on the MCP server."""
        if not tool_name:
            return ToolResult.fail(self.name, "call", "tool_name is required")
        return ToolResult.fail(
            self.name, "call",
            "MCPTool not yet implemented",
        )

    def cmd_list_tools(self, **kwargs: Any) -> ToolResult:
        """List all discovered MCP tools."""
        return ToolResult.ok(
            self.name, "list_tools",
            {"tools": self._discovered_tools, "count": len(self._discovered_tools)},
            f"{len(self._discovered_tools)} MCP tool(s) discovered",
        )


# -- Action dispatch map --------------------------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "discover": MCPTool.cmd_discover,
    "call": MCPTool.cmd_call,
    "list_tools": MCPTool.cmd_list_tools,
}
