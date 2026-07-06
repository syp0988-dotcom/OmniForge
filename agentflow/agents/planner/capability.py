"""Capability Registry — maps semantic capabilities to concrete tool names.

The Planner reasons in terms of *capabilities* (what the user needs) and never
mentions tool names.  The Capability Registry is the single source of truth
for resolving capabilities to tools at runtime.

New capabilities are added here; the corresponding tools are registered
separately in the ``ToolRegistry``.  This separation means:
  - Adding a new capability = adding one row here + creating a Tool class
  - The Planner never hardcodes tool names
  - The ToolRegistry never knows about capabilities
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------
# (capability, tool_name, description)
# "tool_name=None" means capability recognised but not yet backed by a tool.
_REGISTRY: list[tuple[str, str | None, str]] = [
    # -- Existing capabilities ------------------------------------------------
    ("web.search", "search", "从互联网搜索最新信息"),
    ("knowledge.retrieve", None, "从本地知识库检索文档资料"),
    ("python.execute", "python", "执行 Python 代码并获取运行结果"),
    # -- Filesystem capabilities ----------------------------------------------
    ("filesystem.create", "filesystem", "创建文件和目录"),
    ("filesystem.read", "filesystem", "读取文件内容"),
    ("filesystem.edit", "filesystem", "编辑和修改文件"),
    ("filesystem.delete", "filesystem", "删除文件"),
    ("filesystem.list", "filesystem", "列出目录内容"),
    ("filesystem.tree", "filesystem", "生成目录树结构"),
    # -- Git capabilities ----------------------------------------------------
    ("git.status", "git", "查看 Git 仓库状态"),
    ("git.diff", "git", "查看文件差异"),
    ("git.add", "git", "暂存文件修改"),
    ("git.commit", "git", "提交暂存区变更"),
    ("git.checkout", "git", "切换分支或恢复文件"),
    ("git.branch", "git", "管理 Git 分支"),
    ("git.log", "git", "查看提交历史"),
    # -- Browser capabilities (interface) ------------------------------------
    ("browser.open", "browser", "在浏览器中打开 URL"),
    ("browser.extract", "browser", "提取页面文本内容"),
    ("browser.screenshot", "browser", "截取页面截图"),
    ("browser.interact", "browser", "与页面元素交互（点击、输入、滚动）"),
    # -- Database capabilities (interface) -----------------------------------
    ("database.query", "database", "执行数据库查询"),
    ("database.insert", "database", "插入数据"),
    ("database.update", "database", "更新数据"),
    ("database.delete", "database", "删除数据"),
    # -- MCP capabilities (interface) ----------------------------------------
    ("mcp.discover", "mcp", "发现 MCP 服务器的可用工具"),
    ("mcp.execute", "mcp", "调用 MCP 工具的指定操作"),
]

# Derived lookup maps (built once)
_CAPABILITY_TO_TOOL: dict[str, str | None] = {
    cap: tool for cap, tool, _ in _REGISTRY
}
_CAPABILITY_DESCRIPTION: dict[str, str] = {
    cap: desc for cap, _, desc in _REGISTRY
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(capability: str) -> str | None:
    """Resolve a capability name to a registered tool name.

    Returns ``None`` when the capability is recognised but not yet backed
    by a concrete tool, or when the capability is unknown.
    """
    return _CAPABILITY_TO_TOOL.get(capability)


def get_description(capability: str) -> str:
    """Return the human-readable description for a capability."""
    return _CAPABILITY_DESCRIPTION.get(capability, "")


def list_capabilities() -> list[str]:
    """Return all registered capability names."""
    return list(_CAPABILITY_TO_TOOL.keys())


def list_tool_capabilities() -> list[str]:
    """Return only capabilities that have a concrete tool backing."""
    return [cap for cap, tool in _CAPABILITY_TO_TOOL.items() if tool is not None]


def list_grouped() -> dict[str, list[dict[str, str]]]:
    """Return capabilities grouped by domain prefix (e.g. ``web``, ``filesystem``)."""
    groups: dict[str, list[dict[str, str]]] = {}
    for cap, tool, desc in _REGISTRY:
        domain = cap.split(".")[0] if "." in cap else "other"
        if domain not in groups:
            groups[domain] = []
        groups[domain].append({
            "capability": cap,
            "tool": tool or "",
            "description": desc,
        })
    return groups


def registry_summary() -> str:
    """Return a formatted summary of all capabilities for use in prompts."""
    lines: list[str] = []
    for cap, _, desc in _REGISTRY:
        lines.append(f"  - {cap}  —  {desc}")
    return "\n".join(lines)


# -- Convenience alias -----------------------------------------------------

capability_registry: dict[str, Any] = {
    "resolve": resolve,
    "get_description": get_description,
    "list_capabilities": list_capabilities,
    "list_tool_capabilities": list_tool_capabilities,
    "registry_summary": registry_summary,
}
