"""Capability Registry — maps semantic capabilities to concrete tool names.

The Planner reasons in terms of *capabilities* (what the user needs) and never
mentions tool names.  The Capability Registry is the single source of truth
for resolving capabilities to tools at runtime.

Usage::

    from agentflow.agents.planner.capability import capability_registry

    tool_name = capability_registry.resolve("web.search")       # "search"
    tool_name = capability_registry.resolve("nonexistent")      # None
    for desc in capability_registry.descriptions():
        print(desc)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

# (capability, tool_name, description)
# "tool_name=None" means the capability is recognised but not yet backed by a tool.
_REGISTRY: list[tuple[str, str | None, str]] = [
    ("web.search", "search", "从互联网搜索最新信息"),
    ("knowledge.retrieve", None, "从本地知识库检索文档资料"),
    ("python.execute", "python", "执行 Python 代码并获取运行结果"),
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
