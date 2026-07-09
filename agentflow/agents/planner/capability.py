"""Capability Registry — maps semantic capabilities to concrete tool names.

The Planner reasons in terms of *capabilities* (what the user needs) and never
mentions tool names.  The Capability Registry resolves capabilities to tools.

**As of the plugin-framework refactoring, this module derives its data
dynamically from the ToolRegistry.**  The old hardcoded ``_REGISTRY`` list
has been removed.  All capability-to-tool resolution is now:
``{tool}.{action}`` → tool name, derived from each tool's ``actions()``.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_CAPABILITIES: dict[str, str] = {
    "web.search": "search",
    "filesystem.create": "filesystem",
    "filesystem.read": "filesystem",
    "filesystem.write": "filesystem",
    "filesystem.mkdir": "filesystem",
    "filesystem.create_file": "filesystem",
    "filesystem.write_file": "filesystem",
    "git.status": "git",
    "browser.open": "browser",
    "browser.open_url": "browser",
    "python.execute": "python",
}

_DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "web.search": "Search the web",
    "filesystem.create": "Create files or directories",
    "filesystem.read": "Read files",
    "filesystem.write": "Write files",
    "git.status": "Show git status",
    "browser.open": "Open a URL in a browser",
    "python.execute": "Execute Python code",
}


def resolve(capability: str, registry: Any | None = None) -> str | None:
    """Resolve a capability name to a registered tool name.

    Capability format: ``{tool}.{action}`` (e.g. ``"filesystem.mkdir"``).
    Returns the tool name (e.g. ``"filesystem"``) or ``None``.

    When *registry* is provided, validates against registered tools.
    """
    if "." not in capability:
        return None
    tool_name = capability.split(".", 1)[0]
    if registry is not None:
        if registry.has_tool(tool_name):
            return tool_name
        return _DEFAULT_CAPABILITIES.get(capability)
    return _DEFAULT_CAPABILITIES.get(capability)


def get_description(capability: str, registry: Any | None = None) -> str:
    """Return the human-readable description for a capability.

    Derives from tool's ``actions()`` metadata when registry is available.
    """
    if registry is None or "." not in capability:
        return _DEFAULT_DESCRIPTIONS.get(capability, "")
    tool_name, action = capability.split(".", 1)
    tool = registry.get(tool_name)
    if tool is None:
        return ""
    action_def = tool.actions().get(action, {})
    return action_def.get("description", "")


def list_capabilities(registry: Any | None = None) -> list[str]:
    """Return all registered capability names (from ToolRegistry or empty)."""
    if registry is not None:
        return sorted(set(registry.get_all_capabilities()) | set(_DEFAULT_CAPABILITIES))
    return sorted(_DEFAULT_CAPABILITIES)


def list_tool_capabilities(registry: Any | None = None) -> list[str]:
    """Return capabilities that have a concrete tool backing."""
    if registry is not None:
        return [
            cap for cap in registry.get_all_capabilities()
            if registry.has_tool(cap.split(".", 1)[0])
        ]
    return sorted(_DEFAULT_CAPABILITIES)


def list_grouped(registry: Any | None = None) -> dict[str, list[dict[str, str]]]:
    """Return capabilities grouped by domain prefix."""
    groups: dict[str, list[dict[str, str]]] = {}
    caps = list_tool_capabilities(registry) if registry else []
    for cap in caps:
        domain = cap.split(".")[0] if "." in cap else "other"
        if domain not in groups:
            groups[domain] = []
        groups[domain].append({
            "capability": cap,
            "tool": cap.split(".", 1)[0],
            "description": get_description(cap, registry),
        })
    return groups


def registry_summary(registry: Any | None = None) -> str:
    """Return a formatted summary of all capabilities for use in prompts.

    Derives from ToolRegistry when provided; falls back to empty string.
    """
    if registry is not None:
        return registry.get_capability_descriptions()
    return "\n".join(
        f"  - {cap}  -  {_DEFAULT_DESCRIPTIONS.get(cap, '')}"
        for cap in sorted(_DEFAULT_CAPABILITIES)
    )


def tool_actions_summary(registry: Any | None = None) -> str:
    """Return a formatted tool→actions summary for use in prompts.

    Example::

        - filesystem: mkdir, write_file, create_file, read_file, ...
        - git: status, diff, add, commit, ...
    """
    if registry is not None:
        return registry.get_tool_actions_text()
    return ""


# -- Convenience alias -----------------------------------------------------

capability_registry: dict[str, Any] = {
    "resolve": resolve,
    "get_description": get_description,
    "list_capabilities": list_capabilities,
    "list_tool_capabilities": list_tool_capabilities,
    "registry_summary": registry_summary,
    "tool_actions_summary": tool_actions_summary,
}
