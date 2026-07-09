"""OpenAI-compatible function definitions for the Planner.

All schemas are now derived **dynamically** from the ``ToolRegistry``.
The hardcoded ``SEARCH_FUNCTIONS``, ``FILESYSTEM_FUNCTIONS``, etc. lists
have been removed in favor of ``registry.get_all_tool_schemas()``.

Function names follow the ``{tool}__{action}`` convention (double underscore
separator) so the Planner can parse them back into ``Task`` objects
(``tool``, ``action``, ``input``).  Using ``__`` avoids dots in function
names (which many LLM APIs reject via ``^[a-zA-Z0-9_-]+$``) while still
allowing dots in action names (e.g. ``web.search``).

Usage::

    from agentflow.tools.registry import ToolRegistry

    registry = ToolRegistry()
    # ... register tools ...

    schemas = registry.get_all_tool_schemas()
    names = [fn["function"]["name"] for fn in schemas]
"""

from __future__ import annotations

from typing import Any


def get_tool_schemas(registry: Any) -> list[dict[str, Any]]:
    """Return the full list of function definitions from the registry.

    This replaces the old hardcoded ``TOOL_FUNCTIONS`` list.  Callers
    pass a ``ToolRegistry`` instance.
    """
    return registry.get_all_tool_schemas()


def get_function_names(registry: Any) -> list[str]:
    """Return all registered function names (``tool__action`` format)."""
    return [fn["function"]["name"] for fn in registry.get_all_tool_schemas()]


def parse_function_name(name: str) -> tuple[str, str]:
    """Parse ``tool__action`` → (tool, action).

    Uses ``__`` as separator (dots are valid in action names for
    nested dispatch like ``web.search``).

    Examples::
        >>> parse_function_name("filesystem__mkdir")
        ("filesystem", "mkdir")
        >>> parse_function_name("search__search")
        ("search", "search")
    """
    sep = name.find("__")
    if sep == -1:
        return name, ""
    return name[:sep], name[sep + 2:]
