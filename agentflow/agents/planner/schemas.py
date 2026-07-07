"""OpenAI-compatible function definitions for the Planner.

Each entry in ``TOOL_FUNCTIONS`` follows the ``tools`` parameter format
of the Chat Completions API so the Planner can pass them directly to
``LLMService.complete_with_tools()``.

Function names follow the ``{tool}.{action}`` convention so the Planner
can parse them back into ``Task`` objects (``tool``, ``action``, ``input``).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


def _req(*props: str) -> list[str]:
    """Shortcut: required property names."""
    return list(props)


def _fn(
    name: str,
    description: str,
    properties: dict[str, dict],
    required: list[str] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one function definition dict.

    Args:
        extra_params: Extra keys merged into the ``parameters`` dict
            (e.g. ``{"additionalProperties": True}``).
    """
    params: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        **({"required": required} if required else {}),
    }
    if extra_params:
        params.update(extra_params)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

SEARCH_FUNCTIONS = [
    _fn(
        "search.web.search",
        "Search the web for real-time information (news, weather, prices, etc.)",
        properties={
            "query": {**_STR, "description": "The search query, concise and specific"},
        },
        required=["query"],
    ),
]

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

PYTHON_FUNCTIONS = [
    _fn(
        "python.execute",
        "Execute Python code in a sandboxed subprocess with a 30 s timeout",
        properties={
            "code": {**_STR, "description": "The Python source code to execute"},
        },
        required=["code"],
    ),
]

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

FILESYSTEM_FUNCTIONS = [
    _fn(
        "filesystem.mkdir",
        "Create a directory (parents=True, exist_ok=True)",
        properties={"path": {**_STR, "description": "Directory path relative to workspace"}},
        required=["path"],
    ),
    _fn(
        "filesystem.create_file",
        "Create a new file (fails if it already exists)",
        properties={
            "path": {**_STR, "description": "File path relative to workspace"},
            "content": {**_STR, "description": "Initial file content"},
        },
        required=["path"],
    ),
    _fn(
        "filesystem.write_file",
        "Write content to a file (overwrites if exists)",
        properties={
            "path": {**_STR, "description": "File path relative to workspace"},
            "content": {**_STR, "description": "Content to write"},
        },
        required=["path", "content"],
    ),
    _fn(
        "filesystem.append_file",
        "Append content to an existing file",
        properties={
            "path": {**_STR, "description": "File path relative to workspace"},
            "content": {**_STR, "description": "Content to append"},
        },
        required=["path", "content"],
    ),
    _fn(
        "filesystem.read_file",
        "Read the full contents of a file",
        properties={"path": {**_STR, "description": "File path relative to workspace"}},
        required=["path"],
    ),
    _fn(
        "filesystem.edit_file",
        "Replace the FIRST occurrence of old_string with new_string in a file",
        properties={
            "path": {**_STR, "description": "File path relative to workspace"},
            "old_string": {**_STR, "description": "Text to search for"},
            "new_string": {**_STR, "description": "Replacement text"},
        },
        required=["path", "old_string", "new_string"],
    ),
    _fn(
        "filesystem.replace_text",
        "Replace ALL occurrences matching a regex pattern in a file",
        properties={
            "path": {**_STR, "description": "File path relative to workspace"},
            "pattern": {**_STR, "description": "Regular expression pattern"},
            "replacement": {**_STR, "description": "Replacement text"},
        },
        required=["path", "pattern", "replacement"],
    ),
    _fn(
        "filesystem.delete_file",
        "Delete a single file (not a directory)",
        properties={"path": {**_STR, "description": "File path relative to workspace"}},
        required=["path"],
    ),
    _fn(
        "filesystem.delete_directory",
        "Delete a directory and all its contents (USE WITH CAUTION)",
        properties={"path": {**_STR, "description": "Directory path relative to workspace"}},
        required=["path"],
    ),
    _fn(
        "filesystem.move_file",
        "Move a file from src to dst",
        properties={
            "src": {**_STR, "description": "Source file path"},
            "dst": {**_STR, "description": "Destination file path"},
        },
        required=["src", "dst"],
    ),
    _fn(
        "filesystem.copy_file",
        "Copy a file from src to dst",
        properties={
            "src": {**_STR, "description": "Source file path"},
            "dst": {**_STR, "description": "Destination file path"},
        },
        required=["src", "dst"],
    ),
    _fn(
        "filesystem.rename_file",
        "Rename a file or directory within the same parent",
        properties={
            "path": {**_STR, "description": "Current file/directory path"},
            "name": {**_STR, "description": "New name (not a full path)"},
        },
        required=["path", "name"],
    ),
    _fn(
        "filesystem.exists",
        "Check whether a file or directory exists in the workspace",
        properties={"path": {**_STR, "description": "Path to check"}},
        required=["path"],
    ),
    _fn(
        "filesystem.list_directory",
        "List the contents of a directory (non-recursive)",
        properties={
            "path": {**_STR, "description": "Directory path (default: workspace root)"},
        },
    ),
    _fn(
        "filesystem.tree",
        "Generate a recursive directory tree (text format)",
        properties={
            "path": {**_STR, "description": "Directory path (default: workspace root)"},
            "max_depth": {**_INT, "description": "Maximum recursion depth", "default": 3},
            "show_hidden": {**_BOOL, "description": "Include hidden files", "default": False},
        },
    ),
]

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

GIT_FUNCTIONS = [
    _fn(
        "git.status",
        "Show the working tree status (modified, staged, untracked files)",
        properties={},
    ),
    _fn(
        "git.diff",
        "Show unstaged changes (diff of working tree vs index)",
        properties={},
    ),
    _fn(
        "git.add",
        "Stage file(s) for commit",
        properties={
            "files": {
                **_STR,
                "description": "Space-separated file paths or '.' for all",
            },
        },
        required=["files"],
    ),
    _fn(
        "git.commit",
        "Create a new commit with the given message",
        properties={
            "message": {**_STR, "description": "Commit message"},
        },
        required=["message"],
    ),
    _fn(
        "git.checkout",
        "Switch branches or restore files",
        properties={
            "branch": {**_STR, "description": "Branch name to switch to"},
        },
        required=["branch"],
    ),
    _fn(
        "git.branch",
        "List branches or create a new branch",
        properties={
            "name": {**_STR, "description": "New branch name (omit to list)"},
        },
    ),
    _fn(
        "git.log",
        "Show commit history",
        properties={
            "max_count": {**_INT, "description": "Max commits to show", "default": 10},
        },
    ),
    _fn(
        "git.show",
        "Show the details of a specific commit",
        properties={
            "hash": {**_STR, "description": "Commit hash to show"},
        },
        required=["hash"],
    ),
]

# ---------------------------------------------------------------------------
# Browser  (interface only — still under development)
# ---------------------------------------------------------------------------

BROWSER_FUNCTIONS = [
    _fn(
        "browser.open_url",
        "[LIMITED] Open a URL in a browser — interface only, not fully implemented",
        properties={
            "url": {**_STR, "description": "URL to open"},
        },
        required=["url"],
    ),
]

# ---------------------------------------------------------------------------
# Database  (interface only — still under development)
# ---------------------------------------------------------------------------

DATABASE_FUNCTIONS = [
    _fn(
        "database.query",
        "[LIMITED] Execute a SELECT / read-only SQL query — interface only",
        properties={
            "sql": {**_STR, "description": "SQL query to execute"},
        },
        required=["sql"],
    ),
]

# ---------------------------------------------------------------------------
# MCP  (interface only — still under development)
# ---------------------------------------------------------------------------

MCP_FUNCTIONS = [
    _fn(
        "mcp.execute",
        "[LIMITED] Invoke an MCP server tool — interface only",
        properties={
            "server": {**_STR, "description": "MCP server name"},
            "tool": {**_STR, "description": "Tool name on the server"},
            "arguments": {**_STR, "description": "JSON string of tool arguments"},
        },
        required=["server", "tool"],
    ),
]

# ---------------------------------------------------------------------------
# Composio  (500+ integrations)
# ---------------------------------------------------------------------------

COMPOSIO_FUNCTIONS = [
    _fn(
        "composio.execute",
        "Execute any tool from the Composio platform — 500+ integrations "
        "(Gmail, Slack, GitHub, Notion, Jira, Linear, Google Sheets, etc.). "
        "Pass the tool slug and all required parameters as keyword arguments.",
        properties={
            "slug": {
                **_STR,
                "description": (
                    "Composio tool slug, e.g. "
                    "GMAIL_SEND_EMAIL, SLACK_POST_MESSAGE, "
                    "GITHUB_CREATE_ISSUE, NOTION_CREATE_PAGE, "
                    "LINEAR_CREATE_ISSUE, GOOGLESHEETS_CREATE_SHEET"
                ),
            },
        },
        required=["slug"],
        extra_params={"additionalProperties": True},
    ),
]

# ---------------------------------------------------------------------------
# Aggregated
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: list[dict[str, Any]] = (
    SEARCH_FUNCTIONS
    + PYTHON_FUNCTIONS
    + FILESYSTEM_FUNCTIONS
    + GIT_FUNCTIONS
    + BROWSER_FUNCTIONS
    + DATABASE_FUNCTIONS
    + MCP_FUNCTIONS
    + COMPOSIO_FUNCTIONS
)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return the full list of function definitions for the LLM ``tools`` parameter."""
    return TOOL_FUNCTIONS


def get_function_names() -> list[str]:
    """Return all registered function names (``tool.action`` format)."""
    return [fn["function"]["name"] for fn in TOOL_FUNCTIONS]


def parse_function_name(name: str) -> tuple[str, str]:
    """Parse ``tool.action`` → (tool, action).

    Examples::
        >>> parse_function_name("filesystem.mkdir")
        ("filesystem", "mkdir")
        >>> parse_function_name("search.web.search")
        ("search", "web.search")
    """
    dot = name.find(".")
    if dot == -1:
        return name, ""
    return name[:dot], name[dot + 1:]
