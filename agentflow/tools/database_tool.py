"""DatabaseTool — structured data query interface (interface placeholder).

Provides a uniform interface for database operations:

  - ``query`` — execute a SELECT / read-only query
  - ``insert`` — insert rows into a table
  - ``update`` — update rows
  - ``delete`` — delete rows

The current implementation is an **interface placeholder**.  Subclass or
configure with a concrete database driver (SQLAlchemy, psycopg, sqlite3, …)
to enable real data access.
"""

from __future__ import annotations

from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult


class DatabaseTool(BaseTool):
    """Structured data query interface (interface — requires concrete driver)."""

    name = "database"
    description = "Structured data query operations — SELECT, INSERT, UPDATE, DELETE"

    def __init__(self, connection_string: str | None = None) -> None:
        self.connection_string = connection_string

    def actions(self) -> dict[str, dict]:
        return {
            "query": {
                "description": "[接口预留] 执行 SELECT / 只读 SQL 查询",
                "parameters": {"sql": {"type": "string", "description": "SQL 查询语句"}},
                "required": ["sql"],
            },
            "insert": {
                "description": "[接口预留] 向表中插入数据行",
                "parameters": {
                    "table": {"type": "string", "description": "目标表名"},
                    "data": {"type": "object", "description": "要插入的数据记录"},
                },
                "required": ["table"],
            },
            "update": {
                "description": "[接口预留] 更新表中的数据行",
                "parameters": {
                    "table": {"type": "string", "description": "目标表名"},
                    "data": {"type": "object", "description": "要更新的数据"},
                    "where": {"type": "string", "description": "WHERE 条件子句"},
                },
                "required": ["table"],
            },
            "delete": {
                "description": "[接口预留] 删除表中的数据行",
                "parameters": {
                    "table": {"type": "string", "description": "目标表名"},
                    "where": {"type": "string", "description": "WHERE 条件子句"},
                },
                "required": ["table"],
            },
        }

    def capabilities(self) -> list[str]:
        return [
            "database.query",
            "database.insert",
            "database.update",
            "database.delete",
        ]

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["status"] = "interface_only"
        base["message"] = (
            "This is an interface placeholder. "
            "Configure with a database driver (SQLAlchemy, psycopg, sqlite3, etc.) "
            "to enable real data access."
        )
        if self.connection_string:
            safe = self.connection_string
            if "://" in safe:
                safe = safe.split("://")[0] + "://***"
            base["connection"] = safe
        return base

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        handler = _ACTION_MAP.get(action)
        if handler is None:
            return ToolResult.fail(
                self.name, action or "execute",
                f"Unknown database action '{action}'. "
                f"Available: {', '.join(sorted(_ACTION_MAP))}",
            )
        return handler(self, **kwargs)

    # ==================================================================
    # Interface stubs — implement with real database driver
    # ==================================================================

    def cmd_query(self, sql: str = "", **kwargs: Any) -> ToolResult:
        if not sql:
            return ToolResult.fail(self.name, "query", "SQL query is required")
        return ToolResult.fail(
            self.name, "query",
            "DatabaseTool not yet implemented — configure a database driver",
        )

    def cmd_insert(self, table: str = "", data: dict | None = None, **kwargs: Any) -> ToolResult:
        if not table:
            return ToolResult.fail(self.name, "insert", "Table name is required")
        return ToolResult.fail(
            self.name, "insert",
            "DatabaseTool not yet implemented",
        )

    def cmd_update(self, table: str = "", data: dict | None = None,
                   where: str = "", **kwargs: Any) -> ToolResult:
        if not table:
            return ToolResult.fail(self.name, "update", "Table name is required")
        return ToolResult.fail(
            self.name, "update",
            "DatabaseTool not yet implemented",
        )

    def cmd_delete(self, table: str = "", where: str = "", **kwargs: Any) -> ToolResult:
        if not table:
            return ToolResult.fail(self.name, "delete", "Table name is required")
        return ToolResult.fail(
            self.name, "delete",
            "DatabaseTool not yet implemented",
        )


# -- Action dispatch map --------------------------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "query": DatabaseTool.cmd_query,
    "insert": DatabaseTool.cmd_insert,
    "update": DatabaseTool.cmd_update,
    "delete": DatabaseTool.cmd_delete,
}
