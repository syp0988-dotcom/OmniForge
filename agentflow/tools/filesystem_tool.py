"""FileSystemTool — safe file and directory operations.

All file operations are restricted to the configured workspace directory.
Path traversal, absolute paths outside the workspace, and dangerous system
directories are rejected at the validation layer.

Supported actions (passed as ``action`` in the task dict, or as the first
keyword argument to ``execute()``):

  - ``mkdir``, ``create_file``, ``write_file``, ``append_file``
  - ``read_file``, ``edit_file``, ``replace_text``
  - ``delete_file``, ``delete_directory``
  - ``move_file``, ``copy_file``, ``rename_file``
  - ``exists``, ``list_directory``, ``tree``
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("filesystem_tool")

# ---------------------------------------------------------------------------
# Safety — directories that are NEVER allowed for any write operation
# ---------------------------------------------------------------------------
_DANGEROUS_DIRS: set[str] = {
    "/etc", "/var", "/sys", "/proc", "/dev", "/boot", "/bin", "/sbin",
    "/lib", "/lib64", "/usr", "/opt", "/root",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\System32", "C:\\Windows\\System32",
}

# Agent source directories — read is OK, write is blocked.
_AGENT_SRC_NAMES: set[str] = {"agentflow", "omniforge"}

# Patterns that look like path traversal attempts.
_TRAVERSAL_PATTERNS = re.compile(r"(\.\./|\.\.\\)")


class FileSystemTool(BaseTool):
    """Safe file system operations restricted to a workspace directory."""

    name = "filesystem"
    description = "Safe file and directory operations within the workspace"

    def __init__(self, workspace: str | None = None) -> None:
        # Use configured workspace or default to cwd
        if workspace:
            self._workspace = Path(workspace).resolve()
        else:
            # Try to get from settings, fallback to cwd
            self._workspace = Path.cwd().resolve()
        self._workspace_str = str(self._workspace)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        return {
            "mkdir": {
                "description": "创建目录（自动创建父目录，目录已存在不报错）",
                "parameters": {"path": {"type": "string", "description": "目录路径（相对于工作区）"}},
                "required": ["path"],
            },
            "create_file": {
                "description": "创建新文件（文件已存在则失败）",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径（相对于工作区）"},
                    "content": {"type": "string", "description": "初始文件内容"},
                },
                "required": ["path"],
            },
            "write_file": {
                "description": "写入文件内容（覆盖已有文件）",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径（相对于工作区）"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
            "append_file": {
                "description": "向已有文件追加内容",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径（相对于工作区）"},
                    "content": {"type": "string", "description": "要追加的内容"},
                },
                "required": ["path", "content"],
            },
            "read_file": {
                "description": "读取文件的完整内容",
                "parameters": {"path": {"type": "string", "description": "文件路径（相对于工作区）"}},
                "required": ["path"],
            },
            "edit_file": {
                "description": "替换文件中首次出现的指定字符串",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径（相对于工作区）"},
                    "old_string": {"type": "string", "description": "要查找并替换的文本"},
                    "new_string": {"type": "string", "description": "替换后的文本"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            "replace_text": {
                "description": "使用正则表达式替换文件中所有匹配的文本",
                "parameters": {
                    "path": {"type": "string", "description": "文件路径（相对于工作区）"},
                    "pattern": {"type": "string", "description": "正则表达式模式"},
                    "replacement": {"type": "string", "description": "替换文本"},
                },
                "required": ["path", "pattern", "replacement"],
            },
            "delete_file": {
                "description": "删除单个文件（不能删除目录）",
                "parameters": {"path": {"type": "string", "description": "文件路径（相对于工作区）"}},
                "required": ["path"],
            },
            "delete_directory": {
                "description": "删除目录及其所有内容（谨慎使用）",
                "parameters": {"path": {"type": "string", "description": "目录路径（相对于工作区）"}},
                "required": ["path"],
            },
            "move_file": {
                "description": "将文件从源路径移动到目标路径",
                "parameters": {
                    "src": {"type": "string", "description": "源文件路径"},
                    "dst": {"type": "string", "description": "目标文件路径"},
                },
                "required": ["src", "dst"],
            },
            "copy_file": {
                "description": "复制文件到目标位置",
                "parameters": {
                    "src": {"type": "string", "description": "源文件路径"},
                    "dst": {"type": "string", "description": "目标文件路径"},
                },
                "required": ["src", "dst"],
            },
            "rename_file": {
                "description": "重命名文件或目录（在同一父目录下）",
                "parameters": {
                    "path": {"type": "string", "description": "文件/目录的当前路径"},
                    "name": {"type": "string", "description": "新名称（非完整路径）"},
                },
                "required": ["path", "name"],
            },
            "exists": {
                "description": "检查文件或目录是否存在于工作区",
                "parameters": {"path": {"type": "string", "description": "要检查的路径"}},
                "required": ["path"],
            },
            "list_directory": {
                "description": "列出目录内容（非递归）",
                "parameters": {
                    "path": {"type": "string", "description": "目录路径（默认：工作区根目录）"},
                },
                "required": [],
            },
            "tree": {
                "description": "生成递归目录树（文本格式）",
                "parameters": {
                    "path": {"type": "string", "description": "目录路径（默认：工作区根目录）"},
                    "max_depth": {"type": "integer", "description": "最大递归深度", "default": 3},
                    "show_hidden": {"type": "boolean", "description": "是否显示隐藏文件", "default": False},
                },
                "required": [],
            },
        }

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["workspace"] = self._workspace_str
        return base

    def capabilities(self) -> list[str]:
        return [
            "filesystem.create",
            "filesystem.read",
            "filesystem.write",
            *super().capabilities(),
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, **kwargs: Any) -> tuple[bool, str]:
        """Reject dangerous paths before execution."""
        # Extract the path argument (many different keys possible)
        path = kwargs.get("path") or kwargs.get("src") or kwargs.get("source") or ""
        if not path:
            return True, ""

        # --- Block path traversal ----------------------------------------------
        if isinstance(path, str) and _TRAVERSAL_PATTERNS.search(path):
            return False, f"Path traversal detected: '{path}' is not allowed"

        # --- Block dangerous system directories --------------------------------
        resolved = self._resolve(path)
        if resolved is None:
            return False, f"Path '{path}' resolves outside the workspace"

        resolved_str = str(resolved)
        for dangerous in _DANGEROUS_DIRS:
            if resolved_str.startswith(dangerous):
                return False, f"Access to system directory '{dangerous}' is forbidden"

        return True, ""

    # ------------------------------------------------------------------
    # Execute — dispatches by ``action`` kwarg or first positional key
    # ------------------------------------------------------------------

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate handler based on *action*."""
        handler = _ACTION_MAP.get(action)
        if handler is None:
            return ToolResult.fail(
                self.name, action,
                f"Unknown action '{action}'. "
                f"Available: {', '.join(sorted(_ACTION_MAP))}",
            )
        return handler(self, **kwargs)

    # ==================================================================
    # Actions
    # ==================================================================

    def cmd_mkdir(self, path: str = "", **kwargs: Any) -> ToolResult:
        """Create a directory (``parents=True``, ``exist_ok=True``)."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("mkdir", path)
        try:
            target.mkdir(parents=True, exist_ok=True)
            return ToolResult.ok(
                self.name, "mkdir", {"path": str(target)},
                f"Directory created: {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "mkdir", str(exc))

    def cmd_create_file(self, path: str = "", content: str = "", **kwargs: Any) -> ToolResult:
        """Create a new file (fails if it already exists)."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("create_file", path)
        if target.exists():
            return ToolResult.fail(
                self.name, "create_file", f"File already exists: {target.name}",
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult.ok(
                self.name, "create_file", {"path": str(target)},
                f"File created: {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "create_file", str(exc))

    def cmd_write_file(self, path: str = "", content: str = "", **kwargs: Any) -> ToolResult:
        """Write content to a file (overwrites if exists)."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("write_file", path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult.ok(
                self.name, "write_file", {"path": str(target)},
                f"File written: {target.name} ({len(content)} chars)",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "write_file", str(exc))

    def cmd_append_file(self, path: str = "", content: str = "", **kwargs: Any) -> ToolResult:
        """Append content to an existing file."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("append_file", path)
        if not target.exists():
            return ToolResult.fail(
                self.name, "append_file", f"File not found: {target.name}",
            )
        try:
            with target.open("a", encoding="utf-8") as f:
                f.write(content)
            return ToolResult.ok(
                self.name, "append_file", {"path": str(target)},
                f"Appended {len(content)} chars to {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "append_file", str(exc))

    def cmd_read_file(self, path: str = "", **kwargs: Any) -> ToolResult:
        """Read the full contents of a file."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("read_file", path)
        if not target.exists():
            return ToolResult.fail(
                self.name, "read_file", f"File not found: {target.name}",
            )
        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult.ok(
                self.name, "read_file",
                {"path": str(target), "content": content, "size": len(content)},
                f"File read: {target.name} ({len(content)} chars)",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "read_file", str(exc))

    def cmd_edit_file(
        self,
        path: str = "",
        old_string: str = "",
        new_string: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Replace the first occurrence of *old_string* with *new_string*."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("edit_file", path)
        if not target.exists():
            return ToolResult.fail(self.name, "edit_file", f"File not found: {target.name}")
        if not old_string:
            return ToolResult.fail(self.name, "edit_file", "old_string is required")
        try:
            content = target.read_text(encoding="utf-8")
            if old_string not in content:
                return ToolResult.fail(
                    self.name, "edit_file",
                    f"String not found in {target.name}",
                )
            new_content = content.replace(old_string, new_string, 1)
            target.write_text(new_content, encoding="utf-8")
            return ToolResult.ok(
                self.name, "edit_file", {"path": str(target)},
                f"Replaced 1 occurrence in {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "edit_file", str(exc))

    def cmd_replace_text(
        self,
        path: str = "",
        pattern: str = "",
        replacement: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Replace all occurrences matching a regex *pattern*."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("replace_text", path)
        if not target.exists():
            return ToolResult.fail(self.name, "replace_text", f"File not found: {target.name}")
        if not pattern:
            return ToolResult.fail(self.name, "replace_text", "pattern is required")
        try:
            content = target.read_text(encoding="utf-8")
            new_content, count = re.subn(pattern, replacement, content)
            if count == 0:
                return ToolResult.fail(
                    self.name, "replace_text",
                    f"Pattern not found in {target.name}",
                )
            target.write_text(new_content, encoding="utf-8")
            return ToolResult.ok(
                self.name, "replace_text", {"path": str(target), "count": count},
                f"Replaced {count} occurrence(s) in {target.name}",
            )
        except re.error as exc:
            return ToolResult.fail(self.name, "replace_text", f"Regex error: {exc}")
        except OSError as exc:
            return ToolResult.fail(self.name, "replace_text", str(exc))

    def cmd_delete_file(self, path: str = "", **kwargs: Any) -> ToolResult:
        """Delete a single file."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("delete_file", path)
        if not target.exists():
            return ToolResult.fail(self.name, "delete_file", f"File not found: {target.name}")
        if not target.is_file():
            return ToolResult.fail(self.name, "delete_file", f"Not a file: {target.name}")
        if self._is_agent_src(target):
            return ToolResult.fail(
                self.name, "delete_file",
                "Cannot delete agent source files",
            )
        try:
            target.unlink()
            return ToolResult.ok(
                self.name, "delete_file", {"path": str(target)},
                f"File deleted: {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "delete_file", str(exc))

    def cmd_delete_directory(self, path: str = "", **kwargs: Any) -> ToolResult:
        """Delete an empty directory (use with caution)."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("delete_directory", path)
        if not target.exists():
            return ToolResult.fail(
                self.name, "delete_directory", f"Directory not found: {target.name}",
            )
        if not target.is_dir():
            return ToolResult.fail(self.name, "delete_directory", f"Not a directory: {target.name}")
        if self._is_agent_src(target):
            return ToolResult.fail(
                self.name, "delete_directory",
                "Cannot delete agent source directories",
            )
        try:
            shutil.rmtree(target)
            return ToolResult.ok(
                self.name, "delete_directory", {"path": str(target)},
                f"Directory deleted: {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "delete_directory", str(exc))

    def cmd_move_file(self, src: str = "", dst: str = "", **kwargs: Any) -> ToolResult:
        """Move a file from *src* to *dst*."""
        source = self._resolve(src)
        dest = self._resolve(dst)
        if source is None:
            return self._invalid_path("move_file", src)
        if dest is None:
            return self._invalid_path("move_file", dst)
        if not source.exists():
            return ToolResult.fail(self.name, "move_file", f"Source not found: {source.name}")
        if self._is_agent_src(source):
            return ToolResult.fail(
                self.name, "move_file",
                "Cannot move agent source files",
            )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            return ToolResult.ok(
                self.name, "move_file", {"src": str(source), "dst": str(dest)},
                f"Moved: {source.name} → {dest.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "move_file", str(exc))

    def cmd_copy_file(self, src: str = "", dst: str = "", **kwargs: Any) -> ToolResult:
        """Copy a file from *src* to *dst*."""
        source = self._resolve(src)
        dest = self._resolve(dst)
        if source is None:
            return self._invalid_path("copy_file", src)
        if dest is None:
            return self._invalid_path("copy_file", dst)
        if not source.exists() or not source.is_file():
            return ToolResult.fail(self.name, "copy_file", f"Source not found: {source.name}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(dest))
            return ToolResult.ok(
                self.name, "copy_file", {"src": str(source), "dst": str(dest)},
                f"Copied: {source.name} → {dest.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "copy_file", str(exc))

    def cmd_rename_file(self, path: str = "", name: str = "", **kwargs: Any) -> ToolResult:
        """Rename a file or directory within the same parent directory."""
        target = self._resolve(path)
        if target is None:
            return self._invalid_path("rename_file", path)
        if not target.exists():
            return ToolResult.fail(self.name, "rename_file", f"Not found: {target.name}")
        if self._is_agent_src(target):
            return ToolResult.fail(
                self.name, "rename_file",
                "Cannot rename agent source files",
            )
        new_name = Path(name).name  # strip any path components
        new_path = target.parent / new_name
        if new_path.exists():
            return ToolResult.fail(
                self.name, "rename_file",
                f"Target already exists: {new_name}",
            )
        try:
            target.rename(new_path)
            return ToolResult.ok(
                self.name, "rename_file",
                {"old": str(target), "new": str(new_path)},
                f"Renamed: {target.name} → {new_name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "rename_file", str(exc))

    def cmd_exists(self, path: str = "", **kwargs: Any) -> ToolResult:
        """Check whether a path exists in the workspace."""
        target = self._resolve(path)
        if target is None:
            return ToolResult.ok(
                self.name, "exists", {"exists": False, "path": path},
                f"Path does not exist (outside workspace): {path}",
            )
        exists = target.exists()
        kind = "file" if target.is_file() else "directory" if target.is_dir() else "other"
        return ToolResult.ok(
            self.name, "exists",
            {"exists": exists, "path": str(target), "kind": kind if exists else None},
            f"Path {'exists' if exists else 'not found'}: {target.name}",
        )

    def cmd_list_directory(self, path: str = "", **kwargs: Any) -> ToolResult:
        """List the contents of a directory (non-recursive)."""
        target = self._resolve(path or ".")
        if target is None:
            return self._invalid_path("list_directory", path)
        if not target.exists():
            return ToolResult.fail(
                self.name, "list_directory", f"Directory not found: {target.name}",
            )
        if not target.is_dir():
            return ToolResult.fail(
                self.name, "list_directory", f"Not a directory: {target.name}",
            )
        try:
            entries: list[dict[str, Any]] = []
            for p in sorted(target.iterdir()):
                entries.append({
                    "name": p.name,
                    "path": str(p),
                    "is_dir": p.is_dir(),
                    "size": p.stat().st_size if p.is_file() else 0,
                })
            return ToolResult.ok(
                self.name, "list_directory",
                {"path": str(target), "entries": entries, "count": len(entries)},
                f"Listed {len(entries)} entries in {target.name}",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "list_directory", str(exc))

    def cmd_tree(
        self,
        path: str = "",
        max_depth: int = 3,
        show_hidden: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Generate a recursive directory tree (text format)."""
        target = self._resolve(path or ".")
        if target is None:
            return self._invalid_path("tree", path)
        if not target.exists() or not target.is_dir():
            return ToolResult.fail(self.name, "tree", f"Invalid directory: {target.name}")
        try:
            lines: list[str] = [f"{target.name}/"]
            self._build_tree(target, "", lines, max_depth, show_hidden)
            return ToolResult.ok(
                self.name, "tree",
                {"path": str(target), "tree": "\n".join(lines), "depth": max_depth},
                f"Tree generated for {target.name} ({len(lines)} lines)",
            )
        except OSError as exc:
            return ToolResult.fail(self.name, "tree", str(exc))

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _resolve(self, raw: str) -> Path | None:
        """Resolve a user-supplied path relative to the workspace.

        Returns ``None`` when the resolved path escapes the workspace.
        """
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._workspace / p).resolve()
        # Must be within workspace
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            return None
        return resolved

    def _is_agent_src(self, path: Path) -> bool:
        """Check if path is inside agent source directories.

        Read access is fine; write / delete / rename are blocked.
        """
        for part in path.parts:
            if part in _AGENT_SRC_NAMES:
                return True
        return False

    def _build_tree(
        self,
        path: Path,
        prefix: str,
        lines: list[str],
        max_depth: int,
        show_hidden: bool,
    ) -> None:
        """Recursive tree builder."""
        if max_depth <= 0:
            return
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        # Filter hidden files
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._build_tree(entry, prefix + extension, lines, max_depth - 1, show_hidden)

    @staticmethod
    def _invalid_path(action: str, raw: str) -> ToolResult:
        return ToolResult.fail(
            "filesystem", action,
            f"Invalid or prohibited path: '{raw}'",
        )


# -- Action dispatch map --------------------------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "mkdir": FileSystemTool.cmd_mkdir,
    "create_file": FileSystemTool.cmd_create_file,
    "write_file": FileSystemTool.cmd_write_file,
    "append_file": FileSystemTool.cmd_append_file,
    "read_file": FileSystemTool.cmd_read_file,
    "edit_file": FileSystemTool.cmd_edit_file,
    "replace_text": FileSystemTool.cmd_replace_text,
    "delete_file": FileSystemTool.cmd_delete_file,
    "delete_directory": FileSystemTool.cmd_delete_directory,
    "move_file": FileSystemTool.cmd_move_file,
    "copy_file": FileSystemTool.cmd_copy_file,
    "rename_file": FileSystemTool.cmd_rename_file,
    "exists": FileSystemTool.cmd_exists,
    "list_directory": FileSystemTool.cmd_list_directory,
    "tree": FileSystemTool.cmd_tree,
}
