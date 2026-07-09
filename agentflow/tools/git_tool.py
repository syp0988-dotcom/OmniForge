"""GitTool — git repository operations.

Wraps common ``git`` subcommands with structured output. All operations
run in the configured repository path (defaults to current working directory).

Supported actions:

  - ``status``, ``diff``
  - ``add`` (stages files)
  - ``commit`` (creates a commit)
  - ``checkout`` (switch branches / restore files)
  - ``branch`` (list / create branches)
  - ``log`` (commit history)
  - ``show`` (show commit details)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("git_tool")

_MAX_OUTPUT = 50_000  # cap output to avoid giant responses


class GitTool(BaseTool):
    """Git repository operations via subprocess."""

    name = "git"
    description = "Git repository operations (status, diff, add, commit, branch, log)"

    def __init__(self, repo_path: str | None = None) -> None:
        self._repo = Path(repo_path).resolve() if repo_path else Path.cwd().resolve()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        return {
            "status": {
                "description": "查看 Git 仓库状态（修改、暂存、未跟踪文件）",
                "parameters": {},
                "required": [],
            },
            "diff": {
                "description": "查看文件差异（工作区 vs 暂存区）",
                "parameters": {
                    "staged": {"type": "boolean", "description": "是否查看已暂存的差异", "default": False},
                    "path": {"type": "string", "description": "指定文件路径（可选）"},
                },
                "required": [],
            },
            "add": {
                "description": "暂存文件修改",
                "parameters": {
                    "files": {"type": "string", "description": "空格分隔的文件路径或 '.' 表示全部"},
                },
                "required": ["files"],
            },
            "commit": {
                "description": "创建新的提交",
                "parameters": {
                    "message": {"type": "string", "description": "提交信息"},
                },
                "required": ["message"],
            },
            "checkout": {
                "description": "切换分支或恢复文件",
                "parameters": {
                    "branch": {"type": "string", "description": "要切换到的分支名称"},
                    "create": {"type": "boolean", "description": "是否创建新分支", "default": False},
                },
                "required": ["branch"],
            },
            "branch": {
                "description": "列出分支或创建新分支",
                "parameters": {
                    "name": {"type": "string", "description": "新分支名称（省略则列出所有分支）"},
                },
                "required": [],
            },
            "log": {
                "description": "查看提交历史",
                "parameters": {
                    "count": {"type": "integer", "description": "显示最近的提交数", "default": 10},
                },
                "required": [],
            },
            "show": {
                "description": "显示指定提交的详细信息",
                "parameters": {
                    "revision": {"type": "string", "description": "提交哈希值"},
                },
                "required": ["revision"],
            },
        }

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base["repo_path"] = str(self._repo)
        return base

    # ------------------------------------------------------------------
    # Execute — dispatches by ``action`` kwarg
    # ------------------------------------------------------------------

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        handler = _ACTION_MAP.get(action)
        if handler is None:
            return ToolResult.fail(
                self.name, action or "execute",
                f"Unknown git action '{action}'. "
                f"Available: {', '.join(sorted(_ACTION_MAP))}",
            )
        return handler(self, **kwargs)

    # ==================================================================
    # Actions
    # ==================================================================

    def cmd_status(self, **kwargs: Any) -> ToolResult:
        out, err, rc = self._git("status", "--short", "-b")
        if rc != 0:
            return ToolResult.fail(self.name, "status", err.strip())
        return ToolResult.ok(self.name, "status", {"output": out}, "Git status")

    def cmd_diff(self, staged: bool = False, path: str = "", **kwargs: Any) -> ToolResult:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            args.append(path)
        out, err, rc = self._git(*args)
        if rc != 0:
            return ToolResult.fail(self.name, "diff", err.strip())
        return ToolResult.ok(
            self.name, "diff",
            {"output": out, "staged": staged},
            f"Diff ({'staged' if staged else 'unstaged'})",
        )

    def cmd_add(self, files: str | list[str] = "", **kwargs: Any) -> ToolResult:
        """Stage files. Accepts a single path or a list."""
        if isinstance(files, str):
            files_list = [files] if files.strip() else []
        else:
            files_list = list(files)
        if not files_list:
            return ToolResult.fail(self.name, "add", "No files specified to add")
        out, err, rc = self._git("add", *files_list)
        if rc != 0:
            return ToolResult.fail(self.name, "add", err.strip())
        return ToolResult.ok(
            self.name, "add",
            {"files": files_list},
            f"Staged {len(files_list)} file(s)",
        )

    def cmd_commit(
        self,
        message: str = "",
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not message.strip():
            return ToolResult.fail(self.name, "commit", "Commit message is required")
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        out, err, rc = self._git(*args)
        if rc != 0:
            return ToolResult.fail(self.name, "commit", err.strip())
        return ToolResult.ok(
            self.name, "commit",
            {"message": message, "output": out},
            "Commit created",
        )

    def cmd_checkout(
        self,
        branch: str = "",
        create: bool = False,
        path: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        args: list[str] = []
        if create and branch:
            args.extend(["-b", branch])
        elif branch:
            args.append(branch)
        elif path:
            args.extend(["--", path])
        else:
            return ToolResult.fail(
                self.name, "checkout",
                "Specify a branch name or use create=true for a new branch",
            )
        out, err, rc = self._git("checkout", *args)
        if rc != 0:
            return ToolResult.fail(self.name, "checkout", err.strip())
        return ToolResult.ok(
            self.name, "checkout",
            {"branch": branch or path, "created": create},
            f"Checked out: {branch or path}",
        )

    def cmd_branch(
        self,
        name: str = "",
        list_all: bool = False,
        delete: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        args: list[str] = []
        if delete:
            args.extend(["-d", delete])
        elif name:
            args.extend([name])
        else:
            args.append("-a" if list_all else "")
        # Filter empty strings
        args = [a for a in args if a]
        out, err, rc = self._git("branch", *args)
        if rc != 0:
            return ToolResult.fail(self.name, "branch", err.strip())
        return ToolResult.ok(
            self.name, "branch",
            {"output": out, "action": "list" if not name and not delete else "create" if name else "delete"},
            "Branches",
        )

    def cmd_log(self, count: int = 10, **kwargs: Any) -> ToolResult:
        out, err, rc = self._git(
            "log", f"--max-count={count}",
            "--pretty=format:%h %s%d [%an] %ar",
        )
        if rc != 0:
            return ToolResult.fail(self.name, "log", err.strip())
        return ToolResult.ok(
            self.name, "log",
            {"output": out, "count": count},
            f"Last {count} commit(s)",
        )

    def cmd_show(self, revision: str = "HEAD", **kwargs: Any) -> ToolResult:
        if not revision.strip():
            revision = "HEAD"
        out, err, rc = self._git("show", revision)
        if rc != 0:
            return ToolResult.fail(self.name, "show", err.strip())
        return ToolResult.ok(
            self.name, "show",
            {"revision": revision, "output": out},
            f"Commit {revision}",
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _git(self, *args: str) -> tuple[str, str, int]:
        """Run a git command in the configured repo directory."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True,
                text=True,
                cwd=str(self._repo),
                timeout=30,
            )
            out = (result.stdout or "")[:_MAX_OUTPUT]
            err = (result.stderr or "")[:_MAX_OUTPUT]
            return out, err, result.returncode
        except FileNotFoundError:
            return "", "git: command not found", -1
        except subprocess.TimeoutExpired:
            return "", "git: command timed out after 30s", -1
        except Exception as exc:
            return "", f"git: {exc}", -1


# -- Action dispatch map --------------------------------------------------------

_ACTION_MAP: dict[str, Any] = {
    "status": GitTool.cmd_status,
    "diff": GitTool.cmd_diff,
    "add": GitTool.cmd_add,
    "commit": GitTool.cmd_commit,
    "checkout": GitTool.cmd_checkout,
    "branch": GitTool.cmd_branch,
    "log": GitTool.cmd_log,
    "show": GitTool.cmd_show,
}
