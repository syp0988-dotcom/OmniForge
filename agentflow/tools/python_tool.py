"""PythonTool — sandboxed Python code execution in a subprocess.

All execution runs in a temporary directory with a minimal environment.
Syntax is validated via ``ast.parse()`` before execution.

Returns unified ``ToolResult`` with execution details in ``result``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from agentflow.config.settings import settings
from agentflow.tools.base import BaseTool
from agentflow.tools.result import ToolResult
from agentflow.utils.logging import build_logger

logger = build_logger("python_tool")

# Env vars to preserve for subprocess stability
_SAFE_ENV_KEYS = {
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMP", "TEMP",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
}

_BLOCKED_IMPORTS = {
    "os", "subprocess", "socket", "shutil", "pathlib", "ctypes",
    "multiprocessing", "threading", "requests", "urllib", "http",
}
_BLOCKED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
}
_BLOCKED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmdir", "rmtree",
    "rename", "replace", "chmod", "chown", "kill",
}


def _build_sandbox_env() -> dict[str, str]:
    """Minimal environment — only safe system variables."""
    return {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}


class PythonTool(BaseTool):
    """Execute Python code in a subprocess with timeout and output limits."""

    name = "python"
    description = "Sandboxed Python code execution with timeout"

    def __init__(self, timeout: int = 30, max_output_chars: int = 10_000) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actions(self) -> dict[str, dict]:
        return {
            "execute": {
                "description": "在沙箱子进程中执行 Python 代码（30 秒超时）",
                "parameters": {
                    "code": {"type": "string", "description": "要执行的 Python 源代码"},
                },
                "required": ["code"],
            },
        }

    def routing_node(self) -> str:
        return "python"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, code: str = "", **kwargs: Any) -> tuple[bool, str]:
        """Check that the code is syntactically valid Python."""
        code = code or kwargs.get("code", "")
        if not code.strip():
            return False, "No code provided"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        if settings.allow_unsafe_python_tool:
            return True, ""

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in _BLOCKED_IMPORTS:
                        return False, f"Import '{root}' is blocked by the safe Python policy"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in _BLOCKED_IMPORTS:
                    return False, f"Import '{root}' is blocked by the safe Python policy"
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                    return False, f"Call '{func.id}' is blocked by the safe Python policy"
                if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTRS:
                    return False, f"Call '*.{func.attr}' is blocked by the safe Python policy"

        return True, ""

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        """Execute Python code and return structured results.

        Args:
            code: Python source to execute.

        Returns:
            ``ToolResult`` with ``result`` containing:
            ``{"status", "stdout", "stderr", "return_code", "duration"}``
        """
        code = code or kwargs.get("code", "")
        if not code.strip():
            return ToolResult.ok(
                self.name, "execute",
                result={"status": "no_code", "stdout": "", "stderr": "",
                        "return_code": 0, "duration": 0.0},
                message="No code provided",
            )

        # Syntax validation
        valid, err = self.validate(code=code)
        if not valid:
            return ToolResult.fail(self.name, "execute", err)

        start = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=_build_sandbox_env(),
                    cwd=tmpdir,
                )

            duration = round(time.time() - start, 3)
            stdout = (proc.stdout or "")[-self.max_output_chars :]
            stderr = (proc.stderr or "")[-self.max_output_chars :]
            is_ok = proc.returncode == 0

            return ToolResult(
                success=is_ok,
                tool=self.name,
                action="execute",
                result={
                    "status": "ok" if is_ok else "error",
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": proc.returncode,
                    "duration": duration,
                },
                message=f"Python {'succeeded' if is_ok else 'failed'} (exit={proc.returncode})",
                duration=duration,
            )

        except subprocess.TimeoutExpired:
            return ToolResult.fail(
                self.name, "execute",
                f"Execution timed out after {self.timeout}s",
                result={"status": "timeout", "stdout": "", "stderr": "",
                        "return_code": -1, "duration": float(self.timeout)},
            )
        except FileNotFoundError:
            return ToolResult.fail(
                self.name, "execute",
                "Python interpreter not found",
                result={"status": "error", "stdout": "", "stderr": "",
                        "return_code": -1, "duration": 0.0},
            )
