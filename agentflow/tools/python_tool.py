"""PythonTool — sandboxed Python code execution in a subprocess."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import time
from typing import Any

from agentflow.tools.base import BaseTool
from agentflow.utils.logging import build_logger

logger = build_logger("python_tool")


class PythonTool(BaseTool):
    """Execute Python code in a subprocess with timeout and output limits.

    Executor usage::

        executor.execute(ctx, Task(
            goal="执行 Python 脚本",
            tool="python",
            input={"code": "print('hello')"},
        ))
    """

    name = "python"

    def __init__(self, timeout: int = 30, max_output_chars: int = 10_000) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str = "", **kwargs: Any) -> dict[str, Any]:
        """Execute Python code and return structured results.

        Args:
            code: Python source code to execute.

        Returns:
            dict with keys: status, stdout, stderr, return_code, duration.
        """
        if not code.strip():
            return {
                "status": "no_code",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "duration": 0.0,
            }

        # Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "syntax_error",
                "stdout": "",
                "stderr": f"SyntaxError: {e}",
                "return_code": -1,
                "duration": 0.0,
            }

        start = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={},
                    cwd=tmpdir,
                )

            duration = round(time.time() - start, 3)
            stdout = (proc.stdout or "")[-self.max_output_chars :]
            stderr = (proc.stderr or "")[-self.max_output_chars :]

            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "stdout": stdout,
                "stderr": stderr,
                "return_code": proc.returncode,
                "duration": duration,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "duration": float(self.timeout),
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "stdout": "",
                "stderr": "Python interpreter not found",
                "return_code": -1,
                "duration": 0.0,
            }
