"""Python Agent — executes Python code in a subprocess with resource limits."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import time
from typing import Any

from agentflow.utils.logging import build_logger

logger = build_logger("python")

_RE_PYTHON_BLOCK = r"```python\n?(.*?)```"
_RE_PY_BLOCK = r"```py\n?(.*?)```"


class PythonAgent:
    """Execute Python code in a subprocess with timeout and output limits."""

    def __init__(
        self, timeout: int = 30, max_output_chars: int = 10_000
    ) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        code = self._extract_code(question)
        result = self._execute_code(code)
        state["python_result"] = result
        return state

    def _extract_code(self, text: str) -> str:
        """Extract Python code from text — try triple-backtick blocks first,
        then fall back to parsing the whole input as Python."""
        import re
        for pattern in (_RE_PYTHON_BLOCK, _RE_PY_BLOCK):
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return "\n".join(m.strip() for m in matches)
        try:
            ast.parse(text)
            return text
        except SyntaxError:
            return ""

    def _execute_code(self, code: str) -> dict[str, Any]:
        if not code.strip():
            return {"status": "no_code", "stdout": "", "stderr": "",
                    "return_code": 0, "duration": 0.0}

        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"status": "syntax_error", "stdout": "", "stderr": f"SyntaxError: {e}",
                    "return_code": -1, "duration": 0.0}

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
                "status": "timeout", "stdout": "", "stderr": "",
                "return_code": -1, "duration": float(self.timeout),
            }
        except FileNotFoundError:
            return {
                "status": "error", "stdout": "", "stderr": "Python interpreter not found",
                "return_code": -1, "duration": 0.0,
            }
