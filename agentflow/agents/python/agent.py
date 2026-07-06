"""Python Agent — decides WHEN to execute Python code and extracts it.

This agent does NOT execute code directly — it delegates execution to
PythonTool (via the Executor in future, directly for now).
"""

from __future__ import annotations

import re
from typing import Any

from agentflow.agents.base import AgentProtocol
from agentflow.tools.python_tool import PythonTool
from agentflow.utils.decorators import safe_run
from agentflow.utils.logging import build_logger

logger = build_logger("python")

_RE_PYTHON_BLOCK = r"```python\n?(.*?)```"
_RE_PY_BLOCK = r"```py\n?(.*?)```"


class PythonAgent(AgentProtocol):
    """Decide whether Python execution is needed and prepare input."""

    def __init__(self) -> None:
        self.tool = PythonTool()

    @safe_run
    def run(self, state: dict[str, object]) -> dict[str, object]:
        question = str(state.get("question", ""))
        code = self._extract_code(question)

        if code:
            logger.info("Executing Python code (%d chars)", len(code))
            result = self.tool.execute(code=code)
        else:
            logger.info("No Python code block found")
            result = {
                "status": "no_code",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "duration": 0.0,
            }

        state["python_result"] = result
        return state

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from text — try triple-backtick blocks first,
        then fall back to parsing the whole input as Python."""
        for pattern in (_RE_PYTHON_BLOCK, _RE_PY_BLOCK):
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return "\n".join(m.strip() for m in matches)
        try:
            import ast
            ast.parse(text)
            return text
        except SyntaxError:
            return ""
