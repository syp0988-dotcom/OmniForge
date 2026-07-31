"""Python Agent — decides WHEN to execute Python code and extracts it.

This agent does NOT execute code directly — it delegates execution to
PythonTool (via the Executor in future, directly for now).
"""

from __future__ import annotations

import re

from agentflow.agents.base import AgentProtocol
from agentflow.config.settings import settings
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
        # Safety check: honour the ALLOW_UNSAFE_PYTHON_TOOL setting
        if not settings.allow_unsafe_python_tool:
            logger.warning("Python execution blocked by ALLOW_UNSAFE_PYTHON_TOOL=False")
            state["python_result"] = {
                "status": "blocked",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "duration": 0.0,
            }
            state["tool_results"] = [{
                "success": True,
                "tool": "python",
                "action": "execute",
                "result": {"status": "blocked"},
                "error": None,
                "message": "Python execution blocked by safety policy (ALLOW_UNSAFE_PYTHON_TOOL=False)",
            }]
            return state

        # Find the python task in the task queue and mark it running
        task_queue: list[dict] = list(state.get("task_queue", []) or [])
        task = self._find_python_task(task_queue)
        if task:
            task["status"] = "running"

        question = str(state.get("question", ""))
        code = self._extract_code(question)
        success = False
        error_msg = ""

        if code:
            logger.info("Executing Python code (%d chars)", len(code))
            try:
                result_raw = self.tool.execute(code=code)
                success = getattr(result_raw, "success", False)
                if hasattr(result_raw, "result") and isinstance(result_raw.result, dict):
                    result = result_raw.result
                else:
                    result = {
                        "status": "ok" if success else "error",
                        "stdout": str(getattr(result_raw, "result", "")),
                        "stderr": getattr(result_raw, "error", "") or "",
                        "return_code": 0 if success else -1,
                        "duration": getattr(result_raw, "duration", 0.0),
                    }
                if not success:
                    error_msg = getattr(result_raw, "error", "") or "Unknown error"
            except Exception as exc:
                logger.error("Python execution crashed: %s", exc)
                result = {
                    "status": "error",
                    "stdout": "",
                    "stderr": str(exc),
                    "return_code": -1,
                    "duration": 0.0,
                }
                error_msg = str(exc)
        else:
            logger.info("No Python code block found")
            result = {
                "status": "no_code",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "duration": 0.0,
            }

        # Update task status in queue and produce tool_result
        if task:
            task["status"] = "done" if success else "failed"
            if error_msg:
                task["error"] = error_msg

        state["python_result"] = result
        state["task_queue"] = task_queue
        state["tool_results"] = [{
            "success": success,
            "tool": "python",
            "action": task.get("goal", "execute") if task else "execute",
            "result": result,
            "error": error_msg or None,
        }] if task else []
        return state

    @staticmethod
    def _find_python_task(queue: list[dict]) -> dict | None:
        """Find the first TODO task with tool='python' in the queue."""
        for t in queue:
            if t.get("tool") == "python" and t.get("status") in ("todo", "running"):
                return t
        return None

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
