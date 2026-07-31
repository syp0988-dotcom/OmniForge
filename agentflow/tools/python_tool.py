"""PythonTool — sandboxed Python code execution in a subprocess.

All execution runs in a temporary directory with a minimal environment.
Syntax is validated via ``ast.parse()`` before execution.  Additional
runtime isolation is applied via a pre-exec wrapper script.

.. warning::
   This is a **best-effort** sandbox (AST checks + subprocess + runtime
   guards), NOT an OS-level sandbox.  It is suitable for a local developer
   tool, but must not be exposed to untrusted/multi-tenant input.  For real
   isolation, run the executor inside a container (see ``Dockerfile``) or a
   dedicated OS user.  ``ALLOW_UNSAFE_PYTHON_TOOL=true`` disables the
   validation layer entirely.

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

# ── AST-level blocking (first line of defence) ─────────────────────────

_BLOCKED_IMPORTS = {
    "os", "subprocess", "socket", "shutil", "pathlib", "ctypes",
    "multiprocessing", "threading", "requests", "urllib", "http",
    "importlib", "signal", "fcntl", "nt", "posix", "pwd", "grp",
    "cffi", "winreg", "_winapi",
}
_BLOCKED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
}
_BLOCKED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmdir", "rmtree",
    "rename", "replace", "chmod", "chown", "kill",
}


def _build_sandbox_env() -> dict[str, str]:
    """Minimal environment — only safe system variables.

    Also sets PYTHONSAFEPATH=1 to prevent ``sys.path`` tampering
    (Python 3.11+).
    """
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    env["PYTHONSAFEPATH"] = "1"
    return env


# ── Runtime isolation wrapper (second line of defence) ─────────────────
#
# The wrapper:
#   1. blocks network access (socket + urllib)
#   2. installs a safe __import__ on the real builtins module
#   3. runs user code with a scrubbed globals dict and a sanitised
#      builtins dict (eval/exec/compile/open/... become raisers)
#   4. removes already-imported dangerous modules from sys.modules
#   5. executes the user program from stdin


_BLOCKED_IMPORTS_SET = set(_BLOCKED_IMPORTS)

_PRE_EXEC_WRAPPER = r"""
import builtins as _b
import sys as _sys

# Saved before patching: the wrapper itself must still be able to run user code.
_orig_exec = _b.exec

# ---- block network access ----
import socket as _socket
_orig_connect = _socket.socket.connect
def _blocked_connect(self, *args, **kwargs):
    raise RuntimeError("Network access blocked by sandbox policy")
_socket.socket.connect = _blocked_connect

# Block urllib and friends
try:
    import urllib.request as _ur
    _ur.urlopen = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Network blocked"))
except ImportError:
    pass

# ---- safe __import__: block dangerous modules ----
_orig_import = _b.__import__
_BLOCKED_MODS = %s
def _safe_import(name, *args, **kwargs):
    root = name.split('.', 1)[0]
    if root in _BLOCKED_MODS:
        raise ImportError("Import '" + root + "' is blocked by sandbox policy")
    return _orig_import(name, *args, **kwargs)

# Patch the real builtins module so code that reaches it via
# ``import builtins`` (e.g. ``builtins.__import__('os')``) is still guarded.
_b.__import__ = _safe_import

# ---- sanitised builtins dict for the user globals ----
def _make_blocked(name):
    def _blocked(*args, **kwargs):
        raise RuntimeError("'" + name + "' is blocked by sandbox policy")
    return _blocked

_safe_builtins = dict(_b.__dict__)
for _blocked_name in ("eval", "exec", "compile", "open", "input",
                      "breakpoint", "exit", "quit", "help"):
    _blocked_fn = _make_blocked(_blocked_name)
    _safe_builtins[_blocked_name] = _blocked_fn
    setattr(_b, _blocked_name, _blocked_fn)
_safe_builtins["__import__"] = _safe_import

# ---- scrub already-imported dangerous modules ----
for _name in _BLOCKED_MODS:
    _sys.modules.pop(_name, None)

# ---- execute user code with scrubbed globals ----
_user_globals = {"__builtins__": _safe_builtins}
_orig_exec(_sys.stdin.read(), _user_globals)
"""


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
        """Check that the code is syntactically valid Python and doesn't
        use blocked imports / calls / attribute access patterns."""
        code = code or kwargs.get("code", "")
        if not code.strip():
            return False, "No code provided"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        if settings.allow_unsafe_python_tool:
            return True, ""

        source_text = code  # keep for text-level checks

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

        # Text-level checks for common __builtins__ bypass patterns
        # (AST can't easily detect dynamic attribute lookups)
        _BYPASS_PATTERNS = [
            "__builtins__",
            "builtins.__dict__",
            "getattr(__builtins__",
            "getattr(builtins",
            "().__class__.__bases__",
            "].__class__.__mro__",
            "object.__subclasses__",
            ".__subclasses__",
            ".__globals__",
            ".__mro__",
        ]
        for pattern in _BYPASS_PATTERNS:
            if pattern in source_text:
                return False, f"Code contains blocked pattern '{pattern}'"

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
                # Use pre-exec wrapper for runtime isolation (network blocking +
                # safe __import__).  The user code is piped via stdin so the
                # wrapper can set up guards before executing it.
                blocked_repr = repr(sorted(_BLOCKED_IMPORTS))
                wrapped_code = _PRE_EXEC_WRAPPER % blocked_repr
                proc = subprocess.run(
                    [sys.executable, "-c", wrapped_code],
                    input=code,
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
