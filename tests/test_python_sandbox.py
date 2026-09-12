"""Tests for PythonTool sandbox — AST validation + runtime isolation."""

import pytest

from agentflow.tools.python_tool import PythonTool

tool = PythonTool()


def test_simple_math_executes():
    """Simple arithmetic should work."""
    result = tool.execute(code="print(1 + 1)")
    assert result.success, f"Failed: stderr={result.result.get('stderr', '')}"
    assert result.result["stdout"].strip() == "2"


def test_blocked_import_os():
    """Direct 'import os' is blocked."""
    result = tool.execute(code="import os; print(os.getcwd())")
    assert not result.success
    assert "blocked" in result.error


def test_blocked_import_subprocess():
    """'import subprocess' is blocked."""
    result = tool.execute(code="import subprocess; subprocess.run(['ls'])")
    assert not result.success
    assert "blocked" in result.error


def test_blocked_call_eval():
    """Calling eval() is blocked."""
    result = tool.execute(code="eval('1+1')")
    assert not result.success
    assert "blocked" in result.error


def test_blocked_call_exec():
    """Calling exec() is blocked."""
    result = tool.execute(code="exec('x=1')")
    assert not result.success
    assert "blocked" in result.error


def test_blocked_call_open():
    """Calling open() is blocked."""
    result = tool.execute(code="open('/tmp/test', 'w')")
    assert not result.success
    assert "blocked" in result.error


@pytest.mark.parametrize(
    "code",
    [
        "import io; print(io.open('/etc/hostname').read())",
        "import _io; print(_io.open('/etc/hostname').read())",
        "import codecs; print(codecs.open('/etc/hostname').read())",
        "import fileinput; print(next(fileinput.input('/etc/hostname')))",
        "import pickle; pickle.loads(b'')",
        "import marshal; marshal.loads(b'')",
    ],
)
def test_blocked_file_object_escape_hatches(code):
    """io/_io/codecs/... must not bypass the blocked ``open`` builtin.

    Regression for security review H2: ``io.open`` and ``_io.open`` used to
    read arbitrary files because only the ``open`` builtin was replaced.
    """
    result = tool.execute(code=code)
    assert not result.success
    assert "blocked" in result.error


def test_open_attribute_access_blocked():
    """``<anything>.open(...)`` is rejected by the attribute-level guard."""
    result = tool.execute(
        code="import sys; sys.modules['builtins'].open('/etc/hostname')"
    )
    assert not result.success
    assert "blocked" in result.error


def test_blocked_builtins_dict_bypass():
    """__builtins__ dict access pattern is blocked."""
    result = tool.execute(code="__builtins__.__dict__['eval']('1+1')")
    assert not result.success, f"Expected failure, got: {result.error}"
    assert "blocked" in result.error


def test_blocked_getattr_builtins():
    """getattr(__builtins__, ...) pattern is blocked."""
    result = tool.execute(code="getattr(__builtins__, 'op' + 'en')('/etc')")
    assert not result.success, f"Expected failure, got: {result.error}"
    assert "blocked" in result.error or "blocked" in str(result.result)


def test_network_blocked():
    """Network access via socket is blocked at runtime."""
    result = tool.execute(
        code="import socket; s=socket.socket(); s.connect(('localhost', 8080))"
    )
    # Should either fail validation (socket blocked) or runtime (connect blocked)
    assert not result.success


def test_allowed_pure_computation():
    """Pure computation without blocked features should work."""
    result = tool.execute(code="""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
print(fib(10))
""")
    assert result.success
    assert result.result["stdout"].strip() == "55"


def test_syntax_error_returns_failure():
    """Invalid Python syntax is rejected early."""
    result = tool.execute(code="this is not valid python {{{")
    assert not result.success
    assert "Syntax" in result.error


def test_empty_code_returns_ok():
    """Empty code should return success but no-op."""
    result = tool.execute(code="")
    assert result.success
    assert result.result["status"] == "no_code"
