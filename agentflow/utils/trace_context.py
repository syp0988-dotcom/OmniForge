"""Request-scoped trace ID via Python contextvars.

Usage in API handlers::

    from agentflow.utils.trace_context import set_trace_id, get_trace_id

    trace_id = set_trace_id()          # generates and sets a new ID
    trace_id = set_trace_id("abc123")  # sets an explicit ID
    current = get_trace_id()           # reads the current ID (empty string if unset)

The trace_id is automatically injected into every log record via
``TraceIdFilter`` (registered in ``build_logger``).  ``contextvars``
propagate correctly through ``asyncio`` tasks and LangGraph's internal
thread pool.
"""

from __future__ import annotations

import contextvars
import uuid

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)


def generate_trace_id() -> str:
    """Return a short hex trace ID (16 hex chars, 64 bits)."""
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str:
    """Return the current trace_id, or empty string if none is set."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    """Set the trace_id for the current request scope.

    Args:
        trace_id: Explicit ID, or None to generate one.

    Returns:
        The trace_id that was set.
    """
    tid = trace_id or generate_trace_id()
    _trace_id_var.set(tid)
    return tid
