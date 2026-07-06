"""Decorators for the AgentFlow framework."""

from __future__ import annotations

import functools
from typing import Any, Callable

from agentflow.utils.logging import build_logger

logger = build_logger("decorators")


def safe_run(
    func: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap an agent's ``run(state)`` so that unhandled exceptions are captured
    and returned as a safe fallback state instead of crashing the workflow.

    Usage::

        @safe_run
        def run(self, state: dict) -> dict:
            ...
    """
    agent_name = getattr(func, "__name__", "unknown_agent")

    @functools.wraps(func)
    def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return func(state, *args, **kwargs)
        except Exception as exc:
            logger.exception("Agent '%s' crashed: %s", agent_name, exc)
            state["error"] = f"[{agent_name}] {exc}"
            return state

    return wrapper
