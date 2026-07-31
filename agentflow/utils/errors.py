"""Unified error recording for the workflow system.

Consolidates the three previous error channels
(``_llm_error``, ``_generation_failure_reason``,
``session_state.metadata.last_failure_reason``) into a single
``state["_errors"]`` list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def record_error(
    state: dict[str, Any],
    source: str,
    error_type: str,
    message: str,
) -> None:
    """Append a structured error to the unified ``_errors`` channel.

    Args:
        state: The workflow state dict (mutated in place).
        source: Which agent or component produced the error, e.g. ``"planner"``.
        error_type: Machine-readable category, e.g. ``"llm_unavailable"``,
            ``"generation_failed"``, ``"rate_limit"``.
        message: Human-readable error description.
    """
    errors: list[dict] = list(state.get("_errors", []) or [])
    entry = {
        "source": source,
        "type": error_type,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    errors.append(entry)
    state["_errors"] = errors

    # Backward-compatible mirror (deprecated, will remove in next release)
    if error_type.startswith("llm"):
        state["_llm_error"] = message
