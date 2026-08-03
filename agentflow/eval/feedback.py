"""Runtime feedback collection for eval-driven iteration.

Every chat turn is a potential evaluation example.  Failures (empty answers,
generation failures, recorded errors) are always persisted; successful
project/coding completions are persisted too, so the offline eval suites can
grow from real usage instead of hand-written fixtures.

Records are appended to ``<project>/data/feedback/feedback.jsonl`` (one JSON
object per line) and can be exported/summarized with
``scripts/export_feedback.py``.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentflow.config.settings import settings
from agentflow.graph.state_utils import get_goal_type

logger = logging.getLogger("eval.feedback")

_FEEDBACK_DIR_NAME = "feedback"
_FEEDBACK_FILE_NAME = "feedback.jsonl"
_ANSWER_TRUNCATE = 2000
_QUESTION_TRUNCATE = 500
_DEDUP_SCAN_LINES = 200

_write_lock = threading.Lock()

# Test override: when set, records are written here instead of the default path.
_feedback_path_override: Path | None = None


def set_feedback_path(path: Path | None) -> None:
    """Override the feedback file location (mainly for tests)."""
    global _feedback_path_override
    _feedback_path_override = path


def feedback_path() -> Path:
    """Return the active feedback JSONL path (directory is created)."""
    if _feedback_path_override is not None:
        path = _feedback_path_override
    else:
        path = settings.project_root / "data" / _FEEDBACK_DIR_NAME / _FEEDBACK_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _truncate(text: Any, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _record_key(record: dict[str, Any]) -> str:
    """Dedup key: same question + goal type + error signature."""
    errors = record.get("errors") or []
    error_sig = "|".join(
        f"{e.get('type', '')}:{str(e.get('message', ''))[:80]}"
        for e in errors
    )
    return f"{record.get('question')}|{record.get('goal_type')}|{error_sig}"


def _recent_keys(path: Path) -> set[str]:
    """Return dedup keys of the most recent records in the JSONL file."""
    if not path.exists():
        return set()
    keys: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            tail = fh.readlines()[-_DEDUP_SCAN_LINES:]
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_record_key(json.loads(line)))
            except (ValueError, TypeError):
                continue
    except OSError:
        pass
    return keys


def record_feedback(
    state: dict[str, Any] | None,
    *,
    session_id: int | str | None = None,
    question: str = "",
    answer: str = "",
    source: str = "chat",
) -> bool:
    """Persist one feedback record; returns True when written.

    Recording policy:
      - failures (empty answer / generation failure / recorded errors):
        always persisted;
      - successful completions: only for completion-worthy goal types
        (project / coding), which the planner/completion evals can use.

    Never raises: feedback collection must not affect the chat response path.
    """
    state = state or {}
    try:
        errors = state.get("_errors") or []
        if not isinstance(errors, list):
            errors = []
        generation_failed = bool(state.get("_generation_failed"))
        answer = answer or state.get("answer", "")
        failed = generation_failed or (not answer) or bool(errors)
        goal_type = get_goal_type(state)

        if not failed and goal_type not in {"project", "coding"}:
            return False

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "session_id": session_id,
            "trace_id": state.get("trace_id", ""),
            "question": _truncate(question or state.get("question", ""), _QUESTION_TRUNCATE),
            "goal_type": goal_type,
            "outcome": "failure" if failed else "success",
            "answer": _truncate(answer, _ANSWER_TRUNCATE),
            "errors": errors[:10],
            "generation_failed": generation_failed,
        }

        path = feedback_path()
        key = _record_key(record)
        with _write_lock:
            if key in _recent_keys(path):
                return False
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug(
            "Feedback recorded: outcome=%s goal_type=%s",
            record["outcome"], goal_type,
        )
        return True
    except Exception as exc:
        logger.warning("Feedback recording failed (non-fatal): %s", exc)
        return False


def record_chat_feedback(
    state: dict[str, Any] | None,
    session_id: int | str | None = None,
    question: str = "",
    answer: str = "",
) -> bool:
    """Convenience wrapper used by the chat endpoints."""
    return record_feedback(
        state,
        session_id=session_id,
        question=question,
        answer=answer,
        source="chat",
    )
