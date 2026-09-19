from __future__ import annotations

import json
import logging
import time
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from agentflow.config.settings import settings


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Rotating handler that tolerates a log file another process holds open.

    The API server, evaluation runs and the test suite all write into the same
    ``logs/`` directory.  On Windows the rollover rename fails with
    ``PermissionError: [WinError 32]`` while another process has the file open,
    and the standard handler then prints a ``--- Logging error ---`` traceback
    for every subsequent record.  Skipping the rotation and retrying at the next
    interval keeps logging usable instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("delay", True)
        super().__init__(*args, **kwargs)

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except OSError:
            # Another process still holds the file — retry next interval.
            self.rolloverAt = int(time.time()) + self.interval


class JsonFormatter(logging.Formatter):
    """Output log records as JSON lines for machine parsing (ELK/Loki)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id") and record.trace_id:
            log_entry["trace_id"] = record.trace_id
        # Include exception info when present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_entry["exception"] = record.exc_text
        return json.dumps(log_entry, ensure_ascii=False)


class TraceIdFilter(logging.Filter):
    """Inject ``trace_id`` from contextvars into every log record.

    Registered once in ``build_logger``; all loggers share the same
    ``contextvars.ContextVar`` so the trace_id propagates
    automatically across async boundaries.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from agentflow.utils.trace_context import get_trace_id

            tid = get_trace_id()
            if tid:
                record.trace_id = tid
        except Exception:
            pass
        return True


def _build_formatter() -> logging.Formatter:
    """Return the formatter configured by settings.log_format."""
    if settings.log_format == "json":
        return JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    return logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_logger(name: str) -> logging.Logger:
    """Create a configured logger for the application."""
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = _build_formatter()

        file_handler = SafeTimedRotatingFileHandler(
            settings.logs_dir / f"{name}.log",
            when=settings.log_rotation_when,
            backupCount=settings.log_rotation_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Register the TraceIdFilter once so all loggers auto-inject trace_id
        logger.addFilter(TraceIdFilter())

    return logger
