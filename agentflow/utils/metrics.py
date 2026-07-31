"""Lightweight in-process metrics, exported in Prometheus text format.

Kept intentionally dependency-free: a few counters and duration aggregates
behind a lock, rendered as ``text/plain; version=0.0.4`` at ``/metrics``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
# name -> tuple(sorted label pairs) -> value
_counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)


def inc(name: str, **labels: str | int) -> None:
    key = tuple(sorted((k, str(v)) for k, v in labels.items()))
    with _lock:
        _counters[name][key] = _counters[name].get(key, 0.0) + 1.0


def add(name: str, value: float, **labels: str | int) -> None:
    key = tuple(sorted((k, str(v)) for k, v in labels.items()))
    with _lock:
        _counters[name][key] = _counters[name].get(key, 0.0) + value


def observe_duration(name: str, seconds: float, **labels: str | int) -> None:
    """Record a duration: increments ``<name>_count`` and ``<name>_sum``."""
    inc(f"{name}_count", **labels)
    add(f"{name}_sum", seconds, **labels)


def timed(name: str, **labels: str | int):
    """Decorator: measure the wrapped callable's wall time into *name*."""
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                observe_duration(name, time.perf_counter() - start, **labels)
        return _wrapper
    return _decorator


def reset() -> None:
    with _lock:
        _counters.clear()


def render() -> str:
    """Render all counters in Prometheus text exposition format."""
    lines: list[str] = []
    with _lock:
        for name in sorted(_counters):
            for key, value in sorted(_counters[name].items()):
                if key:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in key) + "}"
                else:
                    label_str = ""
                lines.append(f"{name}{label_str} {value:g}")
    return "\n".join(lines) + ("\n" if lines else "")
