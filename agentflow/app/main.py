"""FastAPI application entry point with background cleanup task."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from agentflow.api.routes import get_store, router
from agentflow.config.settings import settings
from agentflow.graph.workflow import get_executor, reset_workflow_cache
from agentflow.utils.logging import build_logger

logger = build_logger("agentflow")

# ------------------------------------------------------------------
# Background cleanup task
# ------------------------------------------------------------------


async def _cleanup_loop() -> None:
    """Periodically delete expired sessions and memories."""
    interval = settings.cleanup_interval_minutes
    logger.info(
        "Cleanup task started: interval=%dmin, session_ttl=%dh, memory_ttl=%dd",
        interval, settings.session_ttl_hours, settings.memory_ttl_days,
    )
    while True:
        try:
            await asyncio.sleep(interval * 60)
            store = get_store()

            sess_count = store.delete_sessions_older_than(settings.session_ttl_hours)
            mem_count = store.delete_old_memories(settings.memory_ttl_days)

            if sess_count or mem_count:
                logger.info("Cleanup: removed %d sessions, %d memories", sess_count, mem_count)
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            return
        except Exception:
            logger.exception("Cleanup task error (non-fatal)")


_cleanup_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage startup/shutdown lifecycle."""
    global _cleanup_task
    reset_workflow_cache()
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("%s started (debug=%s)", settings.app_name, settings.debug)
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
        logger.info("Cleanup task stopped")


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

# Enable CORS for local frontend development (Vite ports).
# In production you should restrict origins appropriately.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── HTTP request logging middleware ──


@app.middleware("http")
async def log_http_requests(request: Request, call_next: object) -> object:
    """Log method, path, status code, and duration for every HTTP request."""
    start = time.perf_counter()
    response = await call_next(request)  # type: ignore[operator]
    duration = time.perf_counter() - start
    logger.info(
        "HTTP %s %s → %d (%.2fs)",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


@app.get("/health")
def healthcheck() -> dict[str, object]:
    """Health check endpoint with component-level status."""
    checks: dict[str, dict] = {
        "database": _check_db(),
        "llm_config": _check_llm_config(),
    }

    executor = get_executor()
    if executor:
        checks["executor"] = {"ok": True, "tools": executor.list_tools()}

    all_ok = all(c.get("ok", False) for c in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "service": settings.app_name,
        "checks": checks,
    }


def _check_db() -> dict:
    """Check SQLite database connectivity."""
    try:
        store = get_store()
        store.list_sessions(limit=1)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_llm_config() -> dict:
    """Check LLM configuration (model name + API key presence, not connectivity)."""
    if settings.deepseek_api_key:
        return {"ok": True, "model": settings.model_name}
    return {"ok": False, "error": "DEEPSEEK_API_KEY is not configured"}
