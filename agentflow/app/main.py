"""FastAPI application entry point with background cleanup task."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from agentflow.api import (
    chat,
    executions,
    files_workspace,
    knowledge,
    memory,
    models,
    sessions,
    system,
)
from agentflow.api.routes import get_store
from agentflow.config.settings import settings
from agentflow.graph.workflow import get_executor, reset_workflow_cache
from agentflow.utils.metrics import inc, observe_duration, render
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


class DynamicCORSMiddleware(CORSMiddleware):
    """CORS middleware that reads ``CORS_ORIGINS`` at request time.

    The default build uses the localhost-only regex; when ``CORS_ORIGINS`` is
    set (comma-separated), exactly those origins are allowed.  Reading the
    setting per-request keeps the behaviour testable and restart-free.
    """

    def is_allowed_origin(self, origin: str) -> bool:
        configured = settings.cors_origins
        if configured:
            allowed = [
                item.strip()
                for item in configured.split(",")
                if item.strip()
            ]
            return origin in allowed
        return super().is_allowed_origin(origin)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage startup/shutdown lifecycle."""
    global _cleanup_task
    _validate_required_env()
    reset_workflow_cache()
    _log_capability_status()
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("%s started (debug=%s)", settings.app_name, settings.debug)
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
        logger.info("Cleanup task stopped")


def _log_capability_status() -> None:
    """Log which optional capabilities are configured.

    The system degrades gracefully when keys are missing (embedding intent
    matching falls back to the LLM, knowledge retrieval to lexical-only,
    web search to DuckDuckGo).  Logging the gaps at startup makes demo-day
    configuration mistakes visible immediately.
    """
    missing: list[str] = []
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY (核心 LLM 未配置)")
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY (向量检索与意图 embedding 快路径走降级)")
    if not os.environ.get("TAVILY_API_KEY", ""):
        missing.append("TAVILY_API_KEY (网页搜索仅剩 DuckDuckGo 降级)")
    if not os.environ.get("COMPOSIO_API_KEY", ""):
        missing.append("COMPOSIO_API_KEY (Composio 集成不可用)")

    if missing:
        logger.warning(
            "Capability check: %d optional capability(ies) not configured -> %s",
            len(missing), "；".join(missing),
        )
    else:
        logger.info("Capability check: all optional capabilities configured")


def _validate_required_env() -> None:
    """Fail fast in production when required credentials are missing.

    In development the app keeps its graceful-degradation behavior; in
    production (``APP_ENV=production`` or ``ENFORCE_REQUIRED_ENV=true``)
    a missing core API key is a deployment error, not a runtime fallback.
    """
    if not (settings.enforce_required_env or settings.app_env == "production"):
        return

    missing: list[str] = []
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if missing:
        raise RuntimeError(
            "Refusing to start in production: missing required environment "
            f"variable(s): {', '.join(missing)}"
        )

app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

# Enable CORS for local frontend development (Vite ports) by default.
# Set CORS_ORIGINS for a deployed origin allow-list (see DynamicCORSMiddleware).
app.add_middleware(
    DynamicCORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(executions.router)
app.include_router(files_workspace.router)
app.include_router(knowledge.router)
app.include_router(memory.router)
app.include_router(models.router)
app.include_router(sessions.router)
app.include_router(system.router)


# ── HTTP request logging middleware ──


@app.middleware("http")
async def log_http_requests(request: Request, call_next: object) -> object:
    """Log method, path, status code, and duration for every HTTP request."""
    start = time.perf_counter()
    # Optional bearer-token auth for non-local deployments. /health stays open
    # so load-balancer probes keep working.
    if settings.auth_token:
        path = request.url.path
        if path != "/health":
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {settings.auth_token}":
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)  # type: ignore[operator]
    duration = time.perf_counter() - start
    inc("http_requests_total", method=request.method, path=request.url.path,
        status=str(response.status_code))
    observe_duration("http_request_duration_seconds", duration,
                     method=request.method, path=request.url.path)
    logger.info(
        "HTTP %s %s → %d (%.2fs)",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus text-format metrics for the running process."""
    return Response(
        content=render(),
        media_type="text/plain; version=0.0.4",
    )


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
