from fastapi import FastAPI

from agentflow.api.routes import router
from agentflow.config.settings import settings
from agentflow.utils.logging import build_logger

logger = build_logger("agentflow")

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok", "service": settings.app_name}
