"""FastAPI application factory.

Run locally:
    uv run uvicorn review_insight.main:create_app --factory --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from review_insight import __version__
from review_insight.config import AppEnv, Settings, get_settings
from review_insight.observability.logging_config import configure_logging

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str
    env: AppEnv


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully configured app. Tests pass their own Settings; production reads the env."""
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: later tasks open the DB pool (T11) and recover batch jobs (T15) here.
        logger.info("app starting", extra={"version": __version__, "env": settings.app_env})
        yield
        # Shutdown: close pools, flush Langfuse buffers (T9, T11).
        logger.info("app stopped")

    app = FastAPI(
        title="ReviewInsight API",
        version=__version__,
        summary="Aspect-based sentiment, urgency and routing for Vietnamese reviews.",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness probe: the process is up. Deliberately checks nothing else."""
        return HealthResponse(status="ok", version=__version__, env=settings.app_env)

    return app
