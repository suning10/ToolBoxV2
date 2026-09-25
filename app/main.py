"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from asgi_correlation_id import CorrelationIdMiddleware

from app.api.v1.api import api_router
from app.api.v1.chatbot import agent
from app.core.cache import cache_service
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import setup_metrics
from app.core.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
    ProfilingMiddleware,
)
from app.core.observability import langfuse_init
from app.services.database import database_service
from app.services.memory import memory_service
from app.services.run_stream import run_stream_service

# Load environment variables
load_dotenv()
langfuse_init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up shared services on startup and release them on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        api_prefix=settings.API_V1_STR,
        environment=settings.ENVIRONMENT.value,
    )

    # The cache and the memory warm-up are optimisations: the services degrade gracefully
    # (Valkey get/set no-op without a client; mem0 initialises lazily on first use), so a
    # failure here must not stop the app from serving.
    try:
        await cache_service.initialize()
    except Exception as e:
        logger.warning("cache_initialization_failed_continuing_without_cache", error=str(e))
    # Unlike the cache this is required for streaming, so it falls back to an in-process buffer
    # (rather than to nothing) if Valkey is configured but unreachable.
    await run_stream_service.initialize()
    try:
        await memory_service.initialize()
    except Exception as e:
        logger.warning("memory_warmup_failed_will_retry_on_first_use", error=str(e))

    # Builds the graph and its Postgres checkpointer. Raises in dev, degrades in production.
    await agent.create_graph()

    try:
        yield
    finally:
        logger.info("application_shutdown")
        try:
            # First, so in-flight runs can record an "interrupted" event while their backend is still open
            await run_stream_service.shutdown()
        finally:
            try:
                await agent.close()
            finally:
                await cache_service.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Prometheus middleware + /metrics
setup_metrics(app)

# Starlette runs the last-added middleware outermost, so the correlation id is set
# before anything that logs, and CORS answers preflights before the rest run.
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingContextMiddleware)
if settings.DEBUG:
    app.add_middleware(ProfilingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    # Never combine credentials with a wildcard origin
    allow_credentials="*" not in settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Run-Id"],  # so browser clients can read the run id needed to re-attach
)
app.add_middleware(CorrelationIdMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation failures as a flat, client-friendly 422.

    Args:
        request: The request that failed validation.
        exc: The validation error raised by FastAPI.

    Returns:
        JSONResponse: A 422 listing each offending field and why.
    """
    errors = [
        {"field": ".".join(str(part) for part in error["loc"] if part != "body"), "message": error["msg"]}
        for error in exc.errors()
    ]
    logger.warning("validation_error", path=request.url.path, error_count=len(errors))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Validation error", "errors": errors},
    )


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint.

    Args:
        request: The request object, required for rate limiting.

    Returns:
        dict: Basic service information.
    """
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs_url": "/docs",
    }


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request):
    """Report service health, including database connectivity.

    Args:
        request: The request object, required for rate limiting.

    Returns:
        JSONResponse: 200 when the database is reachable, 503 otherwise.
    """
    db_healthy = await database_service.health_check()
    return JSONResponse(
        status_code=status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if db_healthy else "degraded",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT.value,
            "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
            "timestamp": datetime.now().isoformat(),
        },
    )
