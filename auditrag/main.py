from fastapi import FastAPI

from auditrag.core.config import get_settings
from auditrag.routers.health import router as health_router
from auditrag.routers.query import router as query_router
from auditrag.routers.ingest import router as ingest_router
from auditrag.routers.metrics import router as metrics_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(metrics_router)
