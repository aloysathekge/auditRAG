from fastapi import FastAPI

from auditrag.config import get_settings
from auditrag.api.health import router as health_router
from auditrag.api.query import router as query_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(query_router)
