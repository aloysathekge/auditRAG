from fastapi import FastAPI

from auditrag.config import get_settings
from auditrag.health import router as health_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(health_router)
