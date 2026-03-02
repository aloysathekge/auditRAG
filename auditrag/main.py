from contextlib import asynccontextmanager

from fastapi import FastAPI

from auditrag.core.config import get_settings
from auditrag.db.session import init_db
from auditrag.routers.health import router as health_router
from auditrag.routers.ingest import router as ingest_router
from auditrag.routers.metrics import router as metrics_router
from auditrag.routers.query import router as query_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(metrics_router)
