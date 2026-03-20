from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auditrag.core.config import get_settings
from auditrag.db.session import init_db
from auditrag.routers.health import router as health_router
from auditrag.routers.ingest import router as ingest_router
from auditrag.routers.financebench import router as financebench_router
from auditrag.routers.documents import router as documents_router
from auditrag.routers.metrics import router as metrics_router
from auditrag.routers.query import router as query_router
from auditrag.routers.evaluate import router as evaluate_router

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
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://auditrag.aloysathekge.com",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(financebench_router)
app.include_router(documents_router)
app.include_router(metrics_router)
app.include_router(evaluate_router)
