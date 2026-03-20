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


def _ensure_qdrant_indexes():
    """Create payload indexes on existing Qdrant collection at startup."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PayloadSchemaType
        from auditrag.ingestion.embedder import QDRANT_COLLECTION
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            check_compatibility=False,
        )
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION in collections:
            for field in ("knowledge_base", "doc_name"):
                try:
                    client.create_payload_index(
                        collection_name=QDRANT_COLLECTION,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    _ensure_qdrant_indexes()
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
