from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from auditrag.config import get_settings

settings = get_settings()
engine = create_engine(settings.postgres_url, future=True, pool_pre_ping=True)
qdrant_client = QdrantClient(url=settings.qdrant_url)

router = APIRouter(tags=["health"])


def check_postgres() -> tuple[bool, str | None]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except SQLAlchemyError as error:
        return False, str(error)


def check_qdrant() -> tuple[bool, str | None]:
    try:
        qdrant_client.get_collections()
        return True, None
    except Exception as error:  # noqa: BLE001
        return False, str(error)


def run_dependency_checks() -> dict:
    postgres_ok, postgres_error = check_postgres()
    qdrant_ok, qdrant_error = check_qdrant()
    status = "ok" if postgres_ok and qdrant_ok else "degraded"
    return {
        "status": status,
        "checks": {
            "postgres": {"ok": postgres_ok, "error": postgres_error},
            "qdrant": {"ok": qdrant_ok, "error": qdrant_error},
        },
    }


@router.get("/health")
def health_check() -> JSONResponse:
    result = run_dependency_checks()
    code = 200 if result["status"] == "ok" else 503
    return JSONResponse(status_code=code, content=result)
