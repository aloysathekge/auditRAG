from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from auditrag.core.config import get_settings
from auditrag.db.session import check_postgres

settings = get_settings()
qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
    check_compatibility=False,
)

router = APIRouter(tags=["health"])


def check_qdrant() -> tuple[bool, str | None]:
    try:
        qdrant_client.get_collections()
        return True, None
    except Exception as e:
        return False, str(e)


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
