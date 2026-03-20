from fastapi import APIRouter
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


@router.get("/health")
def health_check() -> dict:
    """Health check returning status for frontend consumption.

    Always returns 200 so the frontend can read the body and display
    per-service status rather than treating the whole response as an error.
    """
    postgres_ok, _ = check_postgres()
    qdrant_ok, _ = check_qdrant()

    overall = "ok" if postgres_ok and qdrant_ok else "degraded"

    return {
        "status": overall,
        "postgres": "connected" if postgres_ok else "disconnected",
        "qdrant": "connected" if qdrant_ok else "disconnected",
    }
