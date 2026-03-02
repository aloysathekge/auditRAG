"""Log query telemetry to PostgreSQL for metrics."""
from auditrag.db.models import QueryLog
from auditrag.db.session import get_db_session


def log_query(
    question: str,
    total_ms: int,
    retrieve_ms: int,
    generate_ms: int | None = None,
    cost_usd: float | None = None,
    model_used: str | None = None,
) -> None:
    """Insert one row into query_log. Swallows errors so logging never breaks the request."""
    try:
        session = get_db_session()
        try:
            session.add(
                QueryLog(
                    question=question,
                    total_ms=total_ms,
                    retrieve_ms=retrieve_ms,
                    generate_ms=generate_ms,
                    cost_usd=cost_usd,
                    model_used=model_used,
                )
            )
            session.commit()
        finally:
            session.close()
    except Exception:
        pass
