"""GET /metrics for aggregate stats from query_log."""
from sqlalchemy import select

from fastapi import APIRouter

from auditrag.db.models import QueryLog
from auditrag.db.session import get_db_session

router = APIRouter(tags=["metrics"])

# Use last N rows for percentile/avg (avoid huge scans)
METRICS_LIMIT = 10_000


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """p in [0, 100]. Returns None if empty."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


@router.get("/metrics")
def get_metrics() -> dict:
    """Aggregate metrics from query_log: p50/p95 latency (ms), avg cost (USD), count."""
    try:
        session = get_db_session()
        try:
            stmt = (
                select(QueryLog.total_ms, QueryLog.cost_usd)
                .order_by(QueryLog.created_at.desc())
                .limit(METRICS_LIMIT)
            )
            result = session.execute(stmt)
            rows = result.all()
        finally:
            session.close()
    except Exception:
        return {
            "message": "Metrics unavailable (e.g. query_log not created or Postgres down)",
            "count": 0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "avg_cost_usd": None,
        }

    if not rows:
        return {
            "message": "No query telemetry yet. Run some /query requests first.",
            "count": 0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "avg_cost_usd": None,
        }

    total_ms_list = sorted(r[0] for r in rows)
    cost_list = [r[1] for r in rows if r[1] is not None]

    p50 = _percentile(total_ms_list, 50)
    p95 = _percentile(total_ms_list, 95)
    avg_cost = sum(cost_list) / len(cost_list) if cost_list else None

    return {
        "count": len(rows),
        "latency_p50_ms": round(p50, 2) if p50 is not None else None,
        "latency_p95_ms": round(p95, 2) if p95 is not None else None,
        "avg_cost_usd": round(avg_cost, 6) if avg_cost is not None else None,
    }
