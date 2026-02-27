"""GET /metrics for aggregate stats (p50/p95 latency, avg cost, etc.). To be implemented."""
from fastapi import APIRouter

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def get_metrics() -> dict:
    """Placeholder until query telemetry is persisted."""
    return {"message": "Metrics not yet implemented", "latency_p50_ms": None, "latency_p95_ms": None, "avg_cost_usd": None}
