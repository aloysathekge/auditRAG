"""FinanceBench-specific ingestion endpoints."""
from fastapi import APIRouter, HTTPException

from auditrag.ingestion import load_financebench
from auditrag.routers.ingest import _run_pipeline
from auditrag.ingestion.loader import download_pdf
from auditrag.retrieval.sparse import invalidate_sparse_cache

router = APIRouter(tags=["financebench"])


@router.post("/ingest/financebench")
def post_ingest_financebench(
    index: int = 0,
    skip_if_exists: bool = True,
    knowledge_base: str = "financebench",
) -> dict:
    """Ingest a FinanceBench document by index (0 = first record)."""
    ds = load_financebench()
    if index < 0 or index >= len(ds):
        raise HTTPException(
            status_code=400,
            detail=f"Index {index} out of range (dataset has {len(ds)} records)",
        )
    sample = ds[index]
    doc_name = sample["doc_name"]
    doc_link = sample["doc_link"]
    try:
        pdf_path = download_pdf(doc_link, doc_name)
        result = _run_pipeline(doc_name, pdf_path, knowledge_base=knowledge_base, skip_if_exists=skip_if_exists)
        if result["chunks_upserted"] > 0:
            invalidate_sparse_cache()
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")


@router.post("/ingest/financebench/bulk")
def post_ingest_financebench_bulk(
    limit: int = 10,
    start: int = 0,
    skip_if_exists: bool = True,
    knowledge_base: str = "financebench",
) -> dict:
    """Ingest multiple FinanceBench documents. Deduplicates by doc_name."""
    ds = load_financebench()
    total = len(ds)
    start_idx = max(0, start)
    end_idx = min(start_idx + limit, total)
    if start_idx >= total:
        raise HTTPException(
            status_code=400,
            detail=f"start={start} beyond dataset size ({total})",
        )
    seen: set[str] = set()
    to_ingest: list[tuple[str, str]] = []
    for i in range(start_idx, end_idx):
        sample = ds[i]
        doc_name = sample["doc_name"]
        doc_link = sample["doc_link"]
        if doc_name in seen:
            continue
        seen.add(doc_name)
        to_ingest.append((doc_name, doc_link))
    results = []
    any_upserted = False
    try:
        for doc_name, doc_link in to_ingest:
            pdf_path = download_pdf(doc_link, doc_name)
            result = _run_pipeline(doc_name, pdf_path, knowledge_base=knowledge_base, skip_if_exists=skip_if_exists)
            results.append(result)
            if result["chunks_upserted"] > 0:
                any_upserted = True
        if any_upserted:
            invalidate_sparse_cache()
        return {
            "status": "ok",
            "range_indices": end_idx - start_idx,
            "unique_docs": len(to_ingest),
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")
