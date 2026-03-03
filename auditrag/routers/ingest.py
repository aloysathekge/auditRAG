"""POST /ingest: trigger document ingestion via API."""
from fastapi import APIRouter, HTTPException

from auditrag.ingestion import (
    download_pdf,
    extract_text,
    chunk_pages,
    embed_chunks,
    upsert_chunks_to_qdrant,
    load_financebench,
)
from auditrag.retrieval.sparse import invalidate_sparse_cache
from auditrag.schemas.ingest import IngestRequest

router = APIRouter(tags=["ingest"])


def _run_pipeline(doc_name: str, doc_link: str, skip_if_exists: bool = True) -> dict:
    pdf_path = download_pdf(doc_link, doc_name)
    pages = extract_text(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from PDF for {doc_name}")
    chunks = chunk_pages(pages, doc_name)
    chunks = embed_chunks(chunks)
    n = upsert_chunks_to_qdrant(chunks, skip_if_doc_exists=skip_if_exists)
    return {"doc_name": doc_name, "chunks_created": len(chunks), "chunks_upserted": n}


@router.post("/ingest")
def post_ingest(body: IngestRequest, skip_if_exists: bool = True) -> dict:
    """Ingest a single document by URL. Returns chunks_created and chunks_upserted."""
    try:
        result = _run_pipeline(body.doc_name, body.doc_link, skip_if_exists=skip_if_exists)
        if result["chunks_upserted"] > 0:
            invalidate_sparse_cache()
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")


@router.post("/ingest/financebench")
def post_ingest_financebench(index: int = 0, skip_if_exists: bool = True) -> dict:
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
        result = _run_pipeline(doc_name, doc_link, skip_if_exists=skip_if_exists)
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
) -> dict:
    """Ingest multiple FinanceBench documents. Deduplicates by doc_name (dataset has duplicate rows)."""
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
            result = _run_pipeline(doc_name, doc_link, skip_if_exists=skip_if_exists)
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
