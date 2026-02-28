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
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")
