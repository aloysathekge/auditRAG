"""POST /ingest and /upload: trigger document ingestion via API."""
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from auditrag.core.config import get_settings
from auditrag.ingestion import (
    download_pdf,
    extract_text,
    chunk_pages,
    embed_chunks,
    upsert_chunks_to_qdrant,
    save_uploaded_pdf,
)
from auditrag.db.models import Document, QAPair
from auditrag.db.session import SessionLocal
from auditrag.generation.qa_generator import generate_qa_pairs
from auditrag.retrieval.sparse import invalidate_sparse_cache
from auditrag.schemas.ingest import IngestRequest

router = APIRouter(tags=["ingest"])


def _run_pipeline(
    doc_name: str,
    pdf_path: Path,
    knowledge_base: str = "default",
    skip_if_exists: bool = True,
) -> dict:
    """Shared pipeline: parse -> chunk -> embed -> upsert."""
    pages = extract_text(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from PDF for {doc_name}")
    chunks = chunk_pages(pages, doc_name, knowledge_base=knowledge_base)
    chunks = embed_chunks(chunks)
    n = upsert_chunks_to_qdrant(chunks, skip_if_doc_exists=skip_if_exists)

    # Track in Postgres and generate Q&A pairs (best-effort)
    qa_count = 0
    if n > 0:
        try:
            session = SessionLocal()
            doc = Document(
                doc_name=doc_name,
                knowledge_base=knowledge_base,
                filename=pdf_path.name,
                chunks_count=len(chunks),
            )
            session.merge(doc)
            session.commit()

            # Auto-generate Q&A pairs for evaluation
            qa_pairs = generate_qa_pairs(chunks)
            for pair in qa_pairs:
                session.add(QAPair(
                    doc_name=pair["doc_name"],
                    knowledge_base=pair["knowledge_base"],
                    chunk_id=pair["chunk_id"],
                    question=pair["question"],
                    answer=pair["answer"],
                ))
            session.commit()
            qa_count = len(qa_pairs)
            session.close()
        except Exception:
            pass

    return {
        "doc_name": doc_name,
        "knowledge_base": knowledge_base,
        "chunks_created": len(chunks),
        "chunks_upserted": n,
        "qa_pairs_generated": qa_count,
    }


@router.post("/upload")
def post_upload(
    file: UploadFile = File(...),
    doc_name: str | None = Form(None),
    knowledge_base: str = Form("default"),
    skip_if_exists: bool = Form(True),
) -> dict:
    """Upload a PDF file and ingest it into the knowledge base."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    settings = get_settings()
    content = file.file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_size_mb} MB.",
        )

    if not doc_name:
        doc_name = file.filename.rsplit(".", 1)[0]

    try:
        pdf_path = save_uploaded_pdf(content, doc_name, knowledge_base)
        result = _run_pipeline(doc_name, pdf_path, knowledge_base=knowledge_base, skip_if_exists=skip_if_exists)
        if result["chunks_upserted"] > 0:
            invalidate_sparse_cache()
        return {"status": "ok", "doc_id": f"{knowledge_base}_{doc_name}", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")


@router.post("/ingest")
def post_ingest(body: IngestRequest, skip_if_exists: bool = True) -> dict:
    """Ingest a single document by URL. Returns chunks_created and chunks_upserted."""
    try:
        pdf_path = download_pdf(body.doc_link, body.doc_name)
        result = _run_pipeline(
            body.doc_name, pdf_path,
            knowledge_base=body.knowledge_base,
            skip_if_exists=skip_if_exists,
        )
        if result["chunks_upserted"] > 0:
            invalidate_sparse_cache()
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e!s}")
