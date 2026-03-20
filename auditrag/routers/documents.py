"""Document management endpoints: list and delete documents."""
from fastapi import APIRouter, HTTPException

from auditrag.ingestion.embedder import delete_doc_from_qdrant, list_docs_in_qdrant
from auditrag.retrieval.sparse import invalidate_sparse_cache

router = APIRouter(tags=["documents"])


@router.get("/documents")
def get_documents(knowledge_base: str | None = None) -> dict:
    """List all documents, optionally filtered by knowledge base."""
    docs = list_docs_in_qdrant(knowledge_base=knowledge_base)
    return {
        "knowledge_base": knowledge_base or "all",
        "documents": docs,
        "total": len(docs),
    }


@router.delete("/documents/{doc_name}")
def delete_document(doc_name: str, knowledge_base: str = "default") -> dict:
    """Delete all chunks for a document from the knowledge base."""
    result = delete_doc_from_qdrant(doc_name, knowledge_base=knowledge_base)
    if result == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_name}' not found in knowledge base '{knowledge_base}'",
        )
    invalidate_sparse_cache()
    return {"status": "ok", "doc_name": doc_name, "knowledge_base": knowledge_base}
