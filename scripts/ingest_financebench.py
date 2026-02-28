"""Run ingestion for FinanceBench sample: uv run python scripts/ingest_financebench.py"""
from auditrag.ingestion import (
    load_financebench,
    download_pdf,
    extract_text,
    chunk_pages,
    embed_chunks,
    upsert_chunks_to_qdrant,
)

if __name__ == "__main__":
    ds = load_financebench()
    print(f"Total records: {len(ds)}")
    sample = ds[0]
    doc_name = sample["doc_name"]
    doc_link = sample["doc_link"]
    print(f"Doc: {doc_name}")

    pdf_path = download_pdf(doc_link, doc_name)
    pages = extract_text(pdf_path)
    chunks = chunk_pages(pages, doc_name)
    chunks = embed_chunks(chunks)
    n = upsert_chunks_to_qdrant(chunks)
    if n > 0:
        from auditrag.retrieval.sparse import invalidate_sparse_cache
        invalidate_sparse_cache()
    print("Done.")
