"""Run ingestion pipeline: python -m auditrag.ingestion"""
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
    print(f"\nDoc: {doc_name}")
    print(f"Link: {doc_link}")

    pdf_path = download_pdf(doc_link, doc_name)
    pages = extract_text(pdf_path)
    print(f"\nExtracted {len(pages)} pages with text")

    chunks = chunk_pages(pages, doc_name)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")

    upsert_chunks_to_qdrant(chunks)
    print("Done.")
