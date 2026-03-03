"""Bulk ingest FinanceBench docs: uv run python scripts/ingest_financebench_bulk.py [--limit N] [--start M]

Examples:
  uv run python scripts/ingest_financebench_bulk.py --limit 5   # first 5 docs
  uv run python scripts/ingest_financebench_bulk.py --limit 20  # first 20 docs
  uv run python scripts/ingest_financebench_bulk.py --limit 10 --start 5  # docs at indices 5..14
"""
import argparse

from auditrag.ingestion import (
    load_financebench,
    download_pdf,
    extract_text,
    chunk_pages,
    embed_chunks,
    upsert_chunks_to_qdrant,
)


def ingest_one(doc_name: str, doc_link: str, skip_if_exists: bool = True) -> int:
    """Run full pipeline for one doc. Returns chunks_upserted (0 if skipped)."""
    pdf_path = download_pdf(doc_link, doc_name)
    pages = extract_text(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from PDF for {doc_name}")
    chunks = chunk_pages(pages, doc_name)
    chunks = embed_chunks(chunks)
    n = upsert_chunks_to_qdrant(chunks, skip_if_doc_exists=skip_if_exists)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest FinanceBench documents.")
    parser.add_argument("--limit", type=int, default=10, help="Number of docs to ingest (default 10)")
    parser.add_argument("--start", type=int, default=0, help="Start at this dataset index (default 0)")
    args = parser.parse_args()

    ds = load_financebench()
    total = len(ds)
    start = max(0, args.start)
    end = min(start + args.limit, total)

    if start >= total:
        print(f"No documents to ingest (start={start}, dataset size={total}).")
        return

    # Build list of unique (doc_name, doc_link) by first occurrence in range (dataset has duplicate doc_names)
    seen_doc_names: set[str] = set()
    to_ingest: list[tuple[str, str]] = []
    for i in range(start, end):
        row = ds[i]
        doc_name = row["doc_name"]
        doc_link = row["doc_link"]
        if doc_name in seen_doc_names:
            continue
        seen_doc_names.add(doc_name)
        to_ingest.append((doc_name, doc_link))

    num_unique = len(to_ingest)
    print(f"FinanceBench: {total} records. Indices {start}..{end - 1} → {num_unique} unique doc(s).\n")
    any_upserted = False

    for idx, (doc_name, doc_link) in enumerate(to_ingest, 1):
        print(f"[{idx}/{num_unique}] {doc_name} ... ", end="", flush=True)
        try:
            n = ingest_one(doc_name, doc_link, skip_if_exists=True)
            if n > 0:
                print(f"upserted {n} chunks")
                any_upserted = True
            else:
                print("skipped (already in Qdrant)")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    if any_upserted:
        from auditrag.retrieval.sparse import invalidate_sparse_cache
        invalidate_sparse_cache()
        print("\nBM25 cache invalidated (rebuilds on next search).")

    print("\nDone.")


if __name__ == "__main__":
    main()
