from pathlib import Path

import httpx
import pdfplumber
from datasets import load_dataset

DATA_DIR = Path("data/financebench/pdfs")


def load_financebench():
    """Load the FinanceBench dataset from HuggingFace."""
    ds = load_dataset("PatronusAI/financebench", split="train")
    return ds


def download_pdf(url: str, doc_name: str) -> Path:
    """Download a PDF if not already cached locally. Returns the file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{doc_name}.pdf"
    if path.exists():
        print(f"  Already cached: {path}")
        return path

    print(f"  Downloading: {url}")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"  Saved: {path}")
    return path


def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


if __name__ == "__main__":
    ds = load_financebench()
    print(f"Total records: {len(ds)}")

    # Pick the first record as a demo
    sample = ds[0]
    doc_name = sample["doc_name"]
    doc_link = sample["doc_link"]
    print(f"\nDoc: {doc_name}")
    print(f"Link: {doc_link}")

    # Step 1: Download
    pdf_path = download_pdf(doc_link, doc_name)

    # Step 2: Extract text
    pages = extract_text(pdf_path)
    print(f"\nExtracted {len(pages)} pages with text")

    # Show first page preview
    if pages:
        print(f"\n--- Page {pages[0]['page']} (first 500 chars) ---")
        print(pages[0]["text"][:500])
