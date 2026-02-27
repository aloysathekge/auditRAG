from pathlib import Path

import httpx
from datasets import load_dataset

DATA_DIR = Path("data/financebench/pdfs")


def load_financebench():
    """Load the FinanceBench dataset from HuggingFace."""
    return load_dataset("PatronusAI/financebench", split="train")


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
