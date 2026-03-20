import re
from pathlib import Path

import httpx
from datasets import load_dataset

from auditrag.core.config import get_settings

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


def _sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filenames."""
    return re.sub(r"[^\w\-.]", "_", name)


def save_uploaded_pdf(file_bytes: bytes, doc_name: str, knowledge_base: str = "default") -> Path:
    """Save uploaded PDF bytes to disk. Returns the file path."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir) / _sanitize_filename(knowledge_base)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(doc_name)
    path = upload_dir / f"{safe_name}.pdf"
    path.write_bytes(file_bytes)
    print(f"  Saved upload: {path}")
    return path
