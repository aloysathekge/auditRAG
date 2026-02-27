from pathlib import Path

import pdfplumber


def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages
