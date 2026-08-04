"""
document_loader.py

Purpose (per SDD Section 5 & 10):
- Load a legal PDF (starting with the Constitution of India).
- Extract raw text page by page.
- Verify the PDF path exists and print pages extracted (debugging strategy).

This module does ONE job: PDF -> raw text per page.
Chunking, cleaning, and metadata tagging happen in later modules
(chunker.py), kept deliberately separate so each stage is testable
in isolation.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF and extract text page by page.

    Args:
        pdf_path: path to the PDF file.

    Returns:
        A list of dicts, one per page:
        [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}, ...]

    Raises:
        FileNotFoundError: if pdf_path does not exist.
    """
    path = Path(pdf_path)

    # Debugging Strategy step 1: verify PDF path
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {path.resolve()}")

    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_number": i, "text": text})

    return pages


if __name__ == "__main__":
    # Manual test run — this is exactly the "print pages extracted" step
    # from the SDD's Debugging Strategy (Section 10).
    PDF_PATH = "../../datasets/raw/constitution_of_india.pdf"

    print(f"Loading PDF from: {PDF_PATH}")
    pages = load_pdf(PDF_PATH)

    print(f"\nTotal pages extracted: {len(pages)}\n")

    # Print a preview of the first 2 pages so we can visually confirm
    # extraction quality (garbled text = OCR/PDF issue to fix now,
    # not later when it's buried under embeddings).
    for p in pages[:2]:
        print(f"--- Page {p['page_number']} (first 300 chars) ---")
        print(p["text"][:300])
        print()