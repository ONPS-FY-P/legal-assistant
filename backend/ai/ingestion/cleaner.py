"""
cleaner.py

Purpose:
- The Constitution PDF is a diglot (Hindi + English) document.
- all-MiniLM-L6-v2 is English-only, so Hindi (Devanagari script) lines
  must be filtered out before chunking/embedding.
- This module also normalizes whitespace and drops empty/junk lines.

Multilingual support is explicitly Future Scope (SDD Section 12) —
this is a deliberate MVP simplification, not a permanent limitation.
"""

import re

# Devanagari Unicode block: U+0900 to U+097F
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def is_devanagari_line(line: str) -> bool:
    """Return True if a line contains Devanagari script characters."""
    return bool(DEVANAGARI_PATTERN.search(line))


def clean_page_text(text: str) -> str:
    """
    Given raw extracted text for one page, return English-only,
    whitespace-normalized text.
    """
    lines = text.split("\n")
    english_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue  # drop empty lines

        if is_devanagari_line(stripped):
            continue  # drop Hindi lines

        english_lines.append(stripped)

    return "\n".join(english_lines)


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Apply clean_page_text to a list of page dicts from document_loader.py.

    Args:
        pages: [{"page_number": 1, "text": "..."}, ...]

    Returns:
        Same structure, with cleaned English-only text.
    """
    cleaned = []
    for page in pages:
        cleaned_text = clean_page_text(page["text"])
        cleaned.append({
            "page_number": page["page_number"],
            "text": cleaned_text,
        })
    return cleaned


if __name__ == "__main__":
    # Manual test: load real pages via document_loader, then clean them,
    # and visually confirm Hindi is gone and English reads correctly.
    from document_loader import load_pdf

    PDF_PATH = "../../datasets/raw/constitution_of_india.pdf"
    pages = load_pdf(PDF_PATH)
    cleaned = clean_pages(pages)

    print(f"Total pages: {len(cleaned)}\n")

    # Preview a few pages further in, past the title page/preface,
    # where actual Articles should start.
    for p in cleaned[5:8]:
        print(f"--- Page {p['page_number']} (cleaned) ---")
        print(p["text"][:500])
        print()