"""Layout-preserving text extraction from datasheet PDFs.

Datasheets are mostly tables, so text is extracted with pdfplumber's
layout mode (spatial whitespace preserved) and page markers are kept:

    === PAGE 1 ===
    ...text...

The extracted text is stored under data/claims/extracted_text/<stem>.txt so
every LLM-extraction run is reproducible against a frozen text snapshot.

Run:  python -m src.agents.pdf_text
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pdfplumber

from src.config import DATA

DATASHEET_DIR = DATA / "claims" / "datasheets"
TEXT_DIR = DATA / "claims" / "extracted_text"

# stem -> gold YAML it corresponds to (documentation; evaluation re-derives)
DOCUMENTS = [
    "a123_apr18650m1a",
    "panasonic_ncr18650b",
    "panasonic_ncr18650b_full_spec_sanyo",
    "lg_inr18650hg2",
    # held-out evaluation document (experiment 11); its text snapshot is frozen
    # under data/claims/extracted_text/ — the PDF itself is not redistributable,
    # so build_all() reports it MISSING and leaves the snapshot untouched.
    "samsung_inr18650_25r",
]

PAGE_MARK = "=== PAGE {n} ==="


def extract_pdf_text(pdf_path: Path) -> tuple[str, dict]:
    """Extract layout text with page markers. Returns (text, stats)."""
    pages_out, empty_pages = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text(layout=True) or ""
                if not txt.strip():
                    empty_pages.append(i)
                pages_out.append(f"{PAGE_MARK.format(n=i)}\n{txt.rstrip()}")
    text = "\n\n".join(pages_out)
    stats = {
        "n_pages": len(pages_out),
        "empty_pages": empty_pages,
        "non_ws_chars": len("".join(text.split())),
        "has_text": len(empty_pages) < len(pages_out),
    }
    return text, stats


def build_all() -> dict[str, dict]:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for stem in DOCUMENTS:
        pdf = DATASHEET_DIR / f"{stem}.pdf"
        if not pdf.exists():
            print(f"[pdf_text] MISSING {pdf.name} — re-download per data/README.md")
            continue
        text, stats = extract_pdf_text(pdf)
        out = TEXT_DIR / f"{stem}.txt"
        out.write_text(text)
        results[stem] = stats
        note = "" if stats["has_text"] else "  <-- NO TEXT LAYER (image-only; needs OCR/vision)"
        print(f"[pdf_text] {stem}: {stats['n_pages']} pages, "
              f"{stats['non_ws_chars']:,} chars, empty pages {stats['empty_pages']}{note}")
    return results


if __name__ == "__main__":
    build_all()
