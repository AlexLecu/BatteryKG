"""Experiment 13 — OCR stage for the image-only Panasonic marketing sheet
(deterministic, NO LLM inside).

`data/claims/datasheets/panasonic_ncr18650b.pdf` has no text layer at all: four
pages, each a single full-page raster, 0 characters and 0 word boxes under
pdfplumber. The frozen pipeline's own snapshot of it
(`data/claims/extracted_text/panasonic_ncr18650b.txt`) is 66 bytes — four page
markers and nothing else — which is why the document is excluded from the
extraction evaluation (paper, Limitations 3).

This module renders each page and OCRs it, writing a frozen snapshot in the SAME
format as `src/agents/pdf_text.py` (`=== PAGE n ===` markers, reusing that
module's `PAGE_MARK`), so the frozen extraction pipeline can consume it without
modification. The snapshot is written HERE, inside the experiment folder — the
empty snapshot under data/ is left untouched, because it is the artifact that
documents the limitation.

Pipeline, and why each step is there:

  1. render at the raster's NATIVE resolution. The embedded image is 1684x1190
     for an 842x595 pt page = 144 dpi. Rendering at 300 or 400 dpi upscales a
     144 dpi source and measurably HURTS: it adds interpolation blur, not
     detail (300 dpi recovered 5/15 target values, native 9/15).
  2. remove table rules. Long horizontal/vertical runs of ink are the ruled
     borders of the specification table; tesseract's layout analysis merges
     short value cells into them and drops the values. Erasing runs longer than
     RULE_MIN_RUN px is the single decisive step in this pipeline
     (confident tokens 47-54 -> 73-85).
  3. upscale x3 (LANCZOS) so glyph height lands in tesseract's comfortable
     range, then OCR with --psm 6 (uniform block), --oem 1 (LSTM).

CONFIGURATION CHOICE. The configuration was selected by a GOLD-INDEPENDENT
criterion — the number of tokens tesseract returns with confidence >= 60,
maximised over an 18-point sweep (rule removal off/60/100 x scale 1/2/3 x
psm 4/6). It was NOT selected by counting how many gold values came out. The two
rankings are reported side by side in the README; the decisive factor, rule
removal, is chosen the same way by either.

Run:  python -m experiments.exp13_ocr_panasonic.ocr
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output

from src.agents.pdf_text import PAGE_MARK
from src.config import DATA, ROOT

DOC = "panasonic_ncr18650b"
PDF_PATH = DATA / "claims" / "datasheets" / f"{DOC}.pdf"
OUT_DIR = ROOT / "experiments" / "exp13_ocr_panasonic"
OCR_TEXT = OUT_DIR / f"{DOC}_ocr.txt"          # the frozen OCR snapshot
OCR_META = OUT_DIR / "ocr_meta.json"

NATIVE_DPI = 144          # the embedded raster's own resolution
RULE_MIN_RUN = 60         # px at native dpi; longer ink runs are table rules
UPSCALE = 3
TESS_CONFIG = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
CONF_THRESHOLD = 60       # for the gold-independent quality statistic


# --- preprocessing -----------------------------------------------------------
def to_ink(img: Image.Image) -> np.ndarray:
    """Boolean ink mask (True = dark)."""
    return np.array(img.convert("L")) < 128


def strip_rules(ink: np.ndarray, min_run: int = RULE_MIN_RUN) -> np.ndarray:
    """Erase horizontal and vertical ink runs longer than `min_run` — the ruled
    borders of the specification table. Glyph strokes are far shorter, so text
    survives; the rules that swallow the value cells do not."""
    out = ink.copy()
    for axis in (1, 0):                       # 1 = horizontal runs, 0 = vertical
        src = ink if axis == 1 else ink.T
        dst = out if axis == 1 else out.T
        for i in range(src.shape[0]):
            row = src[i]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if e - s >= min_run:
                    dst[i, s:e] = False
    return out


def preprocess(page: Image.Image) -> Image.Image:
    ink = strip_rules(to_ink(page))
    img = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    return img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.LANCZOS)


# --- OCR ---------------------------------------------------------------------
@dataclass
class PageOCR:
    page: int
    text: str
    n_tokens: int
    n_conf_tokens: int          # tokens with confidence >= CONF_THRESHOLD
    mean_conf: float


def ocr_page(page: Image.Image, n: int) -> PageOCR:
    img = preprocess(page)
    text = pytesseract.image_to_string(img, config=TESS_CONFIG)
    data = pytesseract.image_to_data(img, config=TESS_CONFIG, output_type=Output.DICT)
    confs = [int(c) for c, t in zip(data["conf"], data["text"])
             if int(c) >= 0 and t.strip()]
    return PageOCR(
        page=n, text=text.rstrip(), n_tokens=len(confs),
        n_conf_tokens=sum(1 for c in confs if c >= CONF_THRESHOLD),
        mean_conf=round(sum(confs) / len(confs), 2) if confs else 0.0)


def _tesseract_version() -> str:
    return str(pytesseract.get_tesseract_version())


def _poppler_version() -> str:
    try:
        out = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True)
        return (out.stderr or out.stdout).splitlines()[0].strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def build(pdf_path: Path = PDF_PATH) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} missing — the datasheet is not redistributed; "
            "re-download it per data/README.md to rebuild the OCR snapshot.")
    pages = convert_from_path(str(pdf_path), dpi=NATIVE_DPI)
    results = [ocr_page(p, i) for i, p in enumerate(pages, start=1)]

    snapshot = "\n\n".join(f"{PAGE_MARK.format(n=r.page)}\n{r.text}" for r in results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_TEXT.write_text(snapshot)

    meta = {
        "document": DOC,
        "pdf": str(pdf_path.relative_to(ROOT)),
        "source_raster": f"{pages[0].width}x{pages[0].height} at {NATIVE_DPI} dpi",
        "tesseract": _tesseract_version(),
        "poppler": _poppler_version(),
        "config": {"native_dpi": NATIVE_DPI, "rule_min_run": RULE_MIN_RUN,
                   "upscale": UPSCALE, "tesseract_config": TESS_CONFIG,
                   "selection_criterion":
                       f"max tokens with confidence >= {CONF_THRESHOLD} "
                       "(gold-independent)"},
        "snapshot": str(OCR_TEXT.relative_to(ROOT)),
        "snapshot_chars": len(snapshot),
        "pages": [{"page": r.page, "chars": len(r.text), "tokens": r.n_tokens,
                   "confident_tokens": r.n_conf_tokens, "mean_conf": r.mean_conf}
                  for r in results],
    }
    OCR_META.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def main() -> dict:
    meta = build()
    print(f"[exp13] {meta['document']}: raster {meta['source_raster']}, "
          f"tesseract {meta['tesseract']}")
    for p in meta["pages"]:
        print(f"[exp13]   page {p['page']}: {p['chars']:5d} chars, "
              f"{p['tokens']:3d} tokens ({p['confident_tokens']:3d} conf>=60), "
              f"mean conf {p['mean_conf']:.1f}")
    print(f"[exp13] snapshot {meta['snapshot_chars']:,} chars -> {OCR_TEXT}")
    print(f"[exp13] wrote {OCR_META}")
    return meta


if __name__ == "__main__":
    main()
