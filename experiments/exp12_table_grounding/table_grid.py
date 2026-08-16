"""Experiment 12 — deterministic table grounding (NO LLM inside).

Turns a datasheet PDF into isolated, row-level extraction contexts:

  1. pdfplumber ruling-line table detection, per page, with provenance
     (page, table index, bbox).
  2. the running page header (a one-row table carrying "Spec. No.") is dropped.
  3. PAGE-BOUNDARY STITCH — a grid whose axis row ends one page and whose value
     row begins the next is rejoined (see `stitch_regions`). On the Samsung
     INR18650-25R this fires exactly once, for section 7.6, which is the
     mechanical reason flat layout-mode text never recovered that grid.
  4. regions are classified as `grid` (>= 2 quantity rows: an axis row and a
     value row) or `spec_table` (2 columns, key/value specification rows).
  5. each region becomes one or more EXTRACTION UNITS — the isolated contexts
     handed to the model one at a time: one unit per grid, one unit per
     specification row.

Also holds `compose_property_name`, the DETERMINISTIC NAMER for grid cells: the
model emits (family, axis_kind, axis_value) and this module composes the
snake_case property name. The naming templates were authored with knowledge of
the gold's naming scheme — the disclosed convention leakage of this pilot — but
the gold's values and grid structure were never shown to the model. The namer
applies to grid cells ONLY; specification rows are named by the model from the
frozen `extractor.PROPERTY_VOCAB`, with no naming rule of any kind.

Nothing here reaches an LLM and nothing here depends on the gold standard.

Run:  python -m experiments.exp12_table_grounding.table_grid
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

from src.config import DATA, ROOT

DOC = "samsung_inr18650_25r"
PDF_PATH = DATA / "claims" / "datasheets" / f"{DOC}.pdf"
OUT_DIR = ROOT / "experiments" / "exp12_table_grounding"
TABLES_JSON = OUT_DIR / "tables_extracted.json"

RUNNING_HEADER_MARKER = "Spec. No."   # the repeated page header table
MIN_QTY_CELLS = 2                     # cells parsing as quantities -> "quantity row"
MIN_QTY_ROWS_GRID = 2                 # a grid needs an axis row AND a value row
SPEC_TABLE_COLS = 2                   # key/value specification table
SPEC_TABLE_MIN_ROWS = 4


# --- cell-level quantity parsing ---------------------------------------------
# A cell is a QUANTITY when it is a bare number with an optional unit suffix
# ("-20℃", "0.50A", "100%", "2,500mAh"). Anything richer — "CCCV, 1.25A, 4.20 ±
# 0.05 V" — is free text, not a quantity.
_UNITS = r"%|℃|°C|mAh|Ah|mA|A|mV|V|mΩ|mOhm|cycles?|g|mm|min|h"
_QTY_RE = re.compile(
    rf"^[-−~]?\s*\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?\s*(?:{_UNITS})?$", re.IGNORECASE)
_NUM_RE = re.compile(r"[-−]?\d{1,3}(?:,\d{3})*(?:\.\d+)?")


def clean_cell(cell: str | None) -> str:
    """Tidy whitespace but PRESERVE the cell's own line breaks.

    pdfplumber returns a multi-line cell newline-joined, and those breaks carry
    structure the model needs: section 3.12 is three storage rows in one cell,
    and flattening it to a single line is what makes them look like one claim.
    """
    lines = [" ".join(ln.split()) for ln in (cell or "").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def flat(cell: str | None) -> str:
    """One-line form, for matching and for grid rendering."""
    return " ".join((cell or "").split())


def is_quantity(cell: str | None) -> bool:
    c = flat(cell)
    return bool(c) and bool(_QTY_RE.match(c))


def is_free_text(cell: str | None) -> bool:
    c = flat(cell)
    return bool(c) and not is_quantity(c)


def has_digit(cell: str | None) -> bool:
    return bool(_NUM_RE.search(flat(cell)))


def quantity_row_indices(rows: list[list[str]]) -> list[int]:
    """Rows holding >= MIN_QTY_CELLS quantity cells (axis rows and value rows)."""
    return [i for i, r in enumerate(rows)
            if sum(is_quantity(c) for c in r) >= MIN_QTY_CELLS]


# --- table regions -----------------------------------------------------------
@dataclass
class TableRegion:
    region_id: str
    pages: list[int]
    rows: list[list[str]]
    n_cols: int
    bboxes: list[list[float]] = field(default_factory=list)
    kind: str = "other"                 # grid | spec_table | other
    stitched: bool = False
    stitch_note: str = ""

    @property
    def qty_rows(self) -> list[int]:
        return quantity_row_indices(self.rows)


def _is_running_header(rows: list[list[str]]) -> bool:
    return any(RUNNING_HEADER_MARKER in clean_cell(c) for r in rows for c in r)


def extract_content_tables(pdf_path: Path) -> list[TableRegion]:
    """Every ruling-line table in the document except the running page header."""
    regions: list[TableRegion] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(pdf_path) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                for ti, table in enumerate(page.find_tables()):
                    rows = [[clean_cell(c) for c in row] for row in table.extract()]
                    if not rows or _is_running_header(rows):
                        continue
                    regions.append(TableRegion(
                        region_id=f"p{pno}_t{ti}",
                        pages=[pno],
                        rows=rows,
                        n_cols=max(len(r) for r in rows),
                        bboxes=[[round(v, 2) for v in table.bbox]]))
    return regions


# --- page-boundary stitch ----------------------------------------------------
def stitch_regions(regions: list[TableRegion]) -> tuple[list[TableRegion], list[dict]]:
    """Rejoin a grid split across a page break.

    A grid needs two quantity rows: an axis row and a value row. When the LAST
    content table on page n carries exactly one quantity row (it is truncated —
    an axis with nothing under it) and the FIRST content table on page n+1
    carries exactly one quantity row and no free text at all (a bare value row
    with its caption left behind on the previous page), and both have the same
    column count, the two are one grid that the page break tore apart.

    This is a minimal heuristic, validated on one document; it is deliberately
    conservative (it cannot fire when either half is already a complete grid).
    Every firing is logged so the decision stays auditable.
    """
    out: list[TableRegion] = []
    log: list[dict] = []
    i = 0
    while i < len(regions):
        a = regions[i]
        b = regions[i + 1] if i + 1 < len(regions) else None
        if (b is not None
                and b.pages[0] == a.pages[-1] + 1
                and a.n_cols == b.n_cols
                and len(a.qty_rows) == 1
                and len(b.qty_rows) == 1
                and not any(is_free_text(c) for r in b.rows for c in r)
                and _is_last_on_page(regions, i) and _is_first_on_page(regions, i + 1)):
            merged = TableRegion(
                region_id=f"{a.region_id}+{b.region_id}",
                pages=[a.pages[-1], b.pages[0]],
                rows=a.rows + b.rows,
                n_cols=a.n_cols,
                bboxes=a.bboxes + b.bboxes,
                stitched=True,
                stitch_note=(f"axis row ends page {a.pages[-1]}, value row begins "
                             f"page {b.pages[0]}; {a.n_cols} columns on both sides"))
            out.append(merged)
            log.append({"merged": merged.region_id, "pages": merged.pages,
                        "n_cols": merged.n_cols, "note": merged.stitch_note})
            i += 2
            continue
        out.append(a)
        i += 1
    return out, log


def _is_last_on_page(regions: list[TableRegion], i: int) -> bool:
    page = regions[i].pages[-1]
    return all(r.pages[0] != page for r in regions[i + 1:])


def _is_first_on_page(regions: list[TableRegion], i: int) -> bool:
    page = regions[i].pages[0]
    return all(r.pages[-1] != page for r in regions[:i])


# --- classification ----------------------------------------------------------
def classify(region: TableRegion) -> str:
    if len(region.qty_rows) >= MIN_QTY_ROWS_GRID:
        return "grid"
    if region.n_cols == SPEC_TABLE_COLS and len(region.rows) >= SPEC_TABLE_MIN_ROWS:
        return "spec_table"
    return "other"


# --- extraction units (the isolated contexts) --------------------------------
@dataclass
class ExtractionUnit:
    unit_id: str
    kind: str                  # "grid" | "spec_row"
    region_id: str
    pages: list[int]
    rows: list[list[str]]
    render: str                # what the model is shown
    stitched: bool = False


def render_rows(rows: list[list[str]]) -> str:
    """Pipe-render a grid; grid cells are short, so they are flattened."""
    width = max(len(r) for r in rows)
    return "\n".join("| " + " | ".join(
        flat((r + [""] * (width - len(r)))[c]) for c in range(width)) + " |"
        for r in rows)


def render_spec_row(row: list[str]) -> str:
    """Key/value rendering that keeps the specification cell's line structure —
    one source line per line, so multi-row cells stay visibly multi-row."""
    item = flat(row[0]) if row else ""
    spec_lines = clean_cell(row[-1] if len(row) > 1 else "").splitlines()
    body = "\n".join(f"  {ln}" for ln in spec_lines)
    return f"Item: {item}\nSpecification:\n{body}"


def units_from_regions(regions: list[TableRegion]) -> list[ExtractionUnit]:
    """One unit per grid; one unit per specification row."""
    units: list[ExtractionUnit] = []
    for reg in regions:
        if reg.kind == "grid":
            units.append(ExtractionUnit(
                unit_id=f"{reg.region_id}_grid", kind="grid", region_id=reg.region_id,
                pages=reg.pages, rows=reg.rows, render=render_rows(reg.rows),
                stitched=reg.stitched))
        elif reg.kind == "spec_table":
            for ri, row in enumerate(reg.rows):
                # header row ("Item | Specification") carries no numbers
                if not has_digit(row[-1] if row else ""):
                    continue
                units.append(ExtractionUnit(
                    unit_id=f"{reg.region_id}_r{ri}", kind="spec_row",
                    region_id=reg.region_id, pages=reg.pages, rows=[row],
                    render=render_spec_row(row)))
    return units


# --- deterministic namer for GRID CELLS ONLY ---------------------------------
# The model emits (family, axis_kind, axis_value); the property name is composed
# here. Convention leakage is disclosed in the README: these templates were
# authored knowing the gold's naming scheme. The gold's values and grid
# structure were not disclosed to the model, and specification rows never touch
# this function.
FAMILIES = ("rel_discharge_capacity", "rel_charge_capacity", "rel_capacity")
AXIS_KINDS = ("temperature_c", "current_a", "charge_mode")
CHARGE_MODES = ("std", "rapid")


def _fmt_num(x: float) -> str:
    """5 -> '5', 0.50 -> '0_5', 20.0 -> '20' (trailing zeros dropped)."""
    return f"{float(x):.10g}".replace(".", "_")


def axis_token(axis_kind: str, axis_value) -> str:
    if axis_kind == "temperature_c":
        v = float(axis_value)
        return f"{'m' if v < 0 else ''}{_fmt_num(abs(v))}c"
    if axis_kind == "current_a":
        return f"{_fmt_num(float(axis_value))}a"
    if axis_kind == "charge_mode":
        mode = str(axis_value).strip().lower()
        if mode not in CHARGE_MODES:
            raise ValueError(f"charge_mode must be one of {CHARGE_MODES}: {axis_value!r}")
        return mode
    raise ValueError(f"axis_kind must be one of {AXIS_KINDS}: {axis_kind!r}")


def compose_property_name(family: str, axis_kind: str, axis_value) -> str:
    """Deterministic snake_case property name for one grid cell."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}: {family!r}")
    tok = axis_token(axis_kind, axis_value)
    if axis_kind == "charge_mode":
        return f"{family}_{tok}_charge_pct"
    return f"{family}_{tok}_pct"


# --- build -------------------------------------------------------------------
def build(pdf_path: Path = PDF_PATH) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} missing — the datasheet is not redistributed; "
            "re-download it per data/README.md to rebuild the table snapshot.")
    raw = extract_content_tables(pdf_path)
    regions, stitch_log = stitch_regions(raw)
    for reg in regions:
        reg.kind = classify(reg)
    units = units_from_regions(regions)
    return {
        "document": DOC,
        "pdf": str(pdf_path.relative_to(ROOT)),
        "n_content_tables_raw": len(raw),
        "n_regions_after_stitch": len(regions),
        "stitches": stitch_log,
        "regions": [{**asdict(r), "qty_rows": r.qty_rows} for r in regions],
        "units": [asdict(u) for u in units],
        "counts": {
            "grid": sum(u.kind == "grid" for u in units),
            "spec_row": sum(u.kind == "spec_row" for u in units),
            "total_units": len(units),
        },
    }


def main() -> dict:
    payload = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    c = payload["counts"]
    print(f"[exp12] {payload['n_content_tables_raw']} content tables -> "
          f"{payload['n_regions_after_stitch']} regions "
          f"({len(payload['stitches'])} page-boundary stitch)")
    for reg in payload["regions"]:
        print(f"[exp12]   {reg['region_id']:14s} pages={reg['pages']} "
              f"{len(reg['rows'])}x{reg['n_cols']:<2d} kind={reg['kind']}"
              f"{'  [STITCHED]' if reg['stitched'] else ''}")
    print(f"[exp12] extraction units: {c['grid']} grid + {c['spec_row']} spec_row "
          f"= {c['total_units']} (x3 runs = {c['total_units'] * 3} LLM calls)")
    print(f"[exp12] wrote {TABLES_JSON}")
    return payload


if __name__ == "__main__":
    main()
