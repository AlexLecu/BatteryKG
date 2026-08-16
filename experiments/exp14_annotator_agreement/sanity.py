"""Step 1 — schema and convention check on the second annotator's five files.

Runs BEFORE scoring: its purpose is to decide whether anything needs a
clarification round with the annotator. It reads the returned files and the
source PDFs (page counts) only; it writes one report and nothing else.

Run:  python -m experiments.exp14_annotator_agreement.sanity [second_set_dir]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from experiments.exp14_annotator_agreement.common import (
    ALLOWED_UNITS, DEFAULT_SECOND_SET, SHIPPED_VOCAB, document_set,
    pdf_page_count, rel_to_root, unit_class,
)

HERE = Path(__file__).resolve().parent

# "min" alone is the time unit, so the bound test needs the abbreviating period
# or the full word; otherwise "45 min" reads as a minimum.
BOUND_WORDS = re.compile(r"\bmax\.|\bmin\.|\bmaximum\b|\bminimum\b|\bover\s+\d|"
                         r"\bat least\b|\bup to\b|\bno more than\b|[≥≤]|>=|<=", re.I)
NON_NORMALISED = {"mah", "ma", "mv"}


def check_document(d: dict) -> dict:
    errors, warnings, infos = [], [], []
    raw, claims = d["sec_raw"], d["sec_claims"]
    where = d["sec_path"].name

    # --- header ------------------------------------------------------------
    for key in ("cell_model", "manufacturer", "source_document", "claims"):
        if key not in raw:
            errors.append(f"header: missing top-level key `{key}`")
    for key in ("annotator", "annotated_date"):
        if not raw.get(key) or str(raw.get(key)).strip().upper() == "TODO":
            warnings.append(f"header: `{key}` not filled in")
    if "omissions" not in raw:
        notes = str(raw.get("annotator_notes") or "").lower()
        if "omission" in notes or "absence check" in notes:
            infos.append("no `omissions:` block, but annotator_notes explains the "
                         "absence check — per guideline §10 this is the documented "
                         "'not exhaustively checked' state")
        else:
            warnings.append("no `omissions:` block and no explanation in "
                            "annotator_notes (guideline §10 asks for one or the other)")

    declared_new = {n.get("name") for n in (raw.get("new_properties") or [])
                    if isinstance(n, dict)}

    # --- per claim ---------------------------------------------------------
    n_pages = pdf_page_count(d["pdf"])
    seen = {}
    used_props = set()
    for c in claims:
        tag = f"claim #{c.idx} ({c.property})"
        used_props.add(c.property)

        if not c.property:
            errors.append(f"{tag}: no `property`")
        if c.value is None:
            errors.append(f"{tag}: no `value`")
        if c.unit is None:
            errors.append(f"{tag}: no `unit`")
        if c.page is None:
            errors.append(f"{tag}: no `page`")

        # value must be numeric, or a 2-element numeric range
        v = c.value
        if isinstance(v, str):
            errors.append(f"{tag}: `value` is a string ({v!r}) — number expected; "
                          f"any ±/≥/≤ belongs in stated_conditions.text")
        elif isinstance(v, list):
            if len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
                errors.append(f"{tag}: range `value` is not [lo, hi] numeric: {v!r}")
            elif v[0] > v[1]:
                warnings.append(f"{tag}: range `value` not in [lo, hi] order: {v!r}")
        elif not isinstance(v, (int, float, type(None))):
            errors.append(f"{tag}: `value` has unexpected type {type(v).__name__}")

        # unit
        u = str(c.unit or "").strip().lower()
        if u and u not in ALLOWED_UNITS:
            warnings.append(f"{tag}: unit {c.unit!r} not in the documented unit list")
        if u in NON_NORMALISED:
            warnings.append(f"{tag}: unit {c.unit!r} not normalised "
                            f"(guideline §5: mAh->Ah, mA->A, mV->V)")

        # page
        if isinstance(c.page, int):
            if c.page < 1:
                errors.append(f"{tag}: page {c.page} < 1")
            elif n_pages and c.page > n_pages:
                errors.append(f"{tag}: page {c.page} beyond the PDF ({n_pages} pages)")
        elif c.page is not None:
            errors.append(f"{tag}: `page` is not an integer ({c.page!r})")

        # conditions
        if not c.cond_text:
            errors.append(f"{tag}: `stated_conditions.text` missing")
        elif c.is_unspecified and c.cond_parsed:
            errors.append(f"{tag}: text is `unspecified` but parsed conditions are "
                          f"set ({sorted(c.cond_parsed)}) — contradictory")
        if c.cond_text and not c.is_unspecified and \
                str(c.cond_text).strip().lower().startswith("unspecified"):
            warnings.append(f"{tag}: condition text starts with 'unspecified' but is "
                            f"not exactly that string — the loader tests equality")

        # notes conventions
        if BOUND_WORDS.search(c.cond_text or "") and not c.notes:
            infos.append(f"{tag}: conditions state a bound but `notes` does not "
                         f"record the direction (guideline §7.7)")

        # vocabulary
        if c.property and c.property not in SHIPPED_VOCAB and \
                c.property not in declared_new:
            warnings.append(f"{tag}: property outside the shipped 27-property "
                            f"vocabulary and not declared in `new_properties`")

        # A repeated (property, value, page) is normal for characteristic grids —
        # sibling cells are told apart by their conditions (§8.2). Only an
        # identical condition string makes it a real duplicate.
        key = (c.property, str(c.value), c.page, " ".join(str(c.cond_text).split()))
        if key in seen:
            warnings.append(f"{tag}: duplicate of claim #{seen[key]} "
                            f"(same property, value, page AND conditions)")
        else:
            seen[key] = c.idx

    for name in declared_new - used_props:
        infos.append(f"`new_properties` declares `{name}` but no claim uses it")

    return dict(document=d["label"], file=where, n_claims=len(claims),
                pdf_pages=n_pages, errors=errors, warnings=warnings, infos=infos)


def main(second_dir: Path = DEFAULT_SECOND_SET) -> dict:
    reports = [check_document(d) for d in document_set(second_dir)]
    total_e = sum(len(r["errors"]) for r in reports)
    total_w = sum(len(r["warnings"]) for r in reports)

    lines = ["# Sanity pass — second annotator's five files", "",
             f"Source: `{rel_to_root(second_dir)}`", "",
             "| document | file | claims | PDF pages | errors | warnings | notes |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in reports:
        lines.append(f"| {r['document']} | `{r['file']}` | {r['n_claims']} | "
                     f"{r['pdf_pages']} | {len(r['errors'])} | "
                     f"{len(r['warnings'])} | {len(r['infos'])} |")
    lines += ["", f"**Totals: {total_e} errors, {total_w} warnings across "
                  f"{len(reports)} files.**", ""]
    lines.append("All five files parse as YAML and load through the same reader the "
                 "reference set uses.\n" if total_e == 0 else
                 "**Blocking issues present — resolve before scoring.**\n")

    for r in reports:
        lines.append(f"## {r['document']}")
        lines.append(f"`{r['file']}` — {r['n_claims']} claims")
        for kind, items in (("ERROR", r["errors"]), ("WARNING", r["warnings"]),
                            ("note", r["infos"])):
            if items:
                lines.append("")
                lines.append(f"**{kind}s ({len(items)}):**")
                lines += [f"- {t}" for t in items]
        if not (r["errors"] or r["warnings"] or r["infos"]):
            lines.append("")
            lines.append("Clean — no schema or convention deviations.")
        lines.append("")

    (HERE / "sanity_report.md").write_text("\n".join(lines))
    (HERE / "sanity.json").write_text(json.dumps(reports, indent=2))
    print("\n".join(lines[:12]))
    print(f"\n[exp14.sanity] wrote sanity_report.md / sanity.json "
          f"({total_e} errors, {total_w} warnings)")
    return dict(reports=reports, errors=total_e, warnings=total_w)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECOND_SET)
