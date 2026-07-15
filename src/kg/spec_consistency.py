"""Claim-vs-claim reconciliation: specification consistency across document
variants of the SAME cell.

Compared, per cell and per property:
  (a) gold datasheet claims  vs  lygte-info.dk 'Official specifications' echoes
  (b) Panasonic marketing sheet  vs  the SANYO/Panasonic full specification

Verdicts per property pair:
  consistent                 values match after unit normalization (2% tol),
                             or a point value satisfies the other side's
                             stated bound (max/min/over/less-than)
  condition_mismatch         values match but the stated test conditions differ
                             (flagged distinctly from value mismatches)
  document_variant_conflict  numeric mismatch
  coverage_gap               property recorded from one document, absent in the
                             other (for gold-derived sides this can reflect
                             extraction scope; the Panasonic marketing sheet's
                             missing cycle life is a VERIFIED omission,
                             annotated from the gold YAML's `omissions` field)

Creates Discrepancy nodes (kind='claim_vs_claim') idempotently and writes
outputs/spec_consistency.md + paper/tables/tab_spec_consistency.tex.

Run:  python -m src.kg.spec_consistency
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import yaml

from src.config import DATA, OUTPUTS, ROOT
from src.kg.connection import get_driver

VALUE_TOL = 0.02
LYGTE_JSON = DATA / "processed" / "lygte_measurements.json"
CLAIMS_DIR = DATA / "claims"
PAPER_TABLE = ROOT / "paper" / "tables" / "tab_spec_consistency.tex"

# condition keys compared when present on BOTH sides of a pair
CONDITION_KEYS = ["temperature_c", "charge_current_a", "discharge_current_a",
                  "charge_c_rate", "discharge_c_rate", "dod_pct",
                  "charge_voltage_v", "retention_pct"]

# NOTE: bare 'min' must not match minute durations ("45 min", "15min") or
# "minutes" — require it NOT to be preceded by a number and NOT be 'minute(s)'.
_BOUND_UPPER = re.compile(r"\b(max|maximum|less than|up to|no more than)\b|<|≤", re.I)
_BOUND_LOWER = re.compile(
    r"(?<!\d)(?<!\d\s)\bmin(?!ute)\b\.?|\b(minimum|over|at least|more than)\b|>|≥", re.I)


# --- unit normalization -------------------------------------------------------
def normalize_value(value, unit: str | None, prop: str):
    """Normalize a value to the property's canonical unit (Ah, A, V, h, ...)."""
    u = (unit or "").strip().lower()
    scale = {"mah": 1e-3, "ma": 1e-3, "mv": 1e-3}.get(u, 1.0)
    if u == "min" and prop.endswith("_h"):
        scale = 1.0 / 60.0
    if isinstance(value, (list, tuple)):
        return [float(v) * scale for v in value]
    return float(value) * scale


def _close(a: float, b: float, tol: float = VALUE_TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def values_equal(a, b) -> bool:
    la = a if isinstance(a, list) else [a]
    lb = b if isinstance(b, list) else [b]
    if len(la) != len(lb):
        return False
    return all(_close(x, y) for x, y in zip(sorted(la), sorted(lb)))


# --- one side of a comparison -----------------------------------------------------
@dataclass
class SpecEntry:
    property: str
    value: object                       # normalized scalar or [lo, hi]
    source_id: str
    fragment: str                       # verbatim source text
    conditions: dict = field(default_factory=dict)
    bound_text: str = ""                # text carrying max/min markers


def _retention_from_text(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return float(m.group(1)) if m else None


# --- lygte spec-echo line parser -----------------------------------------------------
def _entry(prop, value, unit, frag, sid, **conds):
    return SpecEntry(property=prop, value=normalize_value(value, unit, prop),
                     source_id=sid, fragment=frag,
                     conditions={k: v for k, v in conds.items() if v is not None},
                     bound_text=frag)


def parse_spec_echo_line(line: str, source_id: str) -> list[SpecEntry]:
    """Deterministic mapping of one 'Official specifications' line to entries."""
    out: list[SpecEntry] = []
    L = line.strip()

    def num(s):
        return float(s.replace(",", ""))

    # capacity (+ optional combined voltage): "Nominal Capacity & Voltage: 1.1Ah, 3.3v"
    m = re.search(r"(?:Nominal|Real) capacity(?: & voltage)?:\s*(?:NCR )?([\d.,]+)\s*(mAh|Ah)(?:,\s*([\d.]+)\s*v)?", L, re.I)
    if m:
        out.append(_entry("nominal_capacity_ah", num(m.group(1)), m.group(2), L, source_id))
        if m.group(3):
            out.append(_entry("nominal_voltage_v", num(m.group(3)), "V", L, source_id))
        return out
    m = re.search(r"Nominal voltage:\s*([\d.]+)\s*V", L, re.I)
    if m:
        return [_entry("nominal_voltage_v", num(m.group(1)), "V", L, source_id)]
    # "Standard charge: 1500mA, 4.2V, 50mA" / "Recommended standard charge method: 1.5A to 3.6v CCCV, 45min"
    m = re.search(r"standard charge(?: method)?:\s*([\d.,]+)\s*(mA|A)\b(?:,\s*|\s+to\s+)([\d.]+)\s*V", L, re.I)
    if m:
        return [_entry("std_charge_current_a", num(m.group(1)), m.group(2), L, source_id,
                       charge_voltage_v=num(m.group(3)))]
    m = re.search(r"fast charge(?: current)?:\s*([\d.,]+)\s*(mA|A)\b", L, re.I)
    if m:
        return [_entry("fast_charge_current_a", num(m.group(1)), m.group(2), L, source_id)]
    m = re.search(r"Max\.? charge voltage:\s*([\d.]+)\s*V", L, re.I)
    if m:
        return [_entry("max_charge_voltage_v", num(m.group(1)), "V", L, source_id)]
    m = re.search(r"Max\.? charge current:\s*([\d.,]+)\s*(mA|A)\b", L, re.I)
    if m:
        return [_entry("max_charge_current_a", num(m.group(1)), m.group(2), L, source_id)]
    m = re.search(r"Standard discharge:\s*([\d.,]+)\s*(mA|A)\b.*?([\d.]+)\s*V", L, re.I)
    if m:
        return [_entry("std_discharge_current_a", num(m.group(1)), m.group(2), L, source_id,
                       discharge_cutoff_v=num(m.group(3)))]
    m = re.search(r"Fast discharge:\s*([\d.,]+)\s*(mA|A),\s*([\d.,]+)\s*(mA|A)", L, re.I)
    if m:
        v = [normalize_value(num(m.group(1)), m.group(2), "fast_discharge_current_a"),
             normalize_value(num(m.group(3)), m.group(4), "fast_discharge_current_a")]
        e = SpecEntry("fast_discharge_current_a", v, source_id, L, {}, L)
        return [e]
    m = re.search(r"Max\.?(?:imum)? continuous discharge:\s*([\d.,]+)\s*(mA|A)\b", L, re.I)
    if m:
        return [_entry("max_cont_discharge_a", num(m.group(1)), m.group(2), L, source_id)]
    m = re.search(r"Pulse discharge.*?:\s*([\d.,]+)\s*A", L, re.I)
    if m:
        return [_entry("max_pulse_discharge_a", num(m.group(1)), "A", L, source_id)]
    # "Cycle life at 10C discharge, 100% DOD: Over 1,000 cycles"
    m = re.search(r"Cycle life at ([\d.]+)C discharge,\s*([\d.]+)% DOD:\s*(?:Over\s*)?([\d.,]+)\s*cycles", L, re.I)
    if m:
        return [_entry("cycle_life_cycles", num(m.group(3)), "cycles", L, source_id,
                       discharge_c_rate=num(m.group(1)), dod_pct=num(m.group(2)))]
    # "Cycle life: 300 at 10A, 200 at 20A both with 4A charge, ramaning capacity minimum 70%"
    m = re.search(r"Cycle life:\s*([\d.,]+)\s*at\s*([\d.]+)A,\s*([\d.,]+)\s*at\s*([\d.]+)A.*?([\d.]+)A charge.*?([\d.]+)\s*%", L, re.I)
    if m:
        ret = num(m.group(6))
        return [
            _entry("cycle_life_cycles", num(m.group(1)), "cycles", L, source_id,
                   discharge_current_a=num(m.group(2)), charge_current_a=num(m.group(5)),
                   retention_pct=ret),
            _entry("cycle_life_cycles", num(m.group(3)), "cycles", L, source_id,
                   discharge_current_a=num(m.group(4)), charge_current_a=num(m.group(5)),
                   retention_pct=ret),
        ]
    m = re.search(r"(?:Cell )?(?:Net)?weight:?\s*(?:Max\.?\s*)?([\d.]+)\s*g", L, re.I)
    if m:
        return [_entry("mass_g", num(m.group(1)), "g", L, source_id)]
    # "Recommended charge/cutoff voltage: 3.5v to 2v"
    m = re.search(r"charge/cut-?off voltage:\s*([\d.]+)\s*v\s*to\s*([\d.]+)\s*v", L, re.I)
    if m:
        return [_entry("charge_voltage_v", num(m.group(1)), "V", L, source_id),
                _entry("discharge_cutoff_v", num(m.group(2)), "V", L, source_id)]
    # "Cut off at 2.75V, full charge 4.35V (Wrong value)"
    m = re.search(r"Cut off at ([\d.]+)\s*V(?:,\s*full charge\s*([\d.]+)\s*V)?", L, re.I)
    if m:
        out.append(_entry("discharge_cutoff_v", num(m.group(1)), "V", L, source_id))
        if m.group(2):
            out.append(_entry("charge_voltage_v", num(m.group(2)), "V", L, source_id))
        return out
    # temperature ranges
    m = re.search(r"Operating temp(?:erature)?:?\s*(?:charge\s*)?(-?\d+)\s*°?C?\s*~\s*\+?(-?\d+)(?:°C|degC)?(?:.*?discharge:?\s*(-?\d+)\s*°?C?\s*~\s*\+?(-?\d+))?", L, re.I)
    if m:
        if m.group(3) is not None:      # separate charge/discharge ranges
            return [_entry("charge_temp_range_c", [num(m.group(1)), num(m.group(2))], "degC", L, source_id),
                    _entry("discharge_temp_range_c", [num(m.group(3)), num(m.group(4))], "degC", L, source_id)]
        return [_entry("operating_temp_range_c", [num(m.group(1)), num(m.group(2))], "degC", L, source_id)]
    m = re.search(r"Storage temperature:.*?1 year:\s*(-?\d+)\s*°?C?\s*~\s*\+?(-?\d+)", L, re.I)
    if m:
        return [_entry("storage_temp_range_c", [num(m.group(1)), num(m.group(2))], "degC", L, source_id)]
    return []


def parse_spec_echo(lines: list[str], source_id: str) -> list[SpecEntry]:
    out: list[SpecEntry] = []
    for line in lines:
        out.extend(parse_spec_echo_line(line, source_id))
    return out


# --- verdict logic --------------------------------------------------------------------
def _bound_direction(text: str) -> str | None:
    if _BOUND_UPPER.search(text or ""):
        return "upper"
    if _BOUND_LOWER.search(text or ""):
        return "lower"
    return None


def _point(v):
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def _conditions_mismatch(a: SpecEntry, b: SpecEntry) -> list[str]:
    diffs = []
    for key in CONDITION_KEYS:
        va, vb = a.conditions.get(key), b.conditions.get(key)
        if va is not None and vb is not None and not _close(float(va), float(vb)):
            diffs.append(f"{key}: {va:g} vs {vb:g}")
    return diffs


def verdict_for(a: SpecEntry, b: SpecEntry) -> tuple[str, str]:
    """Verdict + detail for two entries of the same property."""
    va, vb = _point(a.value), _point(b.value)
    if values_equal(va, vb):
        diffs = _conditions_mismatch(a, b)
        if diffs:
            return "condition_mismatch", "; ".join(diffs)
        return "consistent", "values match"
    # bound satisfaction: a point value inside the OTHER side's stated bound.
    # Applies only when exactly one side is a bound — two differing values that
    # are BOTH stated as e.g. 'Max.' are rival spec values, i.e. a conflict.
    for x, y in ((a, b), (b, a)):
        dx = _bound_direction(x.bound_text)
        dy = _bound_direction(y.bound_text)
        if not dx or dy == dx:
            continue
        px, py = _point(x.value), _point(y.value)
        if isinstance(px, float) and isinstance(py, float):
            if dx == "upper" and py <= px * (1 + VALUE_TOL):
                return "consistent", f"{py:g} satisfies stated upper bound {px:g}"
            if dx == "lower" and py >= px * (1 - VALUE_TOL):
                return "consistent", f"{py:g} satisfies stated lower bound {px:g}"
    return "document_variant_conflict", f"{va} vs {vb}"


def _pair_entries(ea: list[SpecEntry], eb: list[SpecEntry]):
    """Within one property: globally greedy pairing by closest value (smallest
    distances first, one use per entry); leftovers on either side = gaps."""
    def _scalar(v):
        p = _point(v)
        try:
            return float(p if not isinstance(p, list) else p[0])
        except (TypeError, ValueError):
            return 0.0

    cands = sorted(((abs(_scalar(a.value) - _scalar(b.value)), i, j)
                    for i, a in enumerate(ea) for j, b in enumerate(eb)),
                   key=lambda t: (t[0], t[1], t[2]))
    used_a, used_b, pairs = set(), set(), []
    for _d, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((ea[i], eb[j]))
    gaps = [("a_only", a) for i, a in enumerate(ea) if i not in used_a]
    gaps += [("b_only", b) for j, b in enumerate(eb) if j not in used_b]
    return pairs, gaps


# --- gold claims from KG -----------------------------------------------------------------
_GOLD_QUERY = """
MATCH (cl:Claim)-[:ABOUT]->(c:Cell)-[:HAS_CHEMISTRY]->()
MATCH (cl)-[:ASSERTED_BY]->(s:Source)
RETURN c.model AS model, s.source_id AS source_id, cl.property AS property,
       cl.value AS value, cl.unit AS unit, cl.conditions_text AS text,
       cl.notes AS notes, cl.claim_id AS claim_id,
       cl.cond_temperature_c AS temperature_c,
       cl.cond_charge_current_a AS charge_current_a,
       cl.cond_discharge_current_a AS discharge_current_a,
       cl.cond_charge_c_rate AS charge_c_rate,
       cl.cond_discharge_c_rate AS discharge_c_rate,
       cl.cond_dod_pct AS dod_pct,
       cl.cond_charge_voltage_v AS charge_voltage_v,
       cl.cond_end_condition AS end_condition
"""


def _gold_entries(driver) -> dict[tuple[str, str], list[SpecEntry]]:
    """{(model, source_id): [SpecEntry...]} from the gold Claim nodes."""
    out: dict[tuple[str, str], list[SpecEntry]] = {}
    with driver.session() as s:
        for r in s.run(_GOLD_QUERY).data():
            conds = {k: r[k] for k in ["temperature_c", "charge_current_a",
                                       "discharge_current_a", "charge_c_rate",
                                       "discharge_c_rate", "dod_pct",
                                       "charge_voltage_v"] if r[k] is not None}
            ret = _retention_from_text(r["end_condition"] or "")
            if ret is not None:
                conds["retention_pct"] = ret
            frag = (r["text"] or "") + (f" [{r['notes']}]" if r["notes"] else "")
            entry = SpecEntry(
                property=r["property"],
                value=normalize_value(r["value"], r["unit"], r["property"]),
                source_id=r["source_id"], fragment=frag, conditions=conds,
                bound_text=frag)
            out.setdefault((r["model"], r["source_id"]), []).append(entry)
    return out


def _verified_omissions() -> dict[tuple[str, str], list[str]]:
    """{(model, source_id): [property...]} from the gold YAMLs' omissions."""
    out = {}
    for path in sorted(CLAIMS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if doc.get("omissions"):
            from pathlib import Path as _P
            sid = f"datasheet_{_P(doc['source_document']).stem.replace('-', '_')}"
            out[(doc["cell_model"], sid)] = [o["property"] for o in doc["omissions"]]
    return out


# --- main computation --------------------------------------------------------------------
_DISCREPANCY_CYPHER = """
MATCH (c:Cell {model: $model})
MERGE (d:Discrepancy {discrepancy_id: $did})
  SET d.kind = 'claim_vs_claim', d.property = $property,
      d.source_a = $source_a, d.source_b = $source_b,
      d.value_a = $value_a, d.value_b = $value_b,
      d.delta = $delta, d.verdict = $verdict, d.rationale = $rationale,
      d.provenance = $provenance,
      d.fragment_a = $fragment_a, d.fragment_b = $fragment_b
MERGE (d)-[:ABOUT]->(c)
"""

_CLEAR_CYPHER = "MATCH (d:Discrepancy {kind: 'claim_vs_claim'}) DETACH DELETE d"


def compute_spec_consistency(driver=None, write_outputs: bool = True) -> list[dict]:
    own = driver is None
    driver = driver or get_driver()
    try:
        gold = _gold_entries(driver)
        omissions = _verified_omissions()
        lygte = json.loads(LYGTE_JSON.read_text()) if LYGTE_JSON.exists() else {"reviews": []}
        echo_by_model = {r["cell_model"]: (r["source_id"],
                                           parse_spec_echo(r["spec_echo"], r["source_id"]))
                         for r in lygte["reviews"]}

        # document pairs: every gold doc vs lygte echo, + Panasonic mkt vs full spec
        pairs_to_compare: list[tuple[str, str, list[SpecEntry], str, list[SpecEntry]]] = []
        for (model, sid), entries in sorted(gold.items()):
            if model in echo_by_model:
                lsid, lentries = echo_by_model[model]
                pairs_to_compare.append((model, sid, entries, lsid, lentries))
        pana = [(sid, e) for (m, sid), e in gold.items() if m == "Panasonic NCR18650B"]
        if len(pana) == 2:
            pana.sort()   # datasheet_panasonic_ncr18650b < ..._full_spec_sanyo? sort for determinism
            (sid_a, ea), (sid_b, eb) = pana
            pairs_to_compare.append(("Panasonic NCR18650B", sid_a, ea, sid_b, eb))

        def _provenance(sid_a: str, sid_b: str) -> str:
            """manufacturer_internal = both documents issued by the manufacturer
            (e.g. Panasonic marketing sheet vs full spec); ecosystem = at least
            one third-party document (lygte echoes / retail wrapper)."""
            return ("manufacturer_internal"
                    if sid_a.startswith("datasheet_") and sid_b.startswith("datasheet_")
                    else "ecosystem")

        rows: list[dict] = []
        with driver.session() as s:
            s.run(_CLEAR_CYPHER)              # clean-rebuild the whole class
            for model, sid_a, ea, sid_b, eb in pairs_to_compare:
                props = sorted({e.property for e in ea} | {e.property for e in eb})
                for prop in props:
                    pa = [e for e in ea if e.property == prop]
                    pb = [e for e in eb if e.property == prop]
                    matched, gaps = _pair_entries(pa, pb)
                    for a, b in matched:
                        verdict, detail = verdict_for(a, b)
                        va, vb = _point(a.value), _point(b.value)
                        delta = None
                        if isinstance(va, float) and isinstance(vb, float) and va:
                            delta = (vb - va) / abs(va)
                        rows.append(dict(model=model, property=prop,
                                         source_a=sid_a, source_b=sid_b,
                                         value_a=str(a.value), value_b=str(b.value),
                                         delta=delta, verdict=verdict, detail=detail,
                                         provenance=_provenance(sid_a, sid_b),
                                         fragment_a=a.fragment, fragment_b=b.fragment))
                    for side, e in gaps:
                        missing_side = sid_b if side == "a_only" else sid_a
                        verified = prop in omissions.get((model, missing_side), [])
                        detail = (f"absent in {missing_side}"
                                  + (" — VERIFIED omission (document exhaustively "
                                     "checked)" if verified
                                     else " — may reflect gold-extraction scope"))
                        rows.append(dict(model=model, property=prop,
                                         source_a=sid_a, source_b=sid_b,
                                         value_a=str(e.value) if side == "a_only" else "—",
                                         value_b=str(e.value) if side == "b_only" else "—",
                                         delta=None, verdict="coverage_gap", detail=detail,
                                         provenance=_provenance(sid_a, sid_b),
                                         fragment_a=e.fragment if side == "a_only" else "",
                                         fragment_b=e.fragment if side == "b_only" else ""))
            for i, r in enumerate(rows):
                did = (f"{r['model']}:specvar:{r['property']}:"
                       f"{r['source_a']}:{r['source_b']}:{i}")
                s.run(_DISCREPANCY_CYPHER, model=r["model"], did=did,
                      property=r["property"], source_a=r["source_a"],
                      source_b=r["source_b"], value_a=r["value_a"],
                      value_b=r["value_b"], delta=r["delta"],
                      verdict=r["verdict"], provenance=r["provenance"],
                      rationale=r["detail"], fragment_a=r["fragment_a"],
                      fragment_b=r["fragment_b"])
        if write_outputs:
            _write_reports(rows)
        _print_summary(rows)
        return rows
    finally:
        if own:
            driver.close()


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        compared = [r for r in rs if r["verdict"] != "coverage_gap"]
        problems = [r for r in rs if r["verdict"] in
                    ("document_variant_conflict", "condition_mismatch")]
        out.append(dict(
            model=model,
            n_compared=len(compared),
            consistent=sum(r["verdict"] == "consistent" for r in rs),
            condition_mismatch=sum(r["verdict"] == "condition_mismatch" for r in rs),
            conflicts=sum(r["verdict"] == "document_variant_conflict" for r in rs),
            conflicts_mfr=sum(r["provenance"] == "manufacturer_internal"
                              for r in problems),
            conflicts_eco=sum(r["provenance"] == "ecosystem" for r in problems),
            gaps=sum(r["verdict"] == "coverage_gap" for r in rs)))
    return out


def _print_summary(rows: list[dict]) -> None:
    for s in _summarize(rows):
        print(f"[spec_consistency] {s['model']}: {s['n_compared']} compared -> "
              f"{s['consistent']} consistent, {s['condition_mismatch']} condition "
              f"mismatches, {s['conflicts']} conflicts "
              f"({s['conflicts_mfr']} mfr-internal / {s['conflicts_eco']} ecosystem "
              f"incl. cond-mismatches), {s['gaps']} coverage gaps")


def _write_reports(rows: list[dict]) -> None:
    lines = ["# Specification consistency across document variants", ""]
    a = lines.append
    a("Per cell: properties recorded in >= 2 documents, compared after unit "
      "normalization (2% tolerance; bound satisfaction counted as consistent). "
      "`condition_mismatch` = same value, different stated test conditions.\n")
    a("| cell | compared | consistent | condition mismatch | conflicts | mfr-internal | ecosystem | coverage gaps |")
    a("|---|---|---|---|---|---|---|---|")
    for s in _summarize(rows):
        a(f"| {s['model']} | {s['n_compared']} | {s['consistent']} "
          f"| {s['condition_mismatch']} | {s['conflicts']} "
          f"| {s['conflicts_mfr']} | {s['conflicts_eco']} | {s['gaps']} |")
    a("\n(mfr-internal / ecosystem columns count conflicts + condition mismatches "
      "by provenance: manufacturer-internal = both documents issued by the "
      "manufacturer; ecosystem = at least one third-party document.)")
    a("\n## Itemized conflicts and condition mismatches\n")
    a("| cell | property | value A | value B | sources | verdict | provenance | detail |")
    a("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["verdict"] in ("document_variant_conflict", "condition_mismatch"):
            a(f"| {r['model']} | {r['property']} | {r['value_a']} | {r['value_b']} "
              f"| {r['source_a']} vs {r['source_b']} | {r['verdict']} "
              f"| {r['provenance']} | {r['detail']} |")
    a("\n## Coverage gaps\n")
    a("| cell | property | present in | detail |")
    a("|---|---|---|---|")
    for r in rows:
        if r["verdict"] == "coverage_gap":
            present = r["source_a"] if r["value_a"] != "—" else r["source_b"]
            a(f"| {r['model']} | {r['property']} | {present} | {r['detail']} |")
    (OUTPUTS / "spec_consistency.md").write_text("\n".join(lines) + "\n")
    print(f"[spec_consistency] -> {OUTPUTS / 'spec_consistency.md'}")

    # booktabs LaTeX summary table
    tex = [
        "% Auto-generated by src/kg/spec_consistency.py — do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Specification consistency across document variants of the same",
        "cell: gold datasheet claims vs.\\ the lygte-info.dk specification echoes,",
        "and (Panasonic) the marketing sheet vs.\\ the full product specification.",
        "Values compared after unit normalization (2\\% tolerance; a point value",
        "satisfying the other document's stated bound counts as consistent).",
        "Condition mismatches --- same value, different stated test conditions ---",
        "are flagged separately from value conflicts. The Mfr./Eco.\\ columns",
        "split conflicts + condition mismatches by provenance:",
        "manufacturer-internal (both documents issued by the manufacturer) vs.\\",
        "ecosystem (at least one third-party document).\\label{tab:specconsistency}}",
        "\\begin{tabularx}{\\textwidth}{lCCCCCCC}",
        "\\toprule",
        "\\textbf{Cell} & \\textbf{Compared} & \\textbf{Consistent} & "
        "\\textbf{Cond.\\ mism.} & \\textbf{Conflict} & \\textbf{Mfr.} & "
        "\\textbf{Eco.} & \\textbf{Gap} \\\\",
        "\\midrule",
    ]
    for s in _summarize(rows):
        tex.append(f"{s['model']} & {s['n_compared']} & {s['consistent']} & "
                   f"{s['condition_mismatch']} & {s['conflicts']} & "
                   f"{s['conflicts_mfr']} & {s['conflicts_eco']} & {s['gaps']} \\\\")
    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text("\n".join(tex) + "\n")
    print(f"[spec_consistency] -> {PAPER_TABLE}")


if __name__ == "__main__":
    compute_spec_consistency()
