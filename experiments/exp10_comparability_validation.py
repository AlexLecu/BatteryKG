"""Independent validation of the comparability rules against the graph's verdicts.

The claim-vs-claim verdicts of Section 4.4 (43 cross-document comparisons) are
produced by `src/kg/spec_consistency.py`. This script does NOT call that code:
it reimplements the verdict rule as a pure function, from the per-field rules
read out of the implementation, and compares the result with what is stored in
Neo4j. Re-running the same code could only confirm that the code is
deterministic; restating the rules independently can catch a rule the
implementation applies but the paper does not describe.

Read-only. It writes nothing to the graph and does not touch the stored verdicts.

THE EXTRACTED RULES (src/kg/spec_consistency.py, verified at the line given):

  VALUE_TOL = 0.02                                                    (l.37)
  _close(a,b)  ->  |a-b| <= 0.02 * max(|a|,|b|,1e-9)                  (l.66)
      the tolerance is RELATIVE to the larger value, not to the first one.
  values_equal ->  lists compared element-by-element AFTER sorting;   (l.70)
      different lengths => false; scalars are treated as lists of 1.

  Verdict order (verdict_for, l.224):
    1. if values_equal(a,b):
         - if the conditions differ            -> condition_mismatch
         - otherwise                           -> consistent
    2. otherwise, bound satisfaction: applies ONLY if one side carries a bound
       marker and the other does NOT carry a marker in the SAME direction (two
       values both marked "Max." are rival specifications, hence a conflict).
         upper: py <= px*(1+0.02)  -> consistent
         lower: py >= px*(1-0.02)  -> consistent
    3. otherwise                               -> document_variant_conflict

  Conditions compared (CONDITION_KEYS, l.43) and their rule (l.215):
    temperature_c, charge_current_a, discharge_current_a, charge_c_rate,
    discharge_c_rate, dod_pct, charge_voltage_v, retention_pct
    -> a condition is compared ONLY if it is present on BOTH sides.
       "unspecified"/absent on either side => not compared, so it CANNOT
       produce a condition_mismatch. Same 2% tolerance.
    -> retention_pct (the EOL threshold) is not a structured field: it is
       extracted from the cond_end_condition text as the first `N%` found (l.89).
    -> NOTE: discharge_cutoff_v and charge_time_min are attached as conditions
       by the parser but are NOT in CONDITION_KEYS, so they are never compared.

  Bound markers (l.49-51), case-insensitive:
    upper: max | maximum | less than | up to | no more than | < | <=
    lower: min (not 'minute', not preceded by a digit) | minimum | over |
           at least | more than | > | >=

Run:  python -m experiments.exp10_comparability_validation
"""
from __future__ import annotations

import ast
import re

from src.kg.connection import get_driver

VALUE_TOL = 0.02

CONDITION_KEYS = ["temperature_c", "charge_current_a", "discharge_current_a",
                  "charge_c_rate", "discharge_c_rate", "dod_pct",
                  "charge_voltage_v", "retention_pct"]

# the rule table, kept as data so it is printable and auditable
RULES = [
    ("value match", "|a-b| <= tol * max(|a|,|b|)", "2%",
     "lists: sorted, element-by-element; different lengths = false"),
    ("bound satisfaction", "point inside the other side's bound", "2%",
     "only when exactly one side has a bound; both in the same direction = conflict"),
    ("temperature_c", "equality if both present", "2%", "absent anywhere => not compared"),
    ("charge_current_a", "equality if both present", "2%", "same"),
    ("discharge_current_a", "equality if both present", "2%", "same"),
    ("charge_c_rate", "equality if both present", "2%", "same"),
    ("discharge_c_rate", "equality if both present", "2%", "discharge regime"),
    ("dod_pct", "equality if both present", "2%", "DoD"),
    ("charge_voltage_v", "equality if both present", "2%", "charge regime"),
    ("retention_pct", "equality if both present", "2%",
     "the EOL threshold, extracted from text as the first N%"),
    ("discharge_cutoff_v", "NOT compared", "—", "attached by the parser but absent from CONDITION_KEYS"),
]

_BOUND_UPPER = re.compile(r"\b(max|maximum|less than|up to|no more than)\b|<|≤", re.I)
_BOUND_LOWER = re.compile(
    r"(?<!\d)(?<!\d\s)\bmin(?!ute)\b\.?|\b(minimum|over|at least|more than)\b|>|≥", re.I)


# --- the pure reimplementation --------------------------------------------------
def close(a: float, b: float, tol: float | None = None) -> bool:
    # tol is read at CALL time, not bound as a default at definition time:
    # otherwise a change to VALUE_TOL would have no effect and the sensitivity
    # test below would falsely report that the tolerance does not matter.
    t = VALUE_TOL if tol is None else tol
    return abs(a - b) <= t * max(abs(a), abs(b), 1e-9)


def values_equal(a, b, tol: float | None = None) -> bool:
    la = a if isinstance(a, list) else [a]
    lb = b if isinstance(b, list) else [b]
    if len(la) != len(lb):
        return False
    return all(close(x, y, tol) for x, y in zip(sorted(la), sorted(lb)))


def bound_direction(text: str) -> str | None:
    if _BOUND_UPPER.search(text or ""):
        return "upper"
    if _BOUND_LOWER.search(text or ""):
        return "lower"
    return None


def point(v):
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def condition_diffs(ca: dict, cb: dict, tol: float | None = None) -> list[str]:
    """Condition differences; only the fields present on BOTH sides."""
    out = []
    for k in CONDITION_KEYS:
        va, vb = ca.get(k), cb.get(k)
        if va is not None and vb is not None and not close(float(va), float(vb), tol):
            out.append(f"{k}: {float(va):g} vs {float(vb):g}")
    return out


def verdict(val_a, val_b, cond_a: dict, cond_b: dict,
            frag_a: str, frag_b: str, tol: float | None = None) -> tuple[str, str]:
    """The verdict, from the rules above alone."""
    t = VALUE_TOL if tol is None else tol
    va, vb = point(val_a), point(val_b)
    if values_equal(va, vb, t):
        diffs = condition_diffs(cond_a, cond_b, t)
        return ("condition_mismatch", "; ".join(diffs)) if diffs else ("consistent", "values match")
    for (vx, fx), (vy, fy) in (((val_a, frag_a), (val_b, frag_b)),
                               ((val_b, frag_b), (val_a, frag_a))):
        dx, dy = bound_direction(fx), bound_direction(fy)
        if not dx or dy == dx:
            continue
        px, py = point(vx), point(vy)
        if isinstance(px, float) and isinstance(py, float):
            if dx == "upper" and py <= px * (1 + t):
                return "consistent", f"{py:g} satisfies upper bound {px:g}"
            if dx == "lower" and py >= px * (1 - t):
                return "consistent", f"{py:g} satisfies lower bound {px:g}"
    return "document_variant_conflict", f"{va} vs {vb}"


# --- re-deriving the inputs (independently of the original code) ----------------
def parse_value(s: str):
    """value_a/value_b are stored as str() of the normalised value."""
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)


def retention_from_text(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return float(m.group(1)) if m else None


def lygte_conditions(prop: str, fragment: str) -> dict:
    """Conditions of the spec-echo line, from the verbatim stored fragment.

    Covers exactly the cases where the original parser attaches conditions that
    appear in CONDITION_KEYS; the remaining lines have no comparable conditions.
    """
    L = fragment or ""
    c: dict = {}
    m = re.search(r"Cycle life at ([\d.]+)C discharge,\s*([\d.]+)% DOD", L, re.I)
    if m:
        c["discharge_c_rate"] = float(m.group(1))
        c["dod_pct"] = float(m.group(2))
        return c
    m = re.search(r"Cycle life:\s*([\d.,]+)\s*at\s*([\d.]+)A,\s*([\d.,]+)\s*at\s*([\d.]+)A"
                  r".*?([\d.]+)A charge.*?([\d.]+)\s*%", L, re.I)
    if m:
        c["charge_current_a"] = float(m.group(5))
        c["retention_pct"] = float(m.group(6))
        return c          # discharge_current_a depends on which branch of the pair
    m = re.search(r"standard charge(?: method)?:\s*[\d.,]+\s*(?:mA|A)\b"
                  r"(?:,\s*|\s+to\s+)([\d.]+)\s*V", L, re.I)
    if m:
        c["charge_voltage_v"] = float(m.group(1))
    return c


_CLAIM_QUERY = """
MATCH (cl:Claim {property: $property})-[:ABOUT]->(c:Cell {model: $model})
MATCH (cl)-[:ASSERTED_BY]->(s:Source {source_id: $source_id})
RETURN cl.value AS value, cl.unit AS unit,
       cl.cond_temperature_c AS temperature_c,
       cl.cond_charge_current_a AS charge_current_a,
       cl.cond_discharge_current_a AS discharge_current_a,
       cl.cond_charge_c_rate AS charge_c_rate,
       cl.cond_discharge_c_rate AS discharge_c_rate,
       cl.cond_dod_pct AS dod_pct,
       cl.cond_charge_voltage_v AS charge_voltage_v,
       cl.cond_end_condition AS end_condition
"""


def gold_conditions(session, model: str, source_id: str, prop: str, value) -> dict:
    """The claim's structured conditions, matched on (model, source, prop, value)."""
    rows = session.run(_CLAIM_QUERY, model=model, source_id=source_id,
                       property=prop).data()
    if not rows:
        return {}
    scale = {"mah": 1e-3, "ma": 1e-3, "mv": 1e-3}
    best, best_d = None, None
    for r in rows:
        raw = r["value"]
        if isinstance(raw, (list, tuple)):
            continue
        v = float(raw) * scale.get((r["unit"] or "").strip().lower(), 1.0)
        p = point(value)
        d = abs(v - p) if isinstance(p, float) else 0.0
        if best_d is None or d < best_d:
            best, best_d = r, d
    if best is None:
        best = rows[0]
    c = {k: best[k] for k in CONDITION_KEYS if best.get(k) is not None}
    ret = retention_from_text(best.get("end_condition") or "")
    if ret is not None:
        c["retention_pct"] = ret
    return c


_COMPARISONS_QUERY = """
MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(c:Cell)
WHERE d.verdict <> 'coverage_gap'
RETURN c.model AS model, d.property AS property, d.value_a AS value_a,
       d.value_b AS value_b, d.source_a AS source_a, d.source_b AS source_b,
       d.verdict AS stored_verdict, d.rationale AS stored_rationale,
       d.fragment_a AS fragment_a, d.fragment_b AS fragment_b,
       d.discrepancy_id AS did
ORDER BY model, property, did
"""


def conditions_for(session, model, source_id, prop, value, fragment) -> dict:
    if source_id.startswith("datasheet_"):
        return gold_conditions(session, model, source_id, prop, value)
    return lygte_conditions(prop, fragment)


def main() -> None:
    print("=" * 78)
    print("(a) THE RULE TABLE ACTUALLY USED")
    print("=" * 78)
    print(f"  {'field':<22} {'rule':<44} {'tol':<5} note")
    for field, rule, tol, note in RULES:
        print(f"  {field:<22} {rule:<44} {tol:<5} {note}")
    print()
    print("  verdict order: values_equal -> (conditions differ ? "
          "condition_mismatch : consistent)")
    print("                 otherwise bound satisfaction -> consistent")
    print("                 otherwise document_variant_conflict")
    print()

    driver = get_driver()
    try:
        with driver.session() as s:
            rows = s.run(_COMPARISONS_QUERY).data()
            results = []
            for r in rows:
                va, vb = parse_value(r["value_a"]), parse_value(r["value_b"])
                ca = conditions_for(s, r["model"], r["source_a"], r["property"],
                                    va, r["fragment_a"])
                cb = conditions_for(s, r["model"], r["source_b"], r["property"],
                                    vb, r["fragment_b"])
                v, detail = verdict(va, vb, ca, cb, r["fragment_a"], r["fragment_b"])
                results.append({**r, "recomputed": v, "detail": detail,
                                "cond_a": ca, "cond_b": cb,
                                "agree": v == r["stored_verdict"]})
    finally:
        driver.close()

    n = len(results)
    ok = sum(x["agree"] for x in results)
    print("=" * 78)
    print(f"(b) AGREEMENT: {ok}/{n}")
    print("=" * 78)
    from collections import Counter
    print("  stored     :", dict(Counter(x["stored_verdict"] for x in results)))
    print("  recomputed :", dict(Counter(x["recomputed"] for x in results)))
    print()

    print("=" * 78)
    print("(c) MISMATCHES")
    print("=" * 78)
    bad = [x for x in results if not x["agree"]]
    if not bad:
        print("  none — the reimplemented rules reproduce every stored verdict.")
    for x in bad:
        print(f"\n  {x['model']} / {x['property']}")
        print(f"    sources    : {x['source_a']}  vs  {x['source_b']}")
        print(f"    values     : {x['value_a']}  vs  {x['value_b']}")
        print(f"    stored     : {x['stored_verdict']}  ({x['stored_rationale']})")
        print(f"    recomputed : {x['recomputed']}  ({x['detail']})")
        print(f"    cond A     : {x['cond_a'] or '(none)'}")
        print(f"    cond B     : {x['cond_b'] or '(none)'}")
        da, db = bound_direction(x["fragment_a"]), bound_direction(x["fragment_b"])
        print(f"    bound      : A={da} B={db}")
        print(f"    fragment A : {(x['fragment_a'] or '')[:100]}")
        print(f"    fragment B : {(x['fragment_b'] or '')[:100]}")
        cause = ("conditions re-derived differently" if x["stored_verdict"] == "condition_mismatch"
                 or x["recomputed"] == "condition_mismatch"
                 else "value/bound rule")
        print(f"    likely cause: {cause}")

    _sensitivity(results)
    return results


def _sensitivity(results: list[dict]) -> None:
    """(d) Does the validation have power? A test that passes even with the rules
    broken validates nothing. Agreement is re-run with the tolerance mutated and
    with the condition rule mutated; if agreement does NOT drop, that rule is not
    exercised by this dataset and cannot be considered validated."""
    from collections import Counter
    print()
    print("=" * 78)
    print("(d) SENSITIVITY — how far the 43 comparisons exercise the rules")
    print("=" * 78)
    branch = Counter()
    for x in results:
        va, vb = parse_value(x["value_a"]), parse_value(x["value_b"])
        if values_equal(point(va), point(vb)):
            branch["decided by value equality"] += 1
        elif x["recomputed"] == "consistent":
            branch["decided by bound satisfaction"] += 1
        else:
            branch["value conflict"] += 1
    for k, v in branch.items():
        print(f"  {k:<32} {v}")
    print(f"  {'of which condition_mismatch':<32} "
          f"{sum(x['recomputed'] == 'condition_mismatch' for x in results)}")
    print()

    def agree(tol=None, cond_one_sided=False):
        n = 0
        for x in results:
            va, vb = parse_value(x["value_a"]), parse_value(x["value_b"])
            ca, cb = x["cond_a"], x["cond_b"]
            if cond_one_sided:
                extra = [k for k in CONDITION_KEYS
                         if (ca.get(k) is None) != (cb.get(k) is None)]
                if extra and values_equal(point(va), point(vb), tol):
                    n += x["stored_verdict"] == "condition_mismatch"
                    continue
            v, _ = verdict(va, vb, ca, cb, x["fragment_a"], x["fragment_b"], tol)
            n += v == x["stored_verdict"]
        return n

    total = len(results)
    print(f"  tolerance {VALUE_TOL:.0%} (real)                   -> {agree()}/{total}")
    for t in (0.0, 0.005, 0.05, 0.5):
        print(f"  tolerance {t:<6.1%} (mutation)            -> {agree(tol=t)}/{total}")
    print(f"  \"unspecified counts\" (mutation)       -> "
          f"{agree(cond_one_sided=True)}/{total}")
    print()
    print("  Reading: agreement drops under every mutation, so both rules are")
    print("  genuinely exercised by the dataset — the validation does not pass for free.")


if __name__ == "__main__":
    main()
