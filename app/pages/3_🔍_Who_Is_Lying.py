"""Who is lying? — spec sheets that disagree with each other, as visual cards.

Reuses the Cell Explorer spec-consistency Cypher verbatim (snapshot-compatible),
concatenated across cells client-side; every card is a stored graph fact.
Headline definition matches the paper (Table 8): a "comparison" is a property
present in BOTH documents (consistent / condition_mismatch /
document_variant_conflict); coverage_gap rows are not comparisons.
"""
import ast
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import make_driver, run_query  # noqa: E402

st.set_page_config(page_title="Who is lying? — BatteryKG", page_icon="🔍",
                   layout="wide")
st.title("🔍 Who is lying?")

COMPARISON_VERDICTS = {"consistent", "condition_mismatch",
                       "document_variant_conflict"}

# plain-language labels for every property in the released gold standard
PROP_LABEL = {
    "capacity_retention_at_temp_pct": "capacity retained at temperature",
    "charge_temp_range_c": "charging temperature range",
    "charge_voltage_v": "charge voltage",
    "cycle_life_cycles": "cycle life",
    "cycle_life_retention_pct": "end-of-life capacity threshold",
    "discharge_cutoff_v": "discharge cut-off voltage",
    "discharge_temp_range_c": "discharge temperature range",
    "fast_charge_current_a": "fast-charge current",
    "fast_discharge_current_a": "fast discharge current",
    "gravimetric_energy_density_wh_kg": "energy density by weight",
    "internal_impedance_mohm": "internal impedance",
    "mass_g": "weight",
    "max_charge_current_a": "max charge current",
    "max_charge_voltage_v": "max charge voltage",
    "max_cont_discharge_a": "max continuous discharge current",
    "max_pulse_discharge_a": "max pulse discharge current",
    "minimum_capacity_ah": "minimum capacity",
    "nominal_capacity_ah": "nominal capacity",
    "nominal_voltage_v": "nominal voltage",
    "operating_temp_range_c": "operating temperature range",
    "rated_capacity_ah": "rated capacity",
    "std_charge_current_a": "standard charge current",
    "std_charge_time_h": "standard charge time",
    "std_discharge_current_a": "standard discharge current",
    "storage_capacity_recovery_pct": "capacity recovered after storage",
    "storage_capacity_remaining_pct": "capacity remaining after storage",
    "storage_temp_range_c": "storage temperature range",
    "volumetric_energy_density_wh_l": "energy density by volume",
}

# unit from the property-key suffix (order matters: longest/most specific first)
UNIT_SUFFIX = [("_cycles", "cycles"), ("_pct", "%"), ("_mohm", "mΩ"),
               ("_wh_kg", "Wh/kg"), ("_wh_l", "Wh/L"), ("_ah", "Ah"),
               ("_c", "°C"), ("_v", "V"), ("_a", "A"), ("_h", "h"),
               ("_g", "g")]


def prop_unit(prop: str) -> str:
    for suffix, unit in UNIT_SUFFIX:
        if prop.endswith(suffix):
            return unit
    return ""


def prop_label(prop: str) -> str:
    if prop in PROP_LABEL:
        return PROP_LABEL[prop]
    label = prop                                 # fallback: prettified key
    for suffix, _ in UNIT_SUFFIX:
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label.replace("_", " ")


def _num(x: float) -> str:
    return f"{float(x):g}"                       # 1000.0 -> 1000, 47.5 -> 47.5


def fmt_value(prop: str, raw) -> str:
    """'[10.0, 45.0]' -> '10–45 °C'; '1000.0' -> '1000 cycles'."""
    unit = prop_unit(prop)
    try:
        v = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return f"{raw} {unit}".strip()
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = v
        sep = " to " if float(lo) < 0 else "–"   # '−5 to 50', '10–45'
        core = f"{_num(lo)}{sep}{_num(hi)}"
    else:
        core = _num(v)
    return f"{core} {unit}".strip() if unit else core


# differing-condition rendering for same-value mismatches (from d.rationale)
COND_LABEL = {
    "discharge_c_rate": ("discharge speed during the test", "{}C"),
    "retention_pct": ("capacity level that counts as end of life", "{}%"),
}


def parse_conditions(detail: str):
    """'discharge_c_rate: 5 vs 10' -> ('discharge speed…', '5C', '10C')."""
    m = re.match(r"^\s*(\w+):\s*(.+?)\s+vs\s+(.+?)\s*$", detail or "")
    if not m:
        return None
    key, a, b = m.groups()
    label, fmt = COND_LABEL.get(key, (key.replace("_", " "), "{}"))
    return label, fmt.format(a), fmt.format(b)


driver, err = make_driver()
if driver is None:
    st.error(f"Knowledge graph unavailable — {err}")
    st.stop()

cells = run_query(driver, "MATCH (c:Cell) RETURN c.model AS model, "
                          "c.manufacturer AS mfr ORDER BY model")
if not cells:
    st.warning("No Cell nodes in the graph — run `python -m src.kg.claims`.")
    st.stop()

frames = []
for c in cells:
    rows = run_query(driver, """
    MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(:Cell {model: $model})
    RETURN d.property AS property, d.value_a AS value_a, d.value_b AS value_b,
           d.source_a AS source_a, d.source_b AS source_b,
           d.verdict AS verdict, d.rationale AS detail,
           d.provenance AS provenance,
           d.fragment_a AS fragment_a, d.fragment_b AS fragment_b
    ORDER BY d.verdict, d.property
""", model=c["model"])
    for r in rows:
        r["cell"] = c["model"]
    frames.append(pd.DataFrame(rows))

sdf = pd.concat([f for f in frames if len(f)], ignore_index=True)
if not len(sdf):
    st.info("No spec-sheet comparisons stored in the graph — run "
            "`python -m src.kg.spec_consistency`.")
    st.stop()

comparisons = sdf[sdf["verdict"].isin(COMPARISON_VERDICTS)]
n_gap = int((sdf["verdict"] == "coverage_gap").sum())
problems = comparisons[comparisons["verdict"] != "consistent"].copy()
problems["rank"] = (problems["provenance"] != "manufacturer_internal").astype(int)
problems = problems.sort_values(["rank", "cell", "property"])

st.markdown(f"## {len(problems)} of {len(comparisons)} spec comparisons "
            "don't add up")
st.caption("We compared the same battery's numbers across different official "
           "documents — these are the places where the paperwork disagrees "
           "with itself.")
st.caption(f"{n_gap} more specs appear in only one document — nothing to "
           "compare them against.")

for _, r in problems.iterrows():
    stamp = ("🏭 same manufacturer, different documents"
             if r["provenance"] == "manufacturer_internal"
             else "🌐 across the ecosystem")
    hard = r["verdict"] == "document_variant_conflict"
    same_value = r["value_a"] == r["value_b"]
    with st.container(border=True):
        head, tag = st.columns([3, 2])
        head.markdown(f"**{r['cell']} — {prop_label(r['property'])}**")
        tag.markdown(("🟥 " if hard else "🟧 ") + stamp)
        if same_value and not hard:
            cond = parse_conditions(r["detail"])
            st.markdown(f"⚠️ same number, different test conditions — both "
                        f"documents say **{fmt_value(r['property'], r['value_a'])}**"
                        + (f", but disagree on the *{cond[0]}*:" if cond else ":"))
            if cond:
                v1, vs, v2 = st.columns([2, 1, 2])
                v1.metric(r["source_a"], cond[1])
                vs.markdown("<h3 style='text-align:center'>vs</h3>",
                            unsafe_allow_html=True)
                v2.metric(r["source_b"], cond[2])
        else:
            v1, vs, v2 = st.columns([2, 1, 2])
            v1.metric(r["source_a"], fmt_value(r["property"], r["value_a"]))
            vs.markdown("<h3 style='text-align:center'>vs</h3>",
                        unsafe_allow_html=True)
            v2.metric(r["source_b"], fmt_value(r["property"], r["value_b"]))
        with st.expander("ℹ️ Show me the exact source lines"):
            st.markdown(f"**{r['source_a']}:**\n> {r['fragment_a']}\n\n"
                        f"**{r['source_b']}:**\n> {r['fragment_b']}\n\n"
                        f"*Stored analysis:* {r['detail']}"
                        + ("" if hard else "\n\n*(Same value, but stated under "
                           "different test conditions — an apples-to-oranges "
                           "risk, not necessarily a contradiction.)*"))

with st.expander(f"ℹ️ All {len(sdf)} stored rows: {len(comparisons)} "
                 f"comparisons (incl. "
                 f"{int((sdf['verdict'] == 'consistent').sum())} consistent) "
                 f"+ {n_gap} single-document specs"):
    st.dataframe(sdf[["cell", "property", "value_a", "value_b", "source_a",
                      "source_b", "verdict", "provenance", "detail"]],
                 use_container_width=True)
    st.markdown("Values are compared after unit normalization (2% tolerance; "
                "bound satisfaction counts as consistent) by "
                "`src/kg/spec_consistency.py`; each row stores which documents "
                "said what, so every disagreement traces to its sources.")

driver.close()
