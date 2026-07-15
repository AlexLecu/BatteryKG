"""Cell Explorer — claims, measured distribution, and the discrepancy verdict."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import BLUE, INK_MUT, ORANGE, make_driver, run_query  # noqa: E402

st.set_page_config(page_title="Cell Explorer — BatteryKG", page_icon="🔎", layout="wide")
st.title("Cell Explorer")

driver, err = make_driver()
if driver is None:
    st.error(f"Knowledge graph unavailable — {err}")
    st.code("docker compose up -d neo4j\npython -m src.kg.load && python -m src.kg.claims "
            "&& python -m src.kg.discrepancy", language="bash")
    st.stop()

cells = run_query(driver, "MATCH (c:Cell) RETURN c.model AS model, "
                          "c.manufacturer AS mfr ORDER BY model")
if not cells:
    st.warning("No Cell nodes in the graph — run `python -m src.kg.claims`.")
    st.stop()

model = st.selectbox("Commercial cell", [c["model"] for c in cells])

# --- claims table -----------------------------------------------------------
claims = run_query(driver, """
    MATCH (cl:Claim)-[:ABOUT]->(:Cell {model: $model})
    MATCH (cl)-[:ASSERTED_BY]->(s:Source)
    RETURN cl.property AS property, cl.value AS value, cl.unit AS unit,
           cl.conditions_text AS stated_conditions, cl.page AS page,
           s.document AS document
    ORDER BY document, property
""", model=model)

st.subheader(f"Datasheet claims ({len(claims)})")
if not claims:
    st.warning("No claims loaded for this cell.")
else:
    df = pd.DataFrame(claims)
    df["value"] = df["value"].astype(str)
    n_unspec = (df["stated_conditions"] == "unspecified").sum()

    def _highlight(row):
        color = ("background-color: #fef3c7" if row["stated_conditions"] == "unspecified"
                 else "")
        return [color] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True,
                 height=min(420, 40 + 35 * len(df)))
    if n_unspec:
        st.caption(f"🟨 {n_unspec} of {len(df)} claims state **no measurement "
                   "conditions** (highlighted) — the omission is recorded, not filled in.")

# --- independent test measurements (lygte-info.dk) -----------------------------
lygte = run_query(driver, """
    MATCH (m:Measurement)-[:ABOUT]->(:Cell {model: $model})
    MATCH (m)-[:MEASURED_BY]->(s:Source {type: 'independent_test'})
    RETURN m.metric AS property, m.value AS value, m.unit AS unit,
           m.chart_only AS chart_only, m.cond_discharge_current_a AS discharge_a,
           m.cond_duration_min AS duration_min, m.cond_note AS note,
           m.source_fragment AS source_fragment,
           s.url AS url, s.retrieved AS retrieved
    ORDER BY property
""", model=model)

if lygte:
    st.subheader("Independent test — lygte-info.dk")
    ldf = pd.DataFrame(lygte)
    ldf["value"] = ldf.apply(
        lambda r: "chart only" if r["chart_only"] else f"{r['value']:g}", axis=1)

    def _chart_highlight(row):
        color = "background-color: #e5e7eb" if row["value"] == "chart only" else ""
        return [color] * len(row)

    show = ldf[["property", "value", "unit", "discharge_a", "duration_min",
                "note", "source_fragment"]]
    st.dataframe(show.style.apply(_chart_highlight, axis=1), use_container_width=True)
    n_chart = int(pd.DataFrame(lygte)["chart_only"].sum())
    st.caption(f"Source: {lygte[0]['url']} (retrieved {lygte[0]['retrieved']}, "
               "fetched once, parsed from cache). "
               + (f"⬜ {n_chart} measurement type(s) exist only as chart images on "
                  "the review page — recorded as `chart only`, no image extraction "
                  "attempted." if n_chart else ""))
    if n_chart:
        st.info("**Capacity reconciliation (claim vs independent measurement) is "
                "blocked:** the review's capacity-vs-current data is chart-only, "
                "so no numeric comparison is computable from text. The comparison "
                "logic is in place and activates if valued capacity measurements "
                "enter the graph.")

# --- specification consistency across document variants -----------------------
spec_rows = run_query(driver, """
    MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(:Cell {model: $model})
    RETURN d.property AS property, d.value_a AS value_a, d.value_b AS value_b,
           d.source_a AS source_a, d.source_b AS source_b,
           d.verdict AS verdict, d.rationale AS detail,
           d.provenance AS provenance,
           d.fragment_a AS fragment_a, d.fragment_b AS fragment_b
    ORDER BY d.verdict, d.property
""", model=model)
if spec_rows:
    st.subheader("Specification consistency across document variants")
    sdf = pd.DataFrame(spec_rows)
    n_cons = (sdf["verdict"] == "consistent").sum()
    n_cond = (sdf["verdict"] == "condition_mismatch").sum()
    n_conf = (sdf["verdict"] == "document_variant_conflict").sum()
    n_gap = (sdf["verdict"] == "coverage_gap").sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consistent", int(n_cons))
    c2.metric("Condition mismatch", int(n_cond))
    c3.metric("Conflicts", int(n_conf), delta_color="inverse")
    c4.metric("Coverage gaps", int(n_gap))

    problems = sdf[sdf["verdict"].isin(["document_variant_conflict",
                                        "condition_mismatch"])]
    if len(problems):
        st.markdown("**Conflicts & condition mismatches** "
                    "(expand a row for both source fragments verbatim):")
        for _, r in problems.iterrows():
            icon = "🟥" if r["verdict"] == "document_variant_conflict" else "🟧"
            prov = ("🏭 manufacturer-internal"
                    if r["provenance"] == "manufacturer_internal" else "🌐 ecosystem")
            with st.expander(f"{icon} {r['property']}: {r['value_a']} vs "
                             f"{r['value_b']} — {r['detail']} ({prov})"):
                st.markdown(f"**{r['source_a']}:**\n> {r['fragment_a']}")
                st.markdown(f"**{r['source_b']}:**\n> {r['fragment_b']}")
    with st.expander(f"All {len(sdf)} property comparisons (incl. consistent + gaps)"):
        st.dataframe(sdf[["property", "value_a", "value_b", "source_a",
                          "source_b", "verdict", "provenance", "detail"]],
                     use_container_width=True)
    st.caption("Verdicts from src/kg/spec_consistency.py: values compared after "
               "unit normalization (2% tolerance; bound satisfaction counts as "
               "consistent); condition mismatches (same value, different stated "
               "test conditions) flagged separately.")

# capacity discrepancies vs independent tests (if any are computable)
cap_disc = run_query(driver, """
    MATCH (d:Discrepancy)-[:ABOUT]->(:Cell {model: $model})
    WHERE d.property CONTAINS 'capacity' AND d.kind IS NULL
      AND d.measured_value IS NOT NULL
    RETURN d.property AS property, d.claim_value AS claim,
           d.measured_value AS measured, d.claim_current_a AS claim_a,
           d.measured_current_a AS measured_a, d.relative_gap AS gap,
           d.conditions_comparable AS comparable, d.rationale AS rationale
""", model=model)
if cap_disc:
    st.subheader("Capacity: claimed vs independently measured")
    for d in cap_disc:
        badge = "✅ comparable" if d["comparable"] else "⚠️ conditions differ"
        st.markdown(f"**{d['property']}** — claimed {d['claim']:g} Ah "
                    f"vs measured {d['measured']:g} Ah -> gap {d['gap']:+.1%} ({badge})")
        st.caption(d["rationale"])

# --- measured distribution + discrepancy (cells with measurements) ------------
meas = run_query(driver, """
    MATCH (m:Measurement {metric: 'cycle_life_nominal'})-[:ABOUT]->
          (ci:CellInstance)-[:INSTANCE_OF]->(:Cell {model: $model})
    WHERE m.value IS NOT NULL
    RETURN m.value AS v
""", model=model)

disc = run_query(driver, """
    MATCH (d:Discrepancy)-[:ABOUT]->(:Cell {model: $model})
    WHERE d.measured_median IS NOT NULL AND d.claim_value IS NOT NULL
    RETURN d.claim_value AS claim, d.measured_median AS med,
           d.measured_min AS mn, d.measured_max AS mx,
           d.measured_iqr_lo AS q1, d.measured_iqr_hi AS q3,
           d.n_measurements AS n, d.relative_gap AS gap,
           d.conditions_comparable AS comparable, d.rationale AS rationale,
           d.claim_conditions AS claim_conditions,
           d.measured_conditions AS measured_conditions
""", model=model)

if meas:
    st.subheader("Measured cycle life vs. the claim")
    values = [m["v"] for m in meas]
    d = disc[0] if disc else None

    col_plot, col_verdict = st.columns([3, 2])
    with col_plot:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.hist(values, bins=30, color=BLUE, alpha=0.8, edgecolor="white")
        if d:
            ax.axvline(d["claim"], color=ORANGE, lw=2.5, ls="--")
            ax.annotate(f"claimed: {d['claim']:.0f}", xy=(d["claim"], ax.get_ylim()[1]),
                        xytext=(d["claim"] + 30, ax.get_ylim()[1] * 0.92),
                        color=ORANGE, fontweight="bold")
            ax.axvline(d["med"], color=BLUE, lw=2)
            ax.annotate(f"measured median: {d['med']:.0f}", xy=(d["med"], 0),
                        xytext=(d["med"] + 30, ax.get_ylim()[1] * 0.72), color=BLUE)
        ax.set_xlabel("cycle life to 80% of nominal capacity")
        ax.set_ylabel(f"cells (n={len(values)})")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        st.caption("Distribution over the Severson instances in the knowledge graph.")

    with col_verdict:
        if d:
            st.markdown("#### ⚖️ Discrepancy verdict")
            m1, m2, m3 = st.columns(3)
            m1.metric("Claimed", f"{d['claim']:.0f}")
            m2.metric("Measured median", f"{d['med']:.0f}",
                      f"{d['gap']:+.1%} vs claim", delta_color="off")
            m3.metric("n", f"{d['n']}")
            if d["comparable"]:
                st.success("**Conditions comparable** — gap is a like-for-like difference.")
            else:
                st.error("**conditions_comparable: false** — this gap is *context, "
                         "not a verdict*.")
            with st.expander("Why not comparable? (stored rationale)", expanded=True):
                st.markdown(f"*{d['rationale']}*")
            st.caption(f"Claim conditions: “{d['claim_conditions']}” · "
                       f"Measured under: {d['measured_conditions']}")
        else:
            st.info("No Discrepancy node computed for this cell yet "
                    "(`python -m src.kg.discrepancy`).")
else:
    st.info("No cycling (degradation) measurements in the knowledge graph for "
            "this cell — the independent lygte-info.dk test above covers "
            "single-discharge behavior, not cycle life. The SNL 86-cell "
            "degradation study covering this cell is pending ingestion.")

driver.close()
