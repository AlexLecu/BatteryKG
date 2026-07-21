"""Promise vs reality — the manufacturer's cycle-life claim over what was measured.

Reuses pre-existing page Cypher verbatim (snapshot-compatible); all numbers
come from the knowledge graph. The plotted distribution is the SAME measurement
set the stored Discrepancy was computed from (Severson-only) — when the raw
measurement query also returns other studies (HUST shares the A123 cell model),
the Severson set is rebuilt client-side from the Graph Neighborhood queries.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import BLUE, ORANGE, make_driver, run_query  # noqa: E402

st.set_page_config(page_title="Promise vs reality — BatteryKG", page_icon="⚖️",
                   layout="wide")
st.title("⚖️ Promise vs reality")

driver, err = make_driver()
if driver is None:
    st.error(f"Knowledge graph unavailable — {err}")
    st.stop()

cells = run_query(driver, "MATCH (c:Cell) RETURN c.model AS model, "
                          "c.manufacturer AS mfr ORDER BY model")
if not cells:
    st.warning("No Cell nodes in the graph — run `python -m src.kg.claims`.")
    st.stop()

model = st.radio("Commercial cell", [c["model"] for c in cells], horizontal=True)

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

claims = run_query(driver, """
    MATCH (cl:Claim)-[:ABOUT]->(:Cell {model: $model})
    MATCH (cl)-[:ASSERTED_BY]->(s:Source)
    RETURN cl.property AS property, cl.value AS value, cl.unit AS unit,
           cl.conditions_text AS stated_conditions, cl.page AS page,
           s.document AS document
    ORDER BY document, property
""", model=model)

if meas:
    values = [m["v"] for m in meas]
    d = disc[0] if disc else None

    if d and len(values) != d["n"]:
        # the raw measurement query spans every study of this cell model (HUST
        # shares the A123), but the stored Discrepancy is Severson-only — rebuild
        # exactly its measurement set from the (already-snapshotted) Graph
        # Neighborhood queries: Severson instance ids are b<batch>c<n>.
        instances = run_query(driver, """
    MATCH (ci:CellInstance) RETURN ci.study_cell_id AS id ORDER BY id""")
        sev_ids = [i["id"] for i in instances
                   if re.match(r"^b[123]c\d+$", i["id"])]
        values = []
        for sid in sev_ids:
            r = run_query(driver, """
    MATCH (:CellInstance {study_cell_id: $center})<-[:ABOUT]-
          (m:Measurement {metric: 'cycle_life_nominal'})
    RETURN m.value AS life""", center=sid)
            if r and r[0]["life"] is not None:
                values.append(r[0]["life"])

    if d and len(values) != d["n"]:
        st.warning(f"Measurement set ({len(values)}) does not match the stored "
                   f"comparison (n={d['n']}) — showing nothing rather than a "
                   "mismatched plot.")
    else:
        plot_col, badge_col = st.columns([3, 2], gap="large")
        with plot_col:
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            ax.hist(values, bins=30, color=BLUE, alpha=0.85, edgecolor="white")
            if d:
                ymax = ax.get_ylim()[1]
                ax.axvline(d["claim"], color=ORANGE, lw=3.5, ls="--")
                ax.annotate("the promise", xy=(d["claim"], ymax * 0.97),
                            xytext=(d["claim"] + 260, ymax * 0.88),
                            color=ORANGE, fontweight="bold", fontsize=12,
                            arrowprops=dict(arrowstyle="->", color=ORANGE,
                                            lw=1.6))
                ax.axvline(d["med"], color=BLUE, lw=2)
                ax.annotate("half the cells died here",
                            xy=(d["med"], ymax * 0.55),
                            xytext=(d["med"] + 260, ymax * 0.68),
                            color=BLUE, fontsize=11,
                            arrowprops=dict(arrowstyle="->", color=BLUE,
                                            lw=1.4))
            ax.set_xlabel("cycles until the battery faded to 80% capacity")
            ax.set_ylabel(f"tested cells (n={len(values)})")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=True)
            st.caption(f"Every bar is one of n={len(values)} real tested cells "
                       "from the knowledge graph (the same measurement set the "
                       "stored comparison uses); the orange line is the "
                       "manufacturer's datasheet claim.")

        with badge_col:
            if d:
                m1, m2 = st.columns(2)
                m1.metric("Promised", f"{d['claim']:,.0f} cycles")
                m2.metric("Measured (median)", f"{d['med']:,.0f} cycles",
                          f"{d['gap']:+.1%} vs claim", delta_color="inverse")
                if not d["comparable"]:
                    st.warning("⚠️ test conditions differ — context, not a "
                               "verdict")
                with st.expander("ℹ️ How this works"):
                    st.markdown(f"""
- **Claim**: “{d['claim_conditions']}” (datasheet wording, page-referenced in
  the graph).
- **Measured**: {d['n']} cells, {d['measured_conditions']}.
- Spread: {d['mn']:.0f}–{d['mx']:.0f} cycles (middle half
  {d['q1']:.0f}–{d['q3']:.0f}).
- Stored rationale for the comparability flag: *{d['rationale']}*
- The comparison is a stored graph fact (a claim-vs-measured record with full
  provenance — every number traces to its source), recomputed by
  `python -m src.kg.discrepancy`, not made up on this page.
""")
            else:
                st.info("Measurements exist but no claim-vs-measured comparison "
                        "is stored in the graph for this cell yet.")
else:
    st.info("No independent cycling measurements for this cell in the graph "
            "yet — below is what the manufacturer promises, awaiting an "
            "independent check.")
    if claims:
        cdf = pd.DataFrame(claims)
        cdf["value"] = cdf["value"].astype(str)
        st.dataframe(cdf, use_container_width=True,
                     height=min(420, 40 + 35 * len(cdf)))
        with st.expander("ℹ️ How this works"):
            st.markdown("Claims are hand-extracted from the official datasheet "
                        "PDFs with page numbers and stated test conditions; "
                        "“unspecified” means the datasheet itself omits the "
                        "conditions — that omission is recorded, not filled in.")

driver.close()
