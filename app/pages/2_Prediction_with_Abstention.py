"""Prediction with abstention — the coverage-gated serving path."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import load_artifacts, predict_with_gate, view_neighbors  # noqa: E402

st.set_page_config(page_title="Prediction — BatteryKG", page_icon="🎯", layout="wide")
st.title("Prediction with abstention")

art, err = load_artifacts()
if art is None:
    st.error(f"Model artifacts unavailable — {err}")
    st.stop()

meta, bank = art["meta"], art["bank"]
st.caption(f"Model trained {meta['trained_at']} on {meta['n_training_cells']} cells. "
           f"Grouped-CV: RMSE {meta['cv_metrics']['rmse_cycles']:.0f} cycles, "
           f"MAPE {meta['cv_metrics']['mape_pct']:.1f}%. Abstention threshold "
           f"{meta['abstention']['threshold']:.2f} chosen by: "
           f"{meta['abstention']['rule']}.")

mode = st.radio("Input", ["Known cell instance", "Manual early-cycle features"],
                horizontal=True)

if mode == "Known cell instance":
    cell_id = st.selectbox("Cell instance (Severson study)", bank["cell_id"].tolist())
    row = bank[bank["cell_id"] == cell_id].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}
    exclude_group = row["policy_group_id"]
    actual = float(row["cycle_life_nominal"])
    st.caption(f"Charge policy: `{row['charge_policy_norm']}` (group "
               f"{exclude_group}, excluded from its own neighborhood — the "
               "grouped-CV condition). Early-cycle features from cycles ≤ 100 only.")
else:
    st.markdown("Enter early-cycle (cycles ≤ 100) and protocol features. "
                "Defaults are the training-set medians (from the model artifact).")
    med = bank[meta["base_features"]].median()
    cols = st.columns(4)
    query = {}
    for i, feat in enumerate(meta["base_features"]):
        with cols[i % 4]:
            query[feat] = st.number_input(feat, value=float(med[feat]), format="%.5f")
    exclude_group = None
    actual = None

res = predict_with_gate(art, query, exclude_group)

st.divider()
gauge_col, verdict_col = st.columns([2, 3])

with gauge_col:
    st.markdown("#### KG neighborhood coverage")
    cov, thr = res["coverage"], res["threshold"]
    st.progress(min(cov / (thr * 2), 1.0),
                text=f"behavior-view coverage = {cov:.2f} (gate at {thr:.2f})")
    st.caption("Coverage = sum of the top-5 similarity weights into the model's "
               "training population, behavior view. The gate threshold was "
               "selected on the grouped-CV risk–coverage curve "
               f"(retains {meta['abstention']['cv_frac_retained']:.0%} of cells at "
               f"{meta['abstention']['cv_rmse_retained_cycles']:.0f} cycles RMSE).")

with verdict_col:
    if res["abstain"]:
        st.error(f"### ⛔ Abstained\n**Insufficient KG neighborhood** "
                 f"(coverage {res['coverage']:.2f} < threshold {res['threshold']:.2f}). "
                 "This cell has no well-covered neighborhood in behavior space — "
                 "a prediction here would be unreliable, so the system refuses "
                 "rather than guesses.")
    else:
        st.success(f"### Predicted cycle life: **{res['prediction_cycles']:.0f} cycles**")
        st.markdown(f"Uncertainty band (per-quantile models, q16–q84): "
                    f"**{res['band_lo']:.0f} – {res['band_hi']:.0f} cycles**")
    if actual is not None:
        st.caption(f"Actual measured cycle life of {cell_id}: **{actual:.0f} cycles** "
                   "(shown for reference; this cell was in the training population — "
                   "the honest error estimate is the grouped-CV RMSE above, not this gap).")

st.divider()
st.markdown("#### Nearest neighbors (behavior view, training population)")
nb = view_neighbors(art, "behavior", query, exclude_group)
nb_disp = nb.rename(columns={"charge_policy_norm": "charge policy",
                             "cycle_life_nominal": "measured cycle life"})
st.dataframe(nb_disp[["cell_id", "weight", "measured cycle life", "charge policy",
                      "policy_group_id"]].style.format({"weight": "{:.3f}",
                                                        "measured cycle life": "{:.0f}"}),
             use_container_width=True)
st.caption("Neighbors are listed whether the gate passes or abstains — an abstention "
           "is explainable by the low weights in this table.")
