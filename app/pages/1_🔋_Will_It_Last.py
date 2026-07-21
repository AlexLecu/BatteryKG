"""Will it last? — interactive cycle-life prediction toy over predict_with_gate().

All numbers come from the trained model artifacts in app/artifacts/ (honesty
rule); the early-cycle profile is always a real tested cell from the model's
neighbor bank, never synthesized.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import (BLUE, GREEN, INK_MUT, ORANGE,  # noqa: E402
                        load_artifacts, predict_with_gate)

st.set_page_config(page_title="Will it last? — BatteryKG", page_icon="🔋",
                   layout="wide")
st.title("🔋 Will it last?")

art, err = load_artifacts()
if art is None:
    st.error(f"Model artifacts unavailable — {err}")
    st.stop()

meta, bank = art["meta"], art["bank"]
CONDITION = ["c_rate_1", "c_rate_2", "soc_transition_pct"]

# slider ranges = the protocol space of the real training cells (neighbor bank)
LIM = {c: (float(bank[c].min()), float(bank[c].max())) for c in CONDITION}

# presets verified by scripts/verify_presets.py: same values, guaranteed outcome
PRESET_PASS = {"profile": "b1c29", "c1": 6.0, "c2": 3.0, "soc": 50.0}
PRESET_REFUSE = {"profile": "b2c1", "c1": 2.0, "c2": 6.0, "soc": 10.0}


def _apply(p):
    st.session_state["profile"] = p["profile"]
    st.session_state["c1"], st.session_state["c2"] = p["c1"], p["c2"]
    st.session_state["soc"] = p["soc"]


for key, val in [("profile", PRESET_PASS["profile"]), ("c1", PRESET_PASS["c1"]),
                 ("c2", PRESET_PASS["c2"]), ("soc", PRESET_PASS["soc"])]:
    st.session_state.setdefault(key, val)

b1, b2 = st.columns(2)
b1.button("▶ typical fast-charged cell", on_click=_apply, args=(PRESET_PASS,),
          use_container_width=True)
b2.button("▶ unusual usage pattern", on_click=_apply, args=(PRESET_REFUSE,),
          use_container_width=True, type="secondary")

left, right = st.columns([2, 3], gap="large")

NOMINAL_AH = 1.1        # A123 APR18650M1A datasheet nominal (the Severson cell)


def _speed_note(rate: float) -> str:
    """Live human translation of a C-rate on this cell."""
    return (f"{rate:.1f}C ≈ {rate * NOMINAL_AH:.1f} A on this "
            f"{NOMINAL_AH:.1f} Ah cell ≈ full charge in ~{60 / rate:.0f} min "
            "at this speed")


with left:
    st.subheader("Charging recipe")
    st.caption("How hard the battery is fast-charged — drag and watch the "
               "answer change.")
    c1 = st.slider("first charging speed — C-rate (1C = full charge in 1 hour)",
                   LIM["c_rate_1"][0], LIM["c_rate_1"][1], step=0.1, key="c1",
                   format="%.1fC")
    st.caption(_speed_note(c1))
    soc = st.slider("switch point — % state of charge where speed changes",
                    LIM["soc_transition_pct"][0], LIM["soc_transition_pct"][1],
                    step=1.0, key="soc", format="%.0f%%")
    c2 = st.slider("second charging speed — C-rate",
                   LIM["c_rate_2"][0], LIM["c_rate_2"][1], step=0.1, key="c2",
                   format="%.1fC")
    st.caption(_speed_note(c2))

    st.subheader("Measured battery profile")
    st.caption("Early-cycle measurement taken from a real tested cell — this "
               "part can't come from a slider, it has to be measured.")
    ids = bank["cell_id"].tolist()
    label = {r["cell_id"]: f"{r['cell_id']} — tested with {r['charge_policy_norm']}"
             for _, r in bank.iterrows()}
    profile = st.selectbox("real tested cell", ids, key="profile",
                           format_func=lambda i: label[i])

row = bank[bank["cell_id"] == profile].iloc[0]
query = {c: float(row[c]) for c in meta["base_features"]}
query["c_rate_1"], query["c_rate_2"], query["soc_transition_pct"] = c1, c2, soc

res = predict_with_gate(art, query)
cov, thr = res["coverage"], res["threshold"]

with right:
    if res["abstain"]:
        st.error("# ❌ I won't guess\n"
                 f"Too few similar batteries in my data (support {cov:.1f} vs "
                 f"required {thr:.1f}) — a standard model would have answered "
                 "anyway and been badly wrong on a case like this.")
    else:
        st.markdown(f"# ≈ {res['prediction_cycles']:,.0f} cycles")
        st.caption("Best estimate of full charge-discharge cycles before the "
                   "battery fades to 80% of its rated capacity.")
        # drawn bar = q16-q84 band widened (if needed) to include the point
        # estimate: the three models are independent, so the quantile pair can
        # sit slightly off the point model — the marker must stay inside the bar
        lo = min(res["band_lo"], res["prediction_cycles"])
        hi = max(res["band_hi"], res["prediction_cycles"])
        fig, ax = plt.subplots(figsize=(6.4, 0.9))
        ax.barh([0], [hi - lo], left=lo, height=0.5, color=BLUE, alpha=0.25)
        ax.plot([res["prediction_cycles"]], [0], "o", color=BLUE, ms=12,
                clip_on=False)
        ax.annotate(f"{lo:,.0f} – {hi:,.0f}", ((lo + hi) / 2, 0),
                    textcoords="offset points", xytext=(0, -24),
                    ha="center", color=INK_MUT)
        ax.set_xlim(0, max(2400.0, hi * 1.15))
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        st.caption("Likely range — the model's middle-68% uncertainty band, "
                   "stretched to include the best estimate when its separate "
                   "quantile models disagree slightly.")

    st.progress(min(cov / (2 * thr), 1.0),
                text=f"📊 data support: how many similar batteries I've seen — "
                     f"{cov:.1f} (refuses below {thr:.1f})")

    with st.expander("ℹ️ How this works"):
        st.markdown(f"""
- The predictor is the paper's graph-mediated XGBoost model, served unchanged
  through `predict_with_gate()`; nothing here is simulated.
- **Sliders** set the charge-protocol features (`c_rate_1`, `c_rate_2`,
  `soc_transition_pct`), ranged to the {meta['n_training_cells']} real
  Severson training cells.
- Harder charging generally shortens cycle life, but the Severson study
  showed the *combination* of the three settings matters, not just the peak
  rate — charging policy alone spanned 147–2236 cycles.
- **The profile** supplies the measured early-cycle features (capacity-fade
  shape over the first 100 cycles) plus batch — these are physical
  measurements, so they come from a real tested cell, honestly labeled.
- **Data support** (shown in the fuel gauge) is the model's *coverage*: the
  summed similarity of the 5 most similar training cells in early-fade
  behavior space. Below the threshold of {thr:.2f} (chosen on the
  cross-validated risk-coverage curve) the system **abstains** — it refuses
  to answer rather than extrapolate.
- The gauge reacts to the chosen profile (measured behavior), while the
  sliders steer the prediction itself — matching what the model actually
  consumes.
- Trained on {meta['n_training_cells']} cells; grouped-CV error
  {meta['cv_metrics']['rmse_cycles']:.0f} cycles RMSE
  ({meta['cv_metrics']['mape_pct']:.1f}% MAPE). Both preset buttons replay
  real training-bank cells verified to produce their outcome.
""")
