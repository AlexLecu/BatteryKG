"""Prove the two Will-It-Last presets produce their advertised outcomes.

Replays exactly what app/pages/1_🔋_Will_It_Last.py computes for each preset
(real bank profile + slider overrides -> predict_with_gate) and asserts:
  ▶ typical fast-charged cell  -> predicts
  ▶ unusual usage pattern      -> refuses (abstains)

Run:  python -m scripts.verify_presets
"""
from __future__ import annotations

from app.common import load_artifacts, predict_with_gate

PRESET_PASS = {"profile": "b1c29", "c1": 6.0, "c2": 3.0, "soc": 50.0}
PRESET_REFUSE = {"profile": "b2c1", "c1": 2.0, "c2": 6.0, "soc": 10.0}


def run_preset(art, p):
    meta, bank = art["meta"], art["bank"]
    row = bank[bank["cell_id"] == p["profile"]].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}
    query["c_rate_1"], query["c_rate_2"] = p["c1"], p["c2"]
    query["soc_transition_pct"] = p["soc"]
    return predict_with_gate(art, query)


def main() -> None:
    art, err = load_artifacts()
    assert art is not None, err

    r = run_preset(art, PRESET_PASS)
    print(f"[preset ▶ typical fast-charged cell]  abstain={r['abstain']}  "
          f"support={r['coverage']:.2f} vs gate {r['threshold']:.2f}  "
          f"prediction={r['prediction_cycles']:.0f} cycles "
          f"(band {r['band_lo']:.0f}–{r['band_hi']:.0f})")
    assert not r["abstain"], "PASS preset unexpectedly abstained"

    r = run_preset(art, PRESET_REFUSE)
    print(f"[preset ▶ unusual usage pattern]      abstain={r['abstain']}  "
          f"support={r['coverage']:.2f} vs gate {r['threshold']:.2f}")
    assert r["abstain"], "REFUSE preset unexpectedly predicted"

    print("[verify_presets] both outcomes confirmed.")


if __name__ == "__main__":
    main()
