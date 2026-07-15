"""Tests for the HUST loader (fixture-based), cross-study coverage, and
threshold-transfer logic."""
import numpy as np
import pandas as pd
import pytest

from src.ingestion.hust import process_cell


def _cycle_df(n_points=60, qd_mah=1100.0, v_hi=3.55, v_lo=2.05, charge_s=3000.0):
    """Synthetic HUST-style cycle DataFrame: charge then one discharge stage."""
    ch = pd.DataFrame({
        "Status": ["Constant current charge"] * 10,
        "Current (mA)": 5500.0,
        "Voltage (V)": np.linspace(2.7, 3.6, 10),
        "Capacity (mAh)": np.linspace(0, qd_mah, 10),
        "Time (s)": np.linspace(0, charge_s, 10),
    })
    dis = pd.DataFrame({
        "Status": ["Constant current discharge_0"] * n_points,
        "Current (mA)": -1100.0,
        "Voltage (V)": np.linspace(v_hi, v_lo, n_points),
        "Capacity (mAh)": np.linspace(qd_mah, 0, n_points),
        "Time (s)": np.linspace(charge_s, charge_s + 3600, n_points),
    })
    return pd.concat([ch, dis], ignore_index=True)


def _synthetic_cell(n_cycles=120, fade_per_cycle=0.5):
    return {n: _cycle_df(qd_mah=1100.0 - fade_per_cycle * n)
            for n in range(1, n_cycles + 1)}


# --- loader on a fixture ------------------------------------------------------------
def test_process_cell_rows_and_features():
    cycles = _synthetic_cell()
    rows, feats = process_cell("9-9", cycles, [5.0, 1.0, 1.0])
    assert len(rows) == 120
    r10 = next(r for r in rows if r["cycle_index"] == 10)
    assert r10["discharge_capacity_ah"] == pytest.approx(1.095, abs=1e-3)
    assert r10["cell_id"] == "HUST_9-9"
    assert r10["c_rate_charge"] == "5C(80%)-1C"
    assert r10["c_rate_discharge"] == "5C/1C/1C/1C stages"
    assert feats is not None
    # linear fade of 0.5 mAh/cycle -> slope = -0.0005 Ah/cycle
    assert feats["slope_2_100"] == pytest.approx(-5e-4, rel=0.05)
    assert feats["capacity_cycle_2"] == pytest.approx(1.099, abs=1e-3)
    assert feats["cap_ratio_100_2"] < 1.0
    assert np.isfinite(feats["var_dQ_100_10"])
    # charge time: 3000 s = 50 min
    assert feats["mean_charge_time_2_100"] == pytest.approx(50.0, rel=0.05)


def test_process_cell_skips_faulty_first_cycles():
    cycles = _synthetic_cell(n_cycles=10)
    rows_75, _ = process_cell("7-5", cycles, [3.0, 1.0, 3.0])   # skips 2 (quirk)
    rows_other, _ = process_cell("1-1", cycles, [5.0, 1.0, 1.0])
    assert len(rows_other) - len(rows_75) == 2
    assert min(r["cycle_index"] for r in rows_75) == 3


# --- cross-study coverage against the Severson-only bank ----------------------------
def test_cross_study_coverage_uses_severson_bank():
    from app.common import load_artifacts, view_neighbors
    art, err = load_artifacts()
    if art is None:
        pytest.skip(f"artifacts not built: {err}")
    meta, bank = art["meta"], art["bank"]
    # a query at the bank's median is densely covered...
    med = {c: float(bank[c].median()) for c in meta["base_features"]}
    nb_close = view_neighbors(art, "behavior", med, exclude_group=None)
    assert (nb_close["cell_id"].str.startswith("b")).all()   # Severson-only bank
    cov_close = nb_close["weight"].sum()
    # ...an out-of-distribution query is not
    far = dict(med)
    for c in meta["behavior_features"]:
        far[c] = float(bank[c].max() + 6 * bank[c].std())
    cov_far = view_neighbors(art, "behavior", far, exclude_group=None)["weight"].sum()
    assert cov_far < cov_close


# --- threshold transfer: the deployed gate applied unchanged --------------------------
def test_threshold_transfer_gate():
    from app.common import load_artifacts, predict_with_gate
    art, err = load_artifacts()
    if art is None:
        pytest.skip(f"artifacts not built: {err}")
    meta, bank = art["meta"], art["bank"]
    thr = meta["abstention"]["threshold"]
    med = {c: float(bank[c].median()) for c in meta["base_features"]}
    res = predict_with_gate(art, med, exclude_group=None)
    assert res["threshold"] == thr                       # transferred, not re-tuned
    assert res["abstain"] == (res["coverage"] < thr)
    far = dict(med)
    for c in meta["behavior_features"]:
        far[c] = float(bank[c].max() + 8 * bank[c].std())
    res_far = predict_with_gate(art, far, exclude_group=None)
    assert res_far["abstain"] is True                    # OOD -> gate refuses
