"""Unit tests for early-cycle feature extraction on a synthetic cell."""
import numpy as np
import pytest

from src.ingestion.features import CYCLE_MAX, compute_early_features


def _synthetic():
    # cycles 1..150; discharge capacity strictly linear: qd = 1.1 - 0.001*cyc
    cyc = np.arange(1, 151, dtype=float)
    qd = 1.1 - 0.001 * cyc
    ct = np.full_like(cyc, 10.0)
    # dQ(V) = qdlin_late - qdlin_early is a known 5-point vector
    qdlin_early = np.zeros(5)
    qdlin_late = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    return cyc, qd, ct, qdlin_early, qdlin_late


def test_features_match_analytic_values():
    feats = compute_early_features(*_synthetic())

    # dQ = [-0.02,-0.01,0,0.01,0.02]: mean 0, var 2e-4, min -0.02
    assert feats["var_dQ_100_10"] == pytest.approx(np.log10(2e-4))       # -3.69897
    assert feats["min_dQ_100_10"] == pytest.approx(np.log10(0.02))       # -1.69897
    # perfectly linear qd with -0.001 per cycle
    assert feats["slope_2_100"] == pytest.approx(-0.001, abs=1e-9)
    assert feats["capacity_cycle_2"] == pytest.approx(1.098)
    assert feats["capacity_cycle_100"] == pytest.approx(1.0)
    assert feats["cap_ratio_100_2"] == pytest.approx(1.0 / 1.098)
    assert feats["mean_charge_time_2_100"] == pytest.approx(10.0)


def test_charge_time_optional():
    cyc, qd, _ct, qe, ql = _synthetic()
    feats = compute_early_features(cyc, qd, None, qe, ql)
    assert np.isnan(feats["mean_charge_time_2_100"])


def test_hard_leakage_boundary_is_enforced():
    cyc, qd, ct, qe, ql = _synthetic()
    # asking for a dQ cycle beyond 100 must raise (label leakage tripwire)
    with pytest.raises(AssertionError):
        compute_early_features(cyc, qd, ct, qe, ql, late=CYCLE_MAX + 50)


def test_only_cycles_up_to_100_affect_slope():
    # corrupting cycles > 100 must NOT change any feature (boundary respected)
    cyc, qd, ct, qe, ql = _synthetic()
    base = compute_early_features(cyc, qd, ct, qe, ql)
    qd_corrupt = qd.copy()
    qd_corrupt[cyc > CYCLE_MAX] = 999.0          # garbage after cycle 100
    after = compute_early_features(cyc, qd_corrupt, ct, qe, ql)
    assert base == after
