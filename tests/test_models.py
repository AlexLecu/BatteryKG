"""Unit tests for the prediction/abstention experiment components."""
import numpy as np
import pandas as pd
import pytest

from src.models.abstention import risk_coverage_sweep
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset
from src.models.graph_model import (
    fold_graph_features,
    graph_feature_row,
    select_train_neighbors,
)


# --- fold-leakage assertion --------------------------------------------------
def test_poisoned_neighbor_set_raises():
    """Deliberately sneak a test cell into a neighbor set -> must raise."""
    train_ids = {"a", "b"}
    test_ids = {"x"}
    poisoned = [("a", 0.9), ("x", 0.8)]        # 'x' is a test cell
    with pytest.raises(AssertionError, match="LEAKAGE"):
        graph_feature_row(poisoned, train_ids, test_ids,
                          {"a": 2.7, "b": 2.8}, view="behavior")


def test_select_train_neighbors_filters_and_caps():
    nbrs = [("t1", 0.9), ("x", 0.8), ("t2", 0.7), ("t3", 0.6), ("t4", 0.5)]
    sel = select_train_neighbors(nbrs, {"t1", "t2", "t3", "t4"}, k=3)
    assert sel == [("t1", 0.9), ("t2", 0.7), ("t3", 0.6)]   # 'x' dropped, top-3 kept


def test_fold_features_never_use_test_cells():
    """End-to-end fold computation: test-cell neighbors restricted to train."""
    edges = {"v": {"c1": [("c2", 0.9), ("c3", 0.8)],
                   "c2": [("c1", 0.9), ("c3", 0.7)],
                   "c3": [("c1", 0.8), ("c2", 0.7)]}}
    train, test = {"c1", "c2"}, {"c3"}
    y = {"c1": 2.5, "c2": 2.9}
    gf = fold_graph_features(["c1", "c2", "c3"], edges, train, test, y, k=2)
    row3 = gf[gf["cell_id"] == "c3"].iloc[0]
    # c3's neighbors are both train cells; weighted mean of 2.5 (w .8) and 2.9 (w .7)
    expected = (0.8 * 2.5 + 0.7 * 2.9) / 1.5
    assert row3["v_nbr_wmean_log_life"] == pytest.approx(expected)
    assert row3["v_coverage_train"] == pytest.approx(1.5)
    # a train cell must not count itself
    row1 = gf[gf["cell_id"] == "c1"].iloc[0]
    assert row1["v_coverage_train"] == pytest.approx(0.9)   # only c2; c3 is test


def test_no_train_neighbors_gives_zero_coverage_and_nan():
    row = graph_feature_row([], {"a"}, {"x"}, {}, view="v")
    assert row["v_coverage_train"] == 0.0
    assert np.isnan(row["v_nbr_wmean_log_life"])


# --- abstention monotonicity on synthetic data --------------------------------
def test_risk_coverage_monotone_when_error_tracks_coverage():
    """|error| strictly decreasing in coverage -> retained risk must be
    non-increasing as the threshold rises (retention falls)."""
    n = 50
    coverage = np.linspace(0, 5, n)
    y_true = np.full(n, 3.0)                       # log10 cycle life = 3 -> 1000 cycles
    err = 0.2 * (1 - coverage / coverage.max())    # high coverage -> tiny error
    y_pred = y_true + err
    sweep = risk_coverage_sweep(coverage, y_true, y_pred, n_random=20, seed=0)
    d = sweep.sort_values("threshold")
    diffs = np.diff(d["rmse_retained"].to_numpy())
    assert (diffs <= 1e-9).all(), "risk must not increase as low-coverage cells drop"
    # and it must beat random at every retention level below 100 %
    below = d[d["frac_retained"] < 1.0]
    assert (below["rmse_retained"] <= below["rmse_random"] + 1e-9).all()


def test_risk_coverage_frac_retained_decreasing_in_threshold():
    rng = np.random.default_rng(1)
    coverage = rng.uniform(0, 5, 40)
    y_true = rng.normal(3, 0.2, 40)
    y_pred = y_true + rng.normal(0, 0.05, 40)
    sweep = risk_coverage_sweep(coverage, y_true, y_pred, n_random=10, seed=1)
    d = sweep.sort_values("threshold")
    assert (np.diff(d["n_retained"].to_numpy()) <= 0).all()
    assert d["frac_retained"].iloc[0] == 1.0       # lowest threshold keeps everyone


# --- dataset assembly ----------------------------------------------------------
def test_dataset_shape_and_nulls():
    # needs local processed data (not shipped in the public release — regenerate
    # with `python -m src.ingestion.download all && python -m src.ingestion.features`)
    try:
        df = build_dataset()
    except FileNotFoundError as e:
        pytest.skip(f"processed data not built: {e}")
    assert len(df) == 124
    assert df["cell_id"].nunique() == 124
    assert df["policy_group_id"].nunique() == 68
    assert int(df["is_anomalous"].sum()) == 4
    key = BASE_FEATURES + [TARGET, "policy_group_id"]
    assert not df[key].isna().any().any(), "no nulls allowed in modeling columns"
    # target is log10 of a 147..2236 cycle life
    assert df[TARGET].between(np.log10(140), np.log10(2300)).all()
    # usable set after QC exclusion
    kept = df[~df["is_anomalous"]]
    assert len(kept) == 120 and kept["policy_group_id"].nunique() == 66
