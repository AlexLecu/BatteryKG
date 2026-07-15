"""Shared pieces of the prediction experiment: seed, CV folds, metrics, model.

Everything downstream (baseline, graph model, abstention) imports from here so
the protocol — leave-one-policy-group-out CV, fixed seeds, one GBM config —
is defined exactly once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

SEED = 42

# One fixed GBM configuration for BOTH models (no tuning; the comparison is
# about the feature set, not hyperparameters). Modest capacity for n=120.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=4,
)


def make_model() -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMS)


def lopgo_folds(df: pd.DataFrame, group_col: str = "policy_group_id"):
    """Leave-one-policy-group-out folds: yield (group_id, train_idx, test_idx).

    Deterministic order (sorted group ids). Indices are positional into `df`.
    """
    groups = np.asarray(df[group_col])
    for gid in sorted(pd.unique(groups)):
        test = np.flatnonzero(groups == gid)
        train = np.flatnonzero(groups != gid)
        yield gid, train, test


# --- metrics (log space and back-transformed cycles) ------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_cycles(y_true_log, y_pred_log) -> float:
    return rmse(10.0 ** np.asarray(y_true_log, float),
                10.0 ** np.asarray(y_pred_log, float))


def mape_cycles(y_true_log, y_pred_log) -> float:
    t = 10.0 ** np.asarray(y_true_log, float)
    p = 10.0 ** np.asarray(y_pred_log, float)
    return float(np.mean(np.abs(p - t) / t) * 100.0)


def metric_table(pred_df: pd.DataFrame, label: str) -> dict:
    """Overall metrics from a predictions frame with y_true_log / y_pred_log."""
    return {
        "model": label,
        "rmse_log": rmse(pred_df["y_true_log"], pred_df["y_pred_log"]),
        "rmse_cycles": rmse_cycles(pred_df["y_true_log"], pred_df["y_pred_log"]),
        "mape_pct": mape_cycles(pred_df["y_true_log"], pred_df["y_pred_log"]),
    }
