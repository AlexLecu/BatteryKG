"""Train the final serving model + abstention threshold -> app/artifacts/.

The app NEVER trains; it only loads these artifacts:
  model.json            XGBoost point model (log10 cycle life)
  model_q16.json/.q84   per-quantile models for the uncertainty band
  neighbor_bank.parquet the 120 training cells: behavior features (raw),
                        z-scored copies, log-life, policy group — the app uses
                        this to place a query cell in behavior space, compute
                        its neighbors/coverage, and build graph features
  meta.json             feature lists, z-score scaler (mean/std), abstention
                        threshold + its provenance, CV metrics, timestamps

Training set: all non-QC-flagged cells (120). Graph features for each training
cell are computed EXCLUDING its own policy group (mirroring the grouped-CV
condition the threshold was derived under). The abstention threshold is chosen
from the grouped-CV risk-coverage sweep: among thresholds retaining >=60% of
cells, the one minimising retained RMSE.

Run:  python -m src.models.train_final
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.kg.features import similarity, zscore
from src.models.abstention import risk_coverage_sweep
from src.models.common import XGB_PARAMS, make_model, metric_table
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset
from src.models.graph_model import fetch_edges, graph_feature_columns, run_graph_model

ARTIFACT_DIR = ROOT / "app" / "artifacts"
BEHAVIOR_FEATURES = ["var_dQ_100_10", "min_dQ_100_10", "cap_ratio_100_2", "slope_2_100"]
K_NEIGHBORS = 5
MIN_RETENTION = 0.60


def _graph_features_xgroup(df: pd.DataFrame, k: int = K_NEIGHBORS) -> pd.DataFrame:
    """Per-cell graph features over BOTH views, excluding same-policy-group
    neighbours (the grouped-CV condition). Computed from the modeling table
    itself so the artifact is self-contained."""
    from src.kg.features import parse_policy  # policy features already in df
    out = {c: [] for c in graph_feature_columns(("condition", "behavior"))}
    view_feats = {"condition": ["c_rate_1", "c_rate_2", "soc_transition_pct"],
                  "behavior": BEHAVIOR_FEATURES}
    Z = {v: zscore(df[cols].to_numpy(float)) for v, cols in view_feats.items()}
    groups = df["policy_group_id"].to_numpy()
    y = df[TARGET].to_numpy(float)
    for i in range(len(df)):
        for view in view_feats:
            sims = [(j, similarity(Z[view][i], Z[view][j]))
                    for j in range(len(df)) if groups[j] != groups[i]]
            sims.sort(key=lambda t: (-t[1], t[0]))
            top = sims[:k]
            w = np.array([s for _, s in top])
            yy = np.array([y[j] for j, _ in top])
            wmean = float(np.sum(w * yy) / np.sum(w))
            wstd = float(np.sqrt(np.sum(w * (yy - wmean) ** 2) / np.sum(w)))
            out[f"{view}_nbr_wmean_log_life"].append(wmean)
            out[f"{view}_nbr_wstd_log_life"].append(wstd)
            out[f"{view}_nbr_mean_weight"].append(float(w.mean()))
            out[f"{view}_coverage_train"].append(float(w.sum()))
    return pd.DataFrame(out, index=df.index)


def select_threshold(graph_preds: pd.DataFrame, min_retention: float = MIN_RETENTION) -> dict:
    """Abstention threshold from the CV risk-coverage sweep: among thresholds
    retaining >= min_retention, the one with minimal retained RMSE."""
    sweep = risk_coverage_sweep(graph_preds["behavior_coverage_train"].to_numpy(),
                                graph_preds["y_true_log"].to_numpy(),
                                graph_preds["y_pred_log"].to_numpy(), n_random=0)
    ok = sweep[sweep["frac_retained"] >= min_retention]
    row = ok.loc[ok["rmse_retained"].idxmin()]
    return {"threshold": float(row["threshold"]),
            "cv_frac_retained": float(row["frac_retained"]),
            "cv_rmse_retained_cycles": float(row["rmse_retained"]),
            "rule": f"min retained RMSE among thresholds with retention >= {min_retention:.0%}"}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)

    # --- CV (for honest metrics + threshold selection) ------------------------
    edges = fetch_edges()
    cv_preds = run_graph_model(df, edges)
    cv_metrics = metric_table(cv_preds, "graph_cv")
    thr = select_threshold(cv_preds)

    # --- final fit on all 120 cells (x-group graph features) -------------------
    gf = _graph_features_xgroup(df)
    X = pd.concat([df[BASE_FEATURES], gf], axis=1)
    feats = list(X.columns)
    y = df[TARGET]

    model = make_model()
    model.fit(X, y)
    model.save_model(ARTIFACT_DIR / "model.json")

    from xgboost import XGBRegressor
    for alpha, name in [(0.16, "model_q16.json"), (0.84, "model_q84.json")]:
        q = XGBRegressor(**XGB_PARAMS, objective="reg:quantileerror", quantile_alpha=alpha)
        q.fit(X, y)
        q.save_model(ARTIFACT_DIR / name)

    # --- neighbor bank + scaler --------------------------------------------------
    bank_cols = list(dict.fromkeys(
        ["cell_id", "policy_group_id", "charge_policy_norm", "batch",
         "cycle_life_nominal", TARGET] + BASE_FEATURES))
    bank = df[bank_cols].copy()
    bank.to_parquet(ARTIFACT_DIR / "neighbor_bank.parquet", index=False)

    scaler = {v: {"mean": df[cols].mean().tolist(), "std": df[cols].std(ddof=0).tolist(),
                  "columns": cols}
              for v, cols in [("behavior", BEHAVIOR_FEATURES),
                              ("condition", ["c_rate_1", "c_rate_2", "soc_transition_pct"])]}

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_training_cells": int(len(df)),
        "features": feats,
        "base_features": BASE_FEATURES,
        "behavior_features": BEHAVIOR_FEATURES,
        "k_neighbors": K_NEIGHBORS,
        "scaler": scaler,
        "abstention": thr,
        "cv_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in cv_metrics.items()},
        "target": TARGET,
        "note": ("graph features at serve time are computed against the neighbor "
                 "bank; for known instances the cell's own policy group is "
                 "excluded, mirroring the CV condition the threshold assumes"),
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[train_final] model + quantiles + bank + meta -> {ARTIFACT_DIR}")
    print(f"[train_final] CV: rmse_cycles={cv_metrics['rmse_cycles']:.1f} "
          f"mape={cv_metrics['mape_pct']:.2f}%")
    print(f"[train_final] abstention threshold={thr['threshold']:.3f} "
          f"({thr['rule']}; CV retained {thr['cv_frac_retained']:.0%} @ "
          f"{thr['cv_rmse_retained_cycles']:.0f} cycles RMSE)")


if __name__ == "__main__":
    main()
