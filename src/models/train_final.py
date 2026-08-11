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
condition the threshold was derived under).

The abstention threshold is the quantile-referenced gate (src.models.
quantile_gate): the q*=41st percentile of the bank's own leave-one-out coverage
distribution, which reproduces the paper's 60% in-study retention while being
scale-free. It replaced an absolute constant picked from the risk-coverage
sweep, which could not transfer to another study's feature scaling.

The threshold is derived from the FROZEN publication edge snapshot
(data/kg_snapshots/severson_edges_publication.json), never from a live graph.
Behaviour-view SIMILAR_TO weights are z-scored over whatever cell population
the graph holds, so once a second study is loaded every Severson weight shifts
and the artifacts would silently stop matching the published model. Only the
CV/model path reads the live KG, and it must be run against a Severson-only
graph; --abstention-only rewrites the threshold alone and needs no database.

Run:  python -m src.models.train_final                   # full retrain (Severson-only KG)
      python -m src.models.train_final --abstention-only # rewrite the gate in meta.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.kg.features import similarity, zscore
from src.models.common import XGB_PARAMS, make_model, metric_table
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset
from src.models.graph_model import fetch_edges, graph_feature_columns, run_graph_model
from src.models.quantile_gate import (BEHAVIOR_VIEW, Q_STAR, fold_loo_distributions,
                                      keep_mask_for_q, loo_coverage, threshold_at_q)
from src.viz.make_paper_figures import publication_edges

ARTIFACT_DIR = ROOT / "app" / "artifacts"
BEHAVIOR_FEATURES = ["var_dQ_100_10", "min_dQ_100_10", "cap_ratio_100_2", "slope_2_100"]
K_NEIGHBORS = 5


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


def select_threshold(df: pd.DataFrame, graph_preds: pd.DataFrame, edges_view: dict,
                     q: float = Q_STAR, k: int = K_NEIGHBORS) -> dict:
    """The deployed quantile-referenced gate, as an absolute serving threshold.

    Two different LOO distributions are involved, for two different questions.

    The SERVED threshold is the q-th percentile of the FULL bank's LOO coverage
    distribution: at serve time the bank is all the model has, so that is the
    population a query cell is being compared against.

    The reported CV numbers use PER-FOLD distributions instead, so that no test
    cell contributes to the percentile that judges it. They are the honest
    in-study retention and retained RMSE, and they are what the paper quotes.
    """
    groups = dict(zip(df["cell_id"], df["policy_group_id"]))
    threshold = threshold_at_q(loo_coverage(df["cell_id"], groups, edges_view, k), q)

    keep = keep_mask_for_q(graph_preds, fold_loo_distributions(df, edges_view, k), q)
    t = 10.0 ** graph_preds["y_true_log"].to_numpy(float)[keep]
    p = 10.0 ** graph_preds["y_pred_log"].to_numpy(float)[keep]
    return {"threshold": threshold,
            "q_star": float(q),
            "cv_n_retained": int(keep.sum()),
            "cv_frac_retained": float(keep.mean()),
            "cv_rmse_retained_cycles": float(np.sqrt(((t - p) ** 2).mean())),
            "rule": (f"coverage >= the q={q:g} percentile of the bank's "
                     "leave-one-out coverage distribution (quantile-referenced "
                     "gate; retention and retained RMSE are grouped-CV, "
                     "per-fold thresholds)"),
            "derived_from": "data/kg_snapshots/severson_edges_publication.json"}


def rewrite_abstention() -> None:
    """Recompute only the abstention block of an existing meta.json.

    The rest of the artifact — model, quantile models, bank, scaler, CV metrics —
    is the publication training run and is left byte-for-byte alone. This path
    reads the frozen edge snapshot and needs no database.
    """
    meta_path = ARTIFACT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text())
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = publication_edges()
    preds = run_graph_model(df, edges)
    old, thr = meta.get("abstention", {}), select_threshold(df, preds, edges[BEHAVIOR_VIEW])
    meta["abstention"] = thr
    meta["abstention_rewritten_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[train_final] abstention: threshold {old.get('threshold', float('nan')):.6f} "
          f"-> {thr['threshold']:.6f}, retained "
          f"{old.get('cv_rmse_retained_cycles', float('nan')):.1f} -> "
          f"{thr['cv_rmse_retained_cycles']:.1f} cycles RMSE "
          f"({thr['cv_n_retained']} of {len(df)} cells)")
    print(f"[train_final] rule: {thr['rule']}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)

    # --- CV (for honest metrics) ----------------------------------------------
    edges = fetch_edges()
    cv_preds = run_graph_model(df, edges)
    cv_metrics = metric_table(cv_preds, "graph_cv")

    # The gate comes from the frozen publication snapshot, not from `edges`: a
    # live graph holding a second study re-z-scores the behaviour view and moves
    # every Severson weight, which would move the threshold with it.
    pub = publication_edges()
    thr = select_threshold(df, run_graph_model(df, pub), pub[BEHAVIOR_VIEW])

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
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--abstention-only", action="store_true",
                   help="recompute only the abstention block of meta.json "
                        "(frozen edge snapshot; no database, no retrain)")
    rewrite_abstention() if p.parse_args().abstention_only else main()
