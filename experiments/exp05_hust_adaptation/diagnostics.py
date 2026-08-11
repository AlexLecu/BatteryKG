"""Why retention does not recover: coverage ceiling and neighbour density.

Main result (run.py): retention stays at 0% for every n in {0,5,10,20,40}
while the retrained predictor's ungated error collapses. This module asks the
obvious follow-up — is the deployed threshold reachable on HUST at all? — and
quantifies what it would take.

Run:  python -m experiments.exp05_hust_adaptation.diagnostics
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.common import _zscore_with, load_artifacts
from experiments.exp05_hust_adaptation.core import (
    OUT, GraphIndex, ape, augmented_bank, design_matrix, hust_table, mape, rmse,
    severson_bank)
from src.models.common import SEED, make_model
from src.models.dataset import TARGET

N_PROBE = [0, 5, 10, 20, 40, 60]         # extends the requested grid; LOO row added after


def knn_distances(meta, bank, queries, exclude_self, exclude_group=None):
    """Distances to the k nearest behaviour-view neighbours (z-scored, deployed scaler)."""
    sc = meta["scaler"]["behavior"]
    cols, k = sc["columns"], meta["k_neighbors"]
    B = _zscore_with(bank[cols].to_numpy(float), sc["mean"], sc["std"])
    Q = _zscore_with(queries[cols].to_numpy(float), sc["mean"], sc["std"])
    D = np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2)
    if exclude_self:
        D[queries["cell_id"].to_numpy()[:, None] == bank["cell_id"].to_numpy()[None, :]] = np.inf
    if exclude_group is not None:
        m = np.asarray(exclude_group, bool)
        D[m[:, None] & (queries["policy_group_id"].to_numpy()[:, None]
                        == bank["policy_group_id"].to_numpy()[None, :])] = np.inf
    return np.sort(D, axis=1)[:, :k]


def main() -> tuple:
    art, _ = load_artifacts()
    meta, models = art["meta"], art["models"]
    thr = meta["abstention"]["threshold"]
    sev, hust = severson_bank(), hust_table()
    rng = np.random.default_rng(SEED)

    # --- coverage ceiling: extend n to leave-one-out --------------------------
    rows = []
    for n in N_PROBE:
        covs, rets = [], []
        for rep in range(5):
            r = np.random.default_rng(SEED + rep)
            add = ([] if n == 0 else
                   sorted(r.choice(hust["cell_id"].to_numpy(), n, replace=False).tolist()))
            bank = augmented_bank(sev, hust, add)
            ev = hust[~hust["cell_id"].isin(add)]
            c = GraphIndex(meta, bank).graph_features(ev)["behavior_coverage_train"].to_numpy()
            covs.append(c)
            rets.append((c >= thr).mean() * 100)
            if n == 0:                                # no randomness to repeat over
                break
        c = np.concatenate(covs)
        rows.append({"config": str(n), "n_in_bank": n, "n_eval": len(covs[0]),
                     "cov_min": c.min(), "cov_median": float(np.median(c)),
                     "cov_max": c.max(), "retention_pct": float(np.mean(rets)),
                     "cells_over_threshold": int((c >= thr).sum()), "n_cov_points": len(c)})

    # ceiling: every OTHER HUST cell in the bank, evaluated on all 77 (self-excluded)
    bank_loo = augmented_bank(sev, hust, hust["cell_id"].tolist())
    c = GraphIndex(meta, bank_loo).graph_features(
        hust, exclude_self=True)["behavior_coverage_train"].to_numpy()
    rows.append({"config": "leave-one-out (76 others)", "n_in_bank": 76, "n_eval": len(hust),
                 "cov_min": c.min(), "cov_median": float(np.median(c)), "cov_max": c.max(),
                 "retention_pct": float((c >= thr).mean() * 100),
                 "cells_over_threshold": int((c >= thr).sum()), "n_cov_points": len(c)})
    ceiling = pd.DataFrame(rows)

    # --- neighbour density: HUST vs the in-study population -------------------
    bank_full = augmented_bank(sev, hust, hust["cell_id"].tolist())
    d_hust = knn_distances(meta, bank_full, hust, exclude_self=True)
    d_sev = knn_distances(meta, bank_full, sev, exclude_self=True,
                          exclude_group=np.ones(len(sev), bool))
    # coverage 4.08 over k=5 needs mean weight 0.816 -> mean 5-NN distance 0.2255
    need_d = 1.0 / (thr / meta["k_neighbors"]) - 1.0
    have_d = float(np.median(d_hust.mean(1)))
    dim = len(meta["scaler"]["behavior"]["columns"])
    factor = (have_d / need_d) ** dim               # density scaling in `dim` dimensions

    dens = {
        "hust_mean_5nn_distance_LOO_median": have_d,
        "severson_mean_5nn_distance_xgroup_median": float(np.median(d_sev.mean(1))),
        "distance_required_for_threshold": float(need_d),
        "behavior_view_dimension": dim,
        "implied_cell_count_multiplier": float(factor),
        "implied_hust_cells_needed": float(factor * len(hust)),
    }

    # --- what the fixed threshold costs at n=40 (DIAGNOSTIC — not the protocol) --
    add = sorted(rng.choice(hust["cell_id"].to_numpy(), 40, replace=False).tolist())
    bank = augmented_bank(sev, hust, add)
    idx = GraphIndex(meta, bank)
    ev = hust[~hust["cell_id"].isin(add)].reset_index(drop=True)
    gf = idx.graph_features(ev)
    X = design_matrix(meta, ev, gf)
    is_sev = (bank["study"] == "severson").to_numpy()
    m = make_model()
    m.fit(design_matrix(meta, bank, idx.graph_features(bank, True, is_sev)),
          bank[TARGET].to_numpy(float))
    pred = 10.0 ** m.predict(X)
    actual = ev["cycle_life_nominal"].to_numpy(float)
    cov = gf["behavior_coverage_train"].to_numpy()
    alt = []
    for t in np.round(np.arange(0.0, 4.2, 0.25), 2):
        k = cov >= t
        if k.sum() == 0:
            alt.append({"threshold": t, "n_accepted": 0, "retention_pct": 0.0,
                        "rmse": np.nan, "mape": np.nan, "false_acceptance": 0})
            continue
        alt.append({"threshold": t, "n_accepted": int(k.sum()),
                    "retention_pct": 100 * k.mean(),
                    "rmse": rmse(actual[k], pred[k]), "mape": mape(actual[k], pred[k]),
                    "false_acceptance": int((ape(actual, pred)[k] > 30).sum())})
    alt_thr = pd.DataFrame(alt)

    ceiling.to_csv(OUT / "diag_coverage_ceiling.csv", index=False)
    alt_thr.to_csv(OUT / "diag_alt_threshold_n40.csv", index=False)
    (OUT / "diag_density.json").write_text(json.dumps(dens, indent=1))

    print("=== coverage ceiling ===")
    print(ceiling.round(3).to_string(index=False))
    print("\n=== neighbour density ===")
    for k, v in dens.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== n=40 bank+retrain under alternative thresholds (DIAGNOSTIC only) ===")
    print(alt_thr.round(2).to_string(index=False))


    return ceiling, dens, alt_thr


if __name__ == "__main__":
    main()
