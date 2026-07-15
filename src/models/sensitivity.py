"""Robustness of the abstention result (extends experiment 01; same predictions).

Sensitivity of the risk-coverage curve to:
  1. k in the top-k coverage sum (k = 3 / 5 / 10), behavior view
  2. the similarity view at k=5 (behavior vs condition vs their mean)
  3. an alternative gate: the GBM's own uncertainty (per-fold XGBoost quantile
     spread, q84-q16 in log space) at equal retention
plus an AURC table with fold-bootstrap 95% CIs (1000 reps), all appended to
outputs/experiment_01_prediction.md as a '## Robustness' section.

The point predictions are exactly those of experiment 01's graph model (same
seed, same folds) — only the abstention signal varies.

Run:  python -m src.models.sensitivity
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

from src.config import OUTPUTS
from src.models.abstention import AURC_RANGE, aurc, risk_coverage_sweep
from src.models.common import SEED, XGB_PARAMS, lopgo_folds
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset
from src.models.graph_model import (
    K_GRAPH,
    fetch_edges,
    fold_graph_features,
    graph_feature_columns,
    run_graph_model,
    select_train_neighbors,
)

K_SWEEP = (3, 5, 10)
N_BOOTSTRAP = 1000

# CVD-validated palettes (scripts/validate_palette.js: ALL CHECKS PASS —
# purple was rejected by the validator: deutan ΔE 1.7 vs blue)
BLUE, ORANGE, GREEN, GOLD, PINK = "#2563eb", "#ea580c", "#059669", "#b45309", "#db2777"
INK, INK_MUT = "#1f2937", "#6b7280"
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_MUT, "ytick.color": INK_MUT,
})


# --- gating signals ---------------------------------------------------------
def coverage_signals(df: pd.DataFrame, edges) -> pd.DataFrame:
    """Per-cell coverage_train under every (view, k) combination.

    Mirrors the experiment's fold structure exactly: a cell's training set is
    every included cell outside its own policy group.
    """
    rows = {cid: {"cell_id": cid} for cid in df["cell_id"]}
    for _gid, train, test in lopgo_folds(df):
        train_ids = set(df.iloc[train]["cell_id"])
        for cid in df.iloc[test]["cell_id"]:
            for view, nbrs in edges.items():
                for k in K_SWEEP:
                    sel = select_train_neighbors(nbrs.get(cid, []), train_ids, k)
                    rows[cid][f"{view}_k{k}"] = float(sum(w for _, w in sel))
    out = pd.DataFrame(rows.values())
    out["mean_k5"] = (out["behavior_k5"] + out["condition_k5"]) / 2.0
    return out


def quantile_spread_signal(df: pd.DataFrame, edges) -> pd.DataFrame:
    """GBM self-uncertainty per cell: per-fold XGBoost quantile spread.

    Same features, folds, and seed as the graph model; q84-q16 in log space
    (clipped at 0 for occasional quantile crossings).
    """
    gcols = graph_feature_columns(tuple(edges))
    qp = {k: v for k, v in XGB_PARAMS.items()}
    rows = []
    for _gid, train, test in lopgo_folds(df):
        train_ids = set(df.iloc[train]["cell_id"])
        test_ids = set(df.iloc[test]["cell_id"])
        y_log_train = dict(zip(df.iloc[train]["cell_id"], df.iloc[train][TARGET]))
        gf = fold_graph_features(df["cell_id"].tolist(), edges,
                                 train_ids, test_ids, y_log_train, K_GRAPH)
        fold_df = df.merge(gf, on="cell_id", how="left")
        feats = BASE_FEATURES + gcols
        tr = fold_df[fold_df["cell_id"].isin(train_ids)]
        te = fold_df[fold_df["cell_id"].isin(test_ids)]

        lo = XGBRegressor(**qp, objective="reg:quantileerror", quantile_alpha=0.16)
        hi = XGBRegressor(**qp, objective="reg:quantileerror", quantile_alpha=0.84)
        lo.fit(tr[feats], tr[TARGET])
        hi.fit(tr[feats], tr[TARGET])
        spread = np.maximum(hi.predict(te[feats]) - lo.predict(te[feats]), 0.0)
        rows += [{"cell_id": c, "quantile_spread": float(s)}
                 for c, s in zip(te["cell_id"], spread)]
    return pd.DataFrame(rows)


# --- hybrid gate --------------------------------------------------------------
RETENTION_GRID = np.linspace(0.05, 1.0, 39)


def hybrid_gate_curve(pred: pd.DataFrame, cov: np.ndarray, spread: np.ndarray,
                      n_thresh: int = 21, retention_grid: np.ndarray = RETENTION_GRID):
    """Hybrid gate: keep a cell iff coverage >= t_cov AND spread <= t_spread.

    Threshold-selection rule (documented in the report): for each target
    retention level r and each CV fold, the (t_cov, t_spread) pair is chosen on
    the OTHER folds' cells only — among candidate pairs whose retention on
    those cells is >= r, pick the one minimising their retained RMSE (ties:
    higher retention). The chosen pair is then applied to the held-out fold's
    cells. No test-fold cell ever influences its own thresholds.

    Candidate thresholds are signal quantiles plus keep-all endpoints, so a
    feasible pair always exists at every retention level.

    Returns (curve_df, keep_matrix) where keep_matrix[i, j] says whether cell i
    is retained at retention_grid[j] (used by the fold bootstrap).
    """
    cov = np.asarray(cov, float)
    spread = np.asarray(spread, float)
    folds = pred["fold"].to_numpy()
    sq = (10 ** pred["y_true_log"].to_numpy() - 10 ** pred["y_pred_log"].to_numpy()) ** 2

    t_cov = np.concatenate([[-np.inf], np.unique(np.quantile(cov, np.linspace(0, 1, n_thresh)))])
    t_spr = np.concatenate([np.unique(np.quantile(spread, np.linspace(0, 1, n_thresh))), [np.inf]])
    pairs = [(a, b) for a in t_cov for b in t_spr]
    keep_by_pair = np.stack([(cov >= a) & (spread <= b) for a, b in pairs])   # (P, n)

    n = len(cov)
    keep_matrix = np.zeros((n, len(retention_grid)), dtype=bool)
    for f in np.unique(folds):
        test_mask = folds == f
        train_mask = ~test_mask
        kp_tr = keep_by_pair[:, train_mask]
        ret_tr = kp_tr.mean(axis=1)
        cnts = kp_tr.sum(axis=1)
        sums = kp_tr @ sq[train_mask]
        rmse_tr = np.where(cnts > 0, np.sqrt(sums / np.maximum(cnts, 1)), np.inf)
        for j, r in enumerate(retention_grid):
            cand = np.flatnonzero(ret_tr >= r - 1e-12)      # never empty (keep-all pair)
            best = cand[np.lexsort((-ret_tr[cand], rmse_tr[cand]))[0]]
            keep_matrix[test_mask, j] = keep_by_pair[best, test_mask]

    rows = []
    for j in range(len(retention_grid)):
        kept = keep_matrix[:, j]
        if kept.sum() == 0:
            continue
        rows.append({"frac_retained": float(kept.mean()),
                     "rmse_retained": float(np.sqrt(sq[kept].mean()))})
    curve = pd.DataFrame(rows).drop_duplicates()
    return curve, keep_matrix


def bootstrap_hybrid_aurc(pred: pd.DataFrame, keep_matrix: np.ndarray,
                          n_reps: int = N_BOOTSTRAP, seed: int = SEED):
    """AURC point + fold-bootstrap CI for the hybrid gate's keep decisions."""
    folds = pred["fold"].to_numpy()
    sq = (10 ** pred["y_true_log"].to_numpy() - 10 ** pred["y_pred_log"].to_numpy()) ** 2
    uniq = np.unique(folds)
    idx_by_fold = {f: np.flatnonzero(folds == f) for f in uniq}

    def curve_aurc(idx):
        fr, rk = [], []
        for j in range(keep_matrix.shape[1]):
            kept = keep_matrix[idx, j]
            if kept.sum() == 0:
                continue
            fr.append(kept.mean())
            rk.append(np.sqrt(sq[idx][kept].mean()))
        return aurc(fr, rk)

    point = curve_aurc(np.arange(len(folds)))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_reps)
    for r in range(n_reps):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        vals[r] = curve_aurc(np.concatenate([idx_by_fold[f] for f in chosen]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# --- bootstrap ---------------------------------------------------------------
def bootstrap_aurc(pred: pd.DataFrame, signal: np.ndarray,
                   n_reps: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """AURC point estimate + fold-bootstrap 95% CI for one gating signal."""
    point_sweep = risk_coverage_sweep(signal, pred["y_true_log"].to_numpy(),
                                      pred["y_pred_log"].to_numpy(), n_random=0)
    point = aurc(point_sweep["frac_retained"], point_sweep["rmse_retained"])

    folds = pred["fold"].to_numpy()
    uniq = np.unique(folds)
    idx_by_fold = {f: np.flatnonzero(folds == f) for f in uniq}
    yt = pred["y_true_log"].to_numpy()
    yp = pred["y_pred_log"].to_numpy()
    rng = np.random.default_rng(seed)

    vals = np.empty(n_reps)
    for r in range(n_reps):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_fold[f] for f in chosen])
        sw = risk_coverage_sweep(signal[idx], yt[idx], yp[idx], n_random=0)
        vals[r] = aurc(sw["frac_retained"], sw["rmse_retained"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_random_aurc(pred: pd.DataFrame, n_reps: int = N_BOOTSTRAP,
                          seed: int = SEED) -> tuple[float, float, float]:
    """Random-gate AURC: risk at any retention = overall RMSE (flat curve)."""
    t, p = 10 ** pred["y_true_log"].to_numpy(), 10 ** pred["y_pred_log"].to_numpy()
    sq = (t - p) ** 2
    point = float(np.sqrt(sq.mean()))
    folds = pred["fold"].to_numpy()
    uniq = np.unique(folds)
    idx_by_fold = {f: np.flatnonzero(folds == f) for f in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_reps)
    for r in range(n_reps):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_fold[f] for f in chosen])
        vals[r] = np.sqrt(sq[idx].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# --- plots ---------------------------------------------------------------------
def _plot_curves(curves, path, title):
    """curves: list of (label, sweep_df or None-for-random, color, linestyle)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, sweep, color, ls in curves:
        d = sweep.sort_values("frac_retained")
        ax.plot(d["frac_retained"], d["rmse_retained"], lw=2, color=color,
                ls=ls, label=label)
    ax.set_xlabel("fraction of cells retained")
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = fetch_edges()

    # experiment-01 predictions (identical seed/protocol)
    pred = run_graph_model(df, edges).sort_values("cell_id").reset_index(drop=True)
    yt, yp = pred["y_true_log"].to_numpy(), pred["y_pred_log"].to_numpy()
    err_log = np.abs(yt - yp)

    # gating signals, aligned to pred rows
    cov = coverage_signals(df, edges).set_index("cell_id").loc[pred["cell_id"]]
    qs = (quantile_spread_signal(df, edges).set_index("cell_id")
          .loc[pred["cell_id"], "quantile_spread"].to_numpy())

    signals = {
        **{f"behavior k={k}": cov[f"behavior_k{k}"].to_numpy() for k in K_SWEEP},
        "condition k=5": cov["condition_k5"].to_numpy(),
        "mean(views) k=5": cov["mean_k5"].to_numpy(),
        "GBM quantile spread": -qs,          # low spread = confident -> retain
    }

    # --- sweeps (with random baseline once, from the k=5 behavior sweep) -----
    sweeps = {name: risk_coverage_sweep(sig, yt, yp) for name, sig in signals.items()}
    rand_curve = sweeps["behavior k=5"][["frac_retained", "rmse_random"]].rename(
        columns={"rmse_random": "rmse_retained"})

    # --- hybrid gate: coverage AND spread, thresholds chosen on other folds ---
    hybrid_curve, hybrid_keep = hybrid_gate_curve(
        pred, signals["behavior k=5"], qs)

    # --- plot 1: k sensitivity ------------------------------------------------
    _plot_curves(
        [("random abstention", rand_curve, INK_MUT, "--"),
         ("behavior k=3", sweeps["behavior k=3"], BLUE, "-"),
         ("behavior k=5", sweeps["behavior k=5"], ORANGE, "-"),
         ("behavior k=10", sweeps["behavior k=10"], GREEN, "-")],
        OUTPUTS / "risk_coverage_k_sensitivity.png",
        "Risk-coverage vs k (behavior-view coverage_train)")

    # --- plot 2: view sensitivity + uncertainty gate ---------------------------
    _plot_curves(
        [("random abstention", rand_curve, INK_MUT, "--"),
         ("behavior k=5", sweeps["behavior k=5"], ORANGE, "-"),
         ("condition k=5", sweeps["condition k=5"], BLUE, "-"),
         ("mean(views) k=5", sweeps["mean(views) k=5"], GREEN, "-"),
         ("GBM quantile spread", sweeps["GBM quantile spread"], GOLD, "-"),
         ("hybrid (coverage + spread)", hybrid_curve, PINK, "-")],
        OUTPUTS / "risk_coverage_view_sensitivity.png",
        "Risk-coverage vs gating signal (k=5)")

    # --- spearman per view ------------------------------------------------------
    spearman = {}
    for name in ("behavior k=5", "condition k=5", "mean(views) k=5"):
        sig = signals[name]
        spearman[name] = {
            "log": stats.spearmanr(sig, err_log),
            "cyc": stats.spearmanr(sig, np.abs(10 ** yt - 10 ** yp)),
        }
    spearman["GBM quantile spread"] = {
        "log": stats.spearmanr(qs, err_log),      # spread itself vs error (positive = signal)
        "cyc": stats.spearmanr(qs, np.abs(10 ** yt - 10 ** yp)),
    }

    # --- AURC table with bootstrap CIs -------------------------------------------
    aurc_rows = []
    for name, sig in signals.items():
        point, lo, hi = bootstrap_aurc(pred, np.asarray(sig, float))
        aurc_rows.append({"gate": name, "aurc": point, "ci_lo": lo, "ci_hi": hi})
    hp, hlo, hhi = bootstrap_hybrid_aurc(pred, hybrid_keep)
    aurc_rows.append({"gate": "hybrid (coverage + spread)", "aurc": hp,
                      "ci_lo": hlo, "ci_hi": hhi})
    rp, rlo, rhi = bootstrap_random_aurc(pred)
    aurc_rows.append({"gate": "random abstention", "aurc": rp, "ci_lo": rlo, "ci_hi": rhi})
    aurc_df = pd.DataFrame(aurc_rows)

    # --- append Robustness section to the report ----------------------------------
    report = OUTPUTS / "experiment_01_prediction.md"
    text = report.read_text() if report.exists() else "# Experiment 01\n"
    text = text.split("\n## Robustness")[0].rstrip() + "\n"   # idempotent re-run

    lines = ["", "## Robustness", ""]
    a = lines.append
    a("Sensitivity of the abstention result. Same predictions as above (graph model, "
      "seed 42); only the gating signal varies. AURC = mean RMSE (cycles) across "
      f"retention levels {AURC_RANGE[0]:.0%}-{AURC_RANGE[1]:.0%}, lower is better; "
      f"95% CIs from {N_BOOTSTRAP} fold-bootstrap resamples.\n")
    a("**Hybrid gate selection rule (no test-set tuning):** keep a cell iff behavior "
      "coverage >= t_cov AND quantile spread <= t_spread. For each target retention "
      "level and each CV fold, the threshold pair is selected on the OTHER folds' "
      "cells only (among pairs meeting the retention target there, minimise retained "
      "RMSE; ties -> higher retention), then applied to the held-out fold. Candidate "
      "thresholds are signal quantiles plus keep-all endpoints.\n")
    a("### AURC (area under risk-coverage)\n")
    a("| gate | AURC (cyc) | 95% CI |")
    a("|---|---|---|")
    for _, r in aurc_df.iterrows():
        a(f"| {r['gate']} | {r['aurc']:.1f} | [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] |")
    a("")
    a("### Spearman(gating signal, |error|)\n")
    a("| signal | rho (log-space err) | p | rho (cycles err) | p |")
    a("|---|---|---|---|---|")
    for name, d in spearman.items():
        note = " (spread vs err; positive = informative)" if "spread" in name else ""
        a(f"| {name}{note} | {d['log'].statistic:.3f} | {d['log'].pvalue:.3g} "
          f"| {d['cyc'].statistic:.3f} | {d['cyc'].pvalue:.3g} |")
    a("")
    a("### Plots\n")
    a("- `risk_coverage_k_sensitivity.png` — k in {3, 5, 10}, behavior view")
    a("- `risk_coverage_view_sensitivity.png` — behavior vs condition vs mean vs "
      "the GBM's own quantile-spread gate (all k=5)")
    report.write_text(text + "\n".join(lines) + "\n")
    print(f"[sensitivity] appended Robustness section -> {report}")

    print("\n=== AURC table ===")
    print(aurc_df.round(1).to_string(index=False))
    print("\n=== Spearman vs |err| (log space) ===")
    for name, d in spearman.items():
        print(f"  {name:22s} rho={d['log'].statistic:+.3f}  p={d['log'].pvalue:.3g}")


if __name__ == "__main__":
    main()
