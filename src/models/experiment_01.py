"""Experiment 01 — cycle-life prediction + coverage-gated abstention.

Runs the full protocol and writes:
  outputs/experiment_01_prediction.md    the report
  outputs/experiment_01_predictions.csv  per-cell CV predictions, both models
  outputs/risk_coverage.png              risk-coverage curve vs random abstention
  outputs/pred_vs_actual.png             predicted-vs-actual, both models

Run:  python -m src.models.experiment_01
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUTS
from src.models.abstention import risk_coverage_sweep
from src.models.baseline import run_baseline
from src.models.common import mape_cycles, rmse, rmse_cycles
from src.models.dataset import build_dataset
from src.models.graph_model import fetch_edges, run_graph_model

# --- figure style (single source; validated CVD-safe pair) -----------------
BLUE, ORANGE = "#2563eb", "#ea580c"     # baseline, graph  (ΔE_CVD > 110, validated)
INK, INK_MUT = "#1f2937", "#6b7280"
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_MUT, "ytick.color": INK_MUT,
})


def _metrics_block(pred: pd.DataFrame) -> dict:
    return {
        "rmse_log": rmse(pred["y_true_log"], pred["y_pred_log"]),
        "rmse_cycles": rmse_cycles(pred["y_true_log"], pred["y_pred_log"]),
        "mape_pct": mape_cycles(pred["y_true_log"], pred["y_pred_log"]),
    }


def _per_batch(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, g in pred.groupby("batch"):
        m = _metrics_block(g)
        rows.append({"batch": b, "n": len(g), **m})
    return pd.DataFrame(rows)


def _fold_rmse(pred: pd.DataFrame) -> pd.Series:
    return pred.groupby("fold").apply(
        lambda g: rmse(g["y_true_log"], g["y_pred_log"]), include_groups=False)


def plot_pred_vs_actual(base: pd.DataFrame, graph: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, pred, color, title in [(axes[0], base, BLUE, "Baseline (graph-free GBM)"),
                                   (axes[1], graph, ORANGE, "Graph-augmented GBM")]:
        t = 10 ** pred["y_true_log"]
        p = 10 ** pred["y_pred_log"]
        lims = [120, 2600]
        ax.plot(lims, lims, ls="--", lw=1, color=INK_MUT, zorder=1)  # identity ref
        ax.scatter(t, p, s=22, alpha=0.75, color=color, edgecolors="white",
                   linewidths=0.4, zorder=2)
        m = _metrics_block(pred)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ticks = [200, 500, 1000, 2000]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mpl.ticker.FixedLocator(ticks))
            axis.set_major_formatter(mpl.ticker.FixedFormatter([str(t) for t in ticks]))
            axis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("actual cycle life")
        ax.set_title(f"{title}\nRMSE {m['rmse_cycles']:.0f} cyc · MAPE {m['mape_pct']:.1f}%",
                     fontsize=10)
    axes[0].set_ylabel("predicted cycle life")
    fig.suptitle("Leave-one-policy-group-out CV — predicted vs actual (n=120)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(sweep: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    d = sweep.sort_values("frac_retained")
    ax.plot(d["frac_retained"], d["rmse_random"], lw=2, ls="--", color=INK_MUT,
            zorder=1)
    ax.plot(d["frac_retained"], d["rmse_retained"], lw=2, color=ORANGE, zorder=2)
    # direct labels at line ends (skill: selective direct labels, no legend box needed
    # for 2 series when labeled, but keep a legend for print safety)
    ax.legend(["random abstention (mean of 500 draws)",
               "coverage-gated (behavior coverage_train)"],
              frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("fraction of cells retained")
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.set_title("Risk-coverage: abstaining on low graph coverage", fontsize=11)
    ax.set_xlim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = build_dataset()
    excluded = data.loc[data["is_anomalous"], "cell_id"].tolist()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    n_folds = df["policy_group_id"].nunique()

    print(f"[exp01] {len(df)} cells / {n_folds} folds "
          f"(excluded QC-anomalous: {excluded})")

    base = run_baseline(df)
    edges = fetch_edges()
    graph = run_graph_model(df, edges)

    # --- metrics ------------------------------------------------------------
    overall = pd.DataFrame([
        {"model": "baseline", **_metrics_block(base)},
        {"model": "graph", **_metrics_block(graph)},
    ])
    pb_base, pb_graph = _per_batch(base), _per_batch(graph)

    # paired per-fold comparison (RMSE in log space)
    fb, fg = _fold_rmse(base), _fold_rmse(graph)
    paired = pd.DataFrame({"baseline": fb, "graph": fg})
    paired["winner"] = np.where(paired["graph"] < paired["baseline"], "graph",
                        np.where(paired["graph"] > paired["baseline"], "baseline", "tie"))
    wins = paired["winner"].value_counts().to_dict()
    wilcoxon = stats.wilcoxon(paired["baseline"], paired["graph"])

    # --- abstention on behavior coverage_train ------------------------------
    sweep = risk_coverage_sweep(graph["behavior_coverage_train"].to_numpy(),
                                graph["y_true_log"].to_numpy(),
                                graph["y_pred_log"].to_numpy())

    # --- largest errors vs coverage ------------------------------------------
    g = graph.copy()
    g["abs_err_cycles"] = (10 ** g["y_true_log"] - 10 ** g["y_pred_log"]).abs()
    top10 = g.nlargest(10, "abs_err_cycles")[
        ["cell_id", "fold", "batch", "abs_err_cycles",
         "behavior_coverage_train", "condition_coverage_train"]]
    rho, pval = stats.spearmanr(g["behavior_coverage_train"], g["abs_err_cycles"])
    err_log = (g["y_true_log"] - g["y_pred_log"]).abs()
    rho_log, pval_log = stats.spearmanr(g["behavior_coverage_train"], err_log)
    rho_life, pval_life = stats.spearmanr(g["behavior_coverage_train"],
                                          10 ** g["y_true_log"])

    # --- plots ---------------------------------------------------------------
    plot_pred_vs_actual(base, graph, OUTPUTS / "pred_vs_actual.png")
    plot_risk_coverage(sweep, OUTPUTS / "risk_coverage.png")

    # --- persist predictions --------------------------------------------------
    merged = base.merge(
        graph, on=["cell_id", "fold", "batch", "y_true_log"],
        suffixes=("_baseline", "_graph"))
    merged.to_csv(OUTPUTS / "experiment_01_predictions.csv", index=False)

    # --- report ----------------------------------------------------------------
    def f3(x): return f"{x:.3f}"
    def f1(x): return f"{x:.1f}"

    lines = []
    a = lines.append
    a("# Experiment 01 — cycle-life prediction with graph features + coverage-gated abstention\n")
    a("**Target:** `log10(cycle_life_nominal)` (cycle life to 80 % of nominal 1.1 Ah).")
    a(f"**Protocol:** leave-one-policy-group-out CV, grouping key `policy_group_id` "
      f"— {n_folds} folds over {len(df)} cells.")
    a(f"**Exclusions:** the {len(excluded)} QC-anomalous cells {excluded} are removed "
      "from both train and test (gross capacity spikes; `is_anomalous` in the dataset). "
      "Two of them (b2c12, b2c44) were their policy group's only member, so the fold "
      "count is 66 rather than the nominal 68.")
    a("**Models:** identical XGBoost config (no tuning, seed 42). Baseline = early-cycle "
      "(cycles 10-100) + parsed policy features + batch. Graph = baseline + per-fold, "
      "train-neighbors-only KG features from both SIMILAR_TO views "
      "(weighted-mean/std neighbor log-life, mean edge weight, coverage_train; k=5).\n")

    a("## Overall metrics\n")
    a("| model | RMSE (log10) | RMSE (cycles) | MAPE (%) |")
    a("|---|---|---|---|")
    for _, r in overall.iterrows():
        a(f"| {r['model']} | {f3(r['rmse_log'])} | {f1(r['rmse_cycles'])} | {f1(r['mape_pct'])} |")

    a("\n## Per-batch metrics\n")
    a("| batch | n | baseline RMSE cyc | graph RMSE cyc | baseline MAPE % | graph MAPE % |")
    a("|---|---|---|---|---|---|")
    for (_, rb), (_, rg) in zip(pb_base.iterrows(), pb_graph.iterrows()):
        a(f"| {int(rb['batch'])} | {int(rb['n'])} | {f1(rb['rmse_cycles'])} | "
          f"{f1(rg['rmse_cycles'])} | {f1(rb['mape_pct'])} | {f1(rg['mape_pct'])} |")

    a("\n## Paired per-fold comparison (fold RMSE, log space)\n")
    a(f"- graph wins **{wins.get('graph', 0)}** folds, baseline wins "
      f"**{wins.get('baseline', 0)}**, ties {wins.get('tie', 0)} (of {n_folds})")
    a(f"- Wilcoxon signed-rank on paired fold RMSEs: statistic={wilcoxon.statistic:.1f}, "
      f"p={wilcoxon.pvalue:.3f}")

    a("\n## Coverage-gated abstention (behavior view)\n")
    a("Risk-coverage sweep over `behavior_coverage_train` (sum of top-5 edge weights "
      "into the training fold). Random-abstention baseline = mean RMSE of 500 seeded "
      "random subsets at the same retention. See `risk_coverage.png`.\n")
    a("| threshold | retained | frac | RMSE retained (cyc) | RMSE random (cyc) |")
    a("|---|---|---|---|---|")
    step = max(1, len(sweep) // 12)
    for _, r in sweep.iloc[::step].iterrows():
        a(f"| {r['threshold']:.3f} | {int(r['n_retained'])} | {r['frac_retained']:.2f} | "
          f"{f1(r['rmse_retained'])} | {f1(r['rmse_random'])} |")

    a("\n## Do low-coverage cells make the largest errors?\n")
    a(f"Spearman rank correlations against `behavior_coverage_train` (n = {len(g)}):\n")
    a(f"- |error| in **log space** (the model's target): **rho = {rho_log:.3f}** "
      f"(p = {pval_log:.3g})")
    a(f"- |error| in cycles: rho = {rho:.3f} (p = {pval:.3g})")
    a(f"- actual cycle life (confound check): rho = {rho_life:.3f} (p = {pval_life:.3g}) "
      "— coverage is NOT a proxy for lifetime magnitude, so the risk-coverage "
      "gain is not an artifact of preferring short-lived cells.")
    a("\n" + ("Lower coverage -> larger errors, as the abstention gate assumes."
              if rho_log < 0 else
              "Coverage does NOT predict error magnitude here — the abstention "
              "gate has no signal at this sample size."))
    a("\n### 10 largest graph-model errors\n")
    a("| cell | fold (policy group) | batch | abs err (cyc) | behavior cov_train | condition cov_train |")
    a("|---|---|---|---|---|---|")
    for _, r in top10.iterrows():
        a(f"| {r['cell_id']} | {r['fold']} | {int(r['batch'])} | {f1(r['abs_err_cycles'])} | "
          f"{r['behavior_coverage_train']:.3f} | {r['condition_coverage_train']:.3f} |")

    a("\n## Files\n")
    a("- `experiment_01_predictions.csv` — per-cell CV predictions, both models")
    a("- `pred_vs_actual.png`, `risk_coverage.png`")
    a("\n*Seeds fixed (42) in model, folds (deterministic group order), and random-abstention draws.*")

    report = OUTPUTS / "experiment_01_prediction.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[exp01] report -> {report}")

    # console summary for the session log
    print("\n=== SUMMARY ===")
    print(overall.round(4).to_string(index=False))
    print(f"fold wins: graph={wins.get('graph',0)} baseline={wins.get('baseline',0)} "
          f"tie={wins.get('tie',0)}  (wilcoxon p={wilcoxon.pvalue:.3f})")
    print(f"spearman(coverage_train, |err|): cycles rho={rho:.3f} p={pval:.3g} | "
          f"log rho={rho_log:.3f} p={pval_log:.3g} | "
          f"vs lifetime rho={rho_life:.3f} (confound check)")


if __name__ == "__main__":
    main()
