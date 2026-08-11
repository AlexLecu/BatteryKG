"""Task 2 — cost-sensitive choice of the abstention operating point.

Reviewer question: the gate's operating point is currently justified by a
risk--coverage curve, which says nothing about what an abstention is worth
relative to a wrong answer. Make that trade-off explicit.

Cost model (per cell, both terms dimensionless fractions on [0, 1]):

    cost(q) = r * frac_abstained(q) + errshare_retained(q)

    r                    = c_abst / c_err, the price of one abstention in units
                           of the error an average-error prediction costs
    frac_abstained(q)    = share of cells the gate refuses
    errshare_retained(q) = share of the population's total absolute error that
                           is still incurred because those cells were served
                           = sum(|err| over retained) / sum(|err| over all)

The normalisation makes the two endpoints readable: serving everything costs
exactly 1, abstaining on everything costs exactly r. So r = 1 is the point where
a blanket abstention and a blanket answer break even, r < 1 means abstentions
are cheap, r > 1 means they are expensive.

Deterministic: same 66-fold CV predictions as everywhere else, no new models.

Run:  python -m experiments.exp07_revision_extras.cost_sensitive
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.exp06_quantile_gate.gate import (
    BEHAVIOR_VIEW, fold_loo_distributions, keep_mask_for_q, threshold_at_q)
from src.config import OUTPUTS
from src.models.dataset import build_dataset
from src.models.graph_model import run_graph_model
from src.viz.make_paper_figures import publication_edges

OUT = OUTPUTS / "experiment_07_revision_extras"
DEPLOYED = OUTPUTS / "experiment_06_quantile_gate" / "deployed_gate.json"
Q_GRID = np.arange(0.0, 100.0, 1.0)
RATIOS = np.logspace(-2, 2, 81)          # 0.01 .. 100


def build_curve(pred: pd.DataFrame, dists: dict) -> pd.DataFrame:
    """Per-q retention, retained RMSE, and share of total absolute error served."""
    t = 10 ** pred["y_true_log"].to_numpy()
    p = 10 ** pred["y_pred_log"].to_numpy()
    abs_err = np.abs(t - p)
    total_err = abs_err.sum()
    rows = []
    for q in Q_GRID:
        keep = keep_mask_for_q(pred, dists, q)
        n_keep = int(keep.sum())
        rows.append({
            "q": float(q),
            "n_retained": n_keep,
            "frac_retained": float(keep.mean()),
            "frac_abstained": float(1.0 - keep.mean()),
            "errshare_retained": float(abs_err[keep].sum() / total_err),
            "rmse_retained": float(np.sqrt(((t[keep] - p[keep]) ** 2).mean()))
            if n_keep else np.nan,
            "mape_retained": float(np.mean(np.abs(p[keep] - t[keep]) / t[keep]) * 100)
            if n_keep else np.nan,
            "median_fold_threshold": float(np.median(
                [threshold_at_q(l, q) for l in dists.values()])),
        })
    return pd.DataFrame(rows)


def optimise(curve: pd.DataFrame, ratios: np.ndarray) -> pd.DataFrame:
    """For each cost ratio, the q minimising total cost (ties -> most retention)."""
    fa = curve["frac_abstained"].to_numpy()
    es = curve["errshare_retained"].to_numpy()
    rows = []
    for r in ratios:
        cost = r * fa + es
        best = int(np.lexsort((curve["q"].to_numpy(), cost))[0])   # min cost, then min q
        row = curve.iloc[best]
        rows.append({"cost_ratio": float(r), "q_opt": float(row["q"]),
                     "retention_pct": 100 * float(row["frac_retained"]),
                     "n_retained": int(row["n_retained"]),
                     "rmse_retained": float(row["rmse_retained"]),
                     "mape_retained": float(row["mape_retained"]),
                     "total_cost": float(cost[best]),
                     "cost_at_serve_all": float(r * fa[0] + es[0]),
                     "median_fold_threshold": float(row["median_fold_threshold"])})
    return pd.DataFrame(rows)


def excess_cost(curve: pd.DataFrame, ratios: np.ndarray, q_star: float) -> pd.DataFrame:
    """How much more the deployed q costs than the best q, at each cost ratio."""
    fa = curve["frac_abstained"].to_numpy()
    es = curve["errshare_retained"].to_numpy()
    i_star = int(np.flatnonzero(curve["q"].to_numpy() == q_star)[0])
    rows = []
    for r in ratios:
        cost = r * fa + es
        rows.append({"cost_ratio": float(r), "cost_q_star": float(cost[i_star]),
                     "cost_optimal": float(cost.min()),
                     "excess_pct": float(100 * (cost[i_star] / cost.min() - 1))})
    return pd.DataFrame(rows)


def implied_ratio(curve: pd.DataFrame, ratios: np.ndarray, q_star: float,
                  tol_pct: float = 5.0) -> dict:
    """Where the deployed q sits in cost terms.

    Reported two ways: the ratios at which it is exactly cost-optimal (if any),
    and — more useful, because over a discrete q grid the optimum jumps rather
    than sliding — the band of ratios where it costs no more than `tol_pct`
    above the best available q.
    """
    opt = optimise(curve, ratios)
    ex = excess_cost(curve, ratios, q_star)
    near = ex[ex["excess_pct"] <= tol_pct]
    hit = opt[opt["q_opt"] == q_star]
    out = {"exactly_optimal": bool(len(hit)), "tol_pct": tol_pct}
    if len(hit):
        out.update(ratio_lo=float(hit["cost_ratio"].min()),
                   ratio_hi=float(hit["cost_ratio"].max()))
    j = int((opt["q_opt"] - q_star).abs().idxmin())
    out.update(closest_ratio=float(opt.loc[j, "cost_ratio"]),
               closest_q_opt=float(opt.loc[j, "q_opt"]),
               min_excess_pct=float(ex["excess_pct"].min()),
               ratio_at_min_excess=float(ex.loc[ex["excess_pct"].idxmin(), "cost_ratio"]))
    if len(near):
        out.update(near_optimal_lo=float(near["cost_ratio"].min()),
                   near_optimal_hi=float(near["cost_ratio"].max()))
    return out


def figure(curve: pd.DataFrame, opt: pd.DataFrame, ex: pd.DataFrame, band: dict,
           q_star: float) -> None:
    import matplotlib.pyplot as plt
    from src.viz.paper_style import BLUE, GOLD, INK_MUT, ORANGE, W_FULL, apply_style
    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(W_FULL, 2.7))

    lo, hi = band.get("near_optimal_lo"), band.get("near_optimal_hi")
    if lo is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvspan(lo, hi, color=GOLD, alpha=0.16, lw=0)

    ax1.semilogx(opt["cost_ratio"], opt["q_opt"], lw=1.8, color=BLUE)
    ax1.axhline(q_star, ls="--", lw=1.2, color=ORANGE)
    ax1.annotate(f"deployed $q^*={q_star:g}$", xy=(0.02, q_star), xycoords=("axes fraction", "data"),
                 xytext=(0, 5), textcoords="offset points", fontsize=7, color=ORANGE)
    ax1.set_xlabel("cost ratio $r = c_\\mathrm{abst}/c_\\mathrm{err}$")
    ax1.set_ylabel("cost-optimal percentile $q$")
    ax1.set_title("Optimal operating point", pad=4, fontsize=8.5)
    ax1.set_ylim(-3, 103)

    ax2.semilogx(opt["cost_ratio"], opt["retention_pct"], lw=1.8, color=BLUE,
                 label="retention")
    ax2.axhline(60, ls="--", lw=1.2, color=ORANGE)
    ax2.set_xlabel("cost ratio $r = c_\\mathrm{abst}/c_\\mathrm{err}$")
    ax2.set_ylabel("retention at the optimum [%]")
    ax2.set_ylim(-3, 103)
    ax2b = ax2.twinx()
    ax2b.semilogx(opt["cost_ratio"], opt["rmse_retained"], lw=1.4, ls=":", color=INK_MUT)
    ax2b.set_ylabel("RMSE on retained [cycles]", color=INK_MUT)
    ax2b.tick_params(axis="y", colors=INK_MUT)
    ax2b.grid(False)
    ax2.set_title("What the optimum buys", pad=4, fontsize=8.5)
    ax2.annotate("60% (paper)", xy=(0.02, 60), xycoords=("axes fraction", "data"),
                 xytext=(0, 4), textcoords="offset points", fontsize=7, color=ORANGE)

    ax3.semilogx(ex["cost_ratio"], ex["excess_pct"], lw=1.8, color=ORANGE)
    ax3.axhline(band["tol_pct"], ls="--", lw=1.0, color=INK_MUT)
    ax3.set_xlabel("cost ratio $r = c_\\mathrm{abst}/c_\\mathrm{err}$")
    ax3.set_ylabel(f"excess cost of $q^*={q_star:g}$ [%]")
    ax3.set_title("Price of the fixed setting", pad=4, fontsize=8.5)
    ax3.set_ylim(0, 60)
    if lo is not None:
        ax3.annotate(f"within {band['tol_pct']:.0f}% for\n$r\\in[{lo:.2f}, {hi:.2f}]$",
                     xy=(0.5, 0.70), xycoords="axes fraction", ha="center",
                     fontsize=6.8, color=GOLD)

    fig.suptitle("Cost-sensitive operating point, in-study "
                 "($r=1$: one abstention costs as much as one average-error answer)",
                 fontsize=8.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cost_sensitive.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q_star = float(json.loads(DEPLOYED.read_text())["q_star"])
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = publication_edges()
    pred = run_graph_model(df, edges).sort_values("cell_id").reset_index(drop=True)
    dists = fold_loo_distributions(df, edges[BEHAVIOR_VIEW])

    curve = build_curve(pred, dists)
    opt = optimise(curve, RATIOS)
    fine = np.logspace(-2, 2, 4001)
    band = implied_ratio(curve, fine, q_star)
    ex = excess_cost(curve, fine, q_star)

    curve.to_csv(OUT / "cost_curve_by_q.csv", index=False)
    opt.to_csv(OUT / "cost_optimal_by_ratio.csv", index=False)
    (OUT / "cost_implied_ratio.json").write_text(json.dumps(
        {"q_star": q_star, **band}, indent=1))
    ex.to_csv(OUT / "cost_excess_of_deployed_q.csv", index=False)
    figure(curve, opt, ex, band, q_star)

    print("=== cost-optimal operating point vs cost ratio ===")
    show = opt[np.isin(np.round(opt.cost_ratio, 6),
                       np.round(np.logspace(-2, 2, 9), 6))]
    print(show[["cost_ratio", "q_opt", "retention_pct", "rmse_retained",
                "mape_retained"]].round(2).to_string(index=False))
    print("\n=== implied cost ratio of the deployed setting ===")
    print(json.dumps(band, indent=1))
    return curve, opt, ex, band, q_star


if __name__ == "__main__":
    main()
