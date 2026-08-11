"""Task 2 — the quantile gate on the full in-study Severson evaluation.

Re-runs the 66-fold leave-one-policy-group-out protocol unchanged (same
predictions, same seed, same pinned publication edge set) and swaps ONLY the
abstention rule: absolute coverage >= 4.08 becomes coverage >= the q-th
percentile of that fold's own training-bank LOO coverage distribution.

Everything is per-fold: the percentile is derived from the fold's training bank
alone, so no test cell influences the threshold that judges it.

The headline claims the paper makes about abstention must survive this swap.
This script asserts that rather than assuming it, and prints a loud banner if
anything moves.

Run:  python -m experiments.exp06_quantile_gate.in_study
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from experiments.exp06_quantile_gate.gate import (
    BEHAVIOR_VIEW, fold_loo_distributions, keep_mask_for_q, threshold_at_q)
from src.config import OUTPUTS
from src.models.abstention import AURC_RANGE, aurc, risk_coverage_sweep
from src.models.common import SEED, metric_table
from src.models.dataset import build_dataset
from src.models.graph_model import run_graph_model
from src.models.sensitivity import (
    N_BOOTSTRAP, bootstrap_aurc, bootstrap_hybrid_aurc, bootstrap_random_aurc,
    quantile_spread_signal)
from src.viz.make_paper_figures import publication_edges

OUT = OUTPUTS / "experiment_06_quantile_gate"
Q_GRID = np.round(np.arange(0.0, 100.0, 1.0), 2)      # percentile sweep
TARGET_RETENTION = 0.60                                # the paper's operating point
# published values this re-run must reproduce (outputs/experiment_01_prediction.md)
PUBLISHED = {"rmse_cycles": 135.15, "mape_pct": 9.69,
             "spearman_log": -0.247, "spearman_cyc": -0.179,
             "aurc_behavior_k5": 86.8, "aurc_random": 135.2,
             "aurc_quantile_spread": 86.4, "deployed_threshold": 4.080029}
TOL = {"rmse": 0.5, "mape": 0.05, "spearman": 0.01, "aurc": 1.0}


def quantile_curve(pred: pd.DataFrame, dists: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Risk-coverage curve for the quantile gate, plus the keep-matrix for bootstrapping."""
    sq = (10 ** pred["y_true_log"].to_numpy() - 10 ** pred["y_pred_log"].to_numpy()) ** 2
    keeps, rows = [], []
    for q in Q_GRID:
        keep = keep_mask_for_q(pred, dists, q)
        keeps.append(keep)
        if keep.sum() == 0:
            continue
        rows.append({"q": float(q), "n_retained": int(keep.sum()),
                     "frac_retained": float(keep.mean()),
                     "rmse_retained": float(np.sqrt(sq[keep].mean())),
                     "median_fold_threshold": float(np.median(
                         [threshold_at_q(l, q) for l in dists.values()]))})
    return pd.DataFrame(rows), np.stack(keeps, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = publication_edges()

    pred = run_graph_model(df, edges).sort_values("cell_id").reset_index(drop=True)
    m = metric_table(pred, "graph")
    yt, yp = pred["y_true_log"].to_numpy(), pred["y_pred_log"].to_numpy()
    err_log = np.abs(yt - yp)
    err_cyc = np.abs(10 ** yt - 10 ** yp)
    cov = pred["behavior_coverage_train"].to_numpy(float)

    # --- the two gates ---------------------------------------------------------
    dists = fold_loo_distributions(df, edges[BEHAVIOR_VIEW])
    abs_sweep = risk_coverage_sweep(cov, yt, yp)                     # absolute gate
    q_curve, q_keep = quantile_curve(pred, dists)                    # quantile gate

    # --- operating point: which q reproduces ~60% retention? -------------------
    q_curve["ret_err"] = (q_curve["frac_retained"] - TARGET_RETENTION).abs()
    op = q_curve.loc[q_curve["ret_err"].idxmin()]
    q_star = float(op["q"])
    thr_by_fold = {g: threshold_at_q(l, q_star) for g, l in dists.items()}

    # --- AURC ------------------------------------------------------------------
    a_abs, lo_abs, hi_abs = bootstrap_aurc(pred, cov)
    a_q, lo_q, hi_q = bootstrap_hybrid_aurc(pred, q_keep)            # keep-matrix bootstrap
    a_rnd, lo_rnd, hi_rnd = bootstrap_random_aurc(pred)
    qs = (quantile_spread_signal(df, edges).set_index("cell_id")
          .loc[pred["cell_id"], "quantile_spread"].to_numpy())
    a_spr, lo_spr, hi_spr = bootstrap_aurc(pred, -qs)

    # --- Spearman --------------------------------------------------------------
    # raw coverage is untouched by the swap, so this must be bit-identical;
    # the within-fold percentile rank is the quantile gate's own ordering.
    pct_rank = np.empty(len(pred))
    folds = pred["fold"].to_numpy()
    for gid, loo in dists.items():
        msk = folds == gid
        pct_rank[msk] = [stats.percentileofscore(loo.to_numpy(), c, kind="mean")
                         for c in cov[msk]]
    sp = {
        "absolute (raw coverage) vs |err| log": stats.spearmanr(cov, err_log),
        "absolute (raw coverage) vs |err| cyc": stats.spearmanr(cov, err_cyc),
        "quantile (within-fold pct) vs |err| log": stats.spearmanr(pct_rank, err_log),
        "quantile (within-fold pct) vs |err| cyc": stats.spearmanr(pct_rank, err_cyc),
    }

    # --- preservation checks ---------------------------------------------------
    checks = []

    def chk(name, got, want, tol, note=""):
        ok = abs(got - want) <= tol
        checks.append({"check": name, "value": round(float(got), 4),
                       "published": want, "tol": tol, "pass": bool(ok), "note": note})
        return ok

    chk("graph RMSE (cycles)", m["rmse_cycles"], PUBLISHED["rmse_cycles"], TOL["rmse"])
    chk("graph MAPE (%)", m["mape_pct"], PUBLISHED["mape_pct"], TOL["mape"])
    chk("Spearman raw coverage vs |err| log", sp["absolute (raw coverage) vs |err| log"].statistic,
        PUBLISHED["spearman_log"], TOL["spearman"])
    chk("Spearman raw coverage vs |err| cyc", sp["absolute (raw coverage) vs |err| cyc"].statistic,
        PUBLISHED["spearman_cyc"], TOL["spearman"])
    chk("AURC absolute gate", a_abs, PUBLISHED["aurc_behavior_k5"], TOL["aurc"])
    chk("AURC random", a_rnd, PUBLISHED["aurc_random"], TOL["aurc"])
    chk("AURC quantile-spread gate", a_spr, PUBLISHED["aurc_quantile_spread"], TOL["aurc"])
    chk("AURC quantile gate vs absolute gate", a_q, a_abs, TOL["aurc"],
        "the swap must not degrade the curve")
    chk("quantile gate beats random", a_rnd - a_q, a_rnd - a_abs, 2.0,
        "margin over random preserved")

    # the operating point specifically: same retention, but are they the same cells?
    thr_abs = PUBLISHED["deployed_threshold"]
    keep_abs = cov >= thr_abs
    keep_q = keep_mask_for_q(pred, dists, q_star)
    sq = (10 ** yt - 10 ** yp) ** 2
    rmse_abs_op = float(np.sqrt(sq[keep_abs].mean()))
    rmse_q_op = float(op["rmse_retained"])
    n_swapped = int((keep_abs != keep_q).sum())
    chk("retention at operating point", keep_q.mean(), keep_abs.mean(), 0.02)
    chk("retained RMSE at operating point", rmse_q_op, rmse_abs_op, 10.0,
        f"{n_swapped} of 120 cells change decision")
    checks_df = pd.DataFrame(checks)
    op_compare = {"n_retained_abs": int(keep_abs.sum()), "n_retained_quantile": int(keep_q.sum()),
                  "rmse_abs": rmse_abs_op, "rmse_quantile": rmse_q_op,
                  "delta_rmse": rmse_q_op - rmse_abs_op, "n_cells_swapped": n_swapped}

    # --- outputs ---------------------------------------------------------------
    abs_sweep.to_csv(OUT / "in_study_absolute_sweep.csv", index=False)
    q_curve.drop(columns="ret_err").to_csv(OUT / "in_study_quantile_sweep.csv", index=False)
    checks_df.to_csv(OUT / "in_study_preservation_checks.csv", index=False)
    pd.DataFrame([{"fold": g, "threshold_at_q": t,
                   "loo_median": float(dists[g].median()),
                   "loo_min": float(dists[g].min()), "loo_max": float(dists[g].max())}
                  for g, t in sorted(thr_by_fold.items())]).to_csv(
        OUT / "in_study_fold_thresholds.csv", index=False)

    deployed = {
        "operating_point_comparison": op_compare,
        "q_star": q_star,
        "target_retention": TARGET_RETENTION,
        "achieved_retention": float(op["frac_retained"]),
        "n_retained": int(op["n_retained"]),
        "rmse_retained_cycles": float(op["rmse_retained"]),
        "median_fold_threshold_at_q_star": float(np.median(list(thr_by_fold.values()))),
        "min_fold_threshold": float(np.min(list(thr_by_fold.values()))),
        "max_fold_threshold": float(np.max(list(thr_by_fold.values()))),
        "old_absolute_threshold": PUBLISHED["deployed_threshold"],
        "selected_on": "Severson in-study CV only; no HUST information used",
        "k_neighbors": 5, "view": BEHAVIOR_VIEW,
    }
    (OUT / "deployed_gate.json").write_text(json.dumps(deployed, indent=1))

    aurc_df = pd.DataFrame([
        {"gate": "coverage, absolute threshold (deployed)", "aurc": a_abs,
         "ci_lo": lo_abs, "ci_hi": hi_abs},
        {"gate": "coverage, quantile-referenced (new)", "aurc": a_q,
         "ci_lo": lo_q, "ci_hi": hi_q},
        {"gate": "GBM quantile spread", "aurc": a_spr, "ci_lo": lo_spr, "ci_hi": hi_spr},
        {"gate": "random abstention", "aurc": a_rnd, "ci_lo": lo_rnd, "ci_hi": hi_rnd},
    ])
    aurc_df.to_csv(OUT / "in_study_aurc.csv", index=False)

    _figure(abs_sweep, q_curve, aurc_df, deployed)
    _report(m, aurc_df, sp, checks_df, deployed, q_curve, dists)

    print("\n=== preservation checks ===")
    print(checks_df.to_string(index=False))
    if not checks_df["pass"].all():
        print("\n" + "!" * 78)
        print("!! HEADLINE SHIFTED — the quantile swap changed a published number.")
        print("!! Failing checks:")
        for _, r in checks_df[~checks_df["pass"]].iterrows():
            print(f"!!   {r['check']}: got {r['value']}, expected {r['published']} "
                  f"(tol {r['tol']})")
        print("!" * 78)
    else:
        print("\nAll preservation checks pass.")
    print(f"\nDeployed setting: q = {q_star:g}th percentile "
          f"-> retention {deployed['achieved_retention']:.1%}, "
          f"median fold threshold {deployed['median_fold_threshold_at_q_star']:.3f} "
          f"(old absolute {PUBLISHED['deployed_threshold']:.3f})")
    return checks_df


def _figure(abs_sweep, q_curve, aurc_df, deployed) -> None:
    import matplotlib.pyplot as plt
    from src.viz.paper_style import BLUE, GOLD, INK_MUT, ORANGE, W_FULL, apply_style
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_FULL, 3.0))

    a = abs_sweep.sort_values("frac_retained")
    q = q_curve.sort_values("frac_retained")
    ax1.plot(a["frac_retained"], a["rmse_random"], lw=1.3, ls="--", color=INK_MUT,
             label="random abstention")
    ax1.plot(a["frac_retained"], a["rmse_retained"], lw=1.9, color=ORANGE,
             label="absolute threshold (deployed)")
    ax1.plot(q["frac_retained"], q["rmse_retained"], lw=1.5, ls=":", color=BLUE,
             label="quantile-referenced (new)")
    ax1.axvline(deployed["achieved_retention"], lw=1.1, color=GOLD, ls="-.")
    ax1.annotate(f"operating point\nq={deployed['q_star']:g}th pct",
                 xy=(deployed["achieved_retention"], 0.30), xycoords=("data", "axes fraction"),
                 xytext=(5, 0), textcoords="offset points", fontsize=7, color=GOLD)
    ax1.set_xlabel("fraction of cells retained")
    ax1.set_ylabel("RMSE on retained cells [cycles]")
    ax1.set_xlim(0, 1.02)
    ax1.set_title("In-study risk–coverage: the swap is a no-op", pad=4, fontsize=8.5)
    ax1.legend(loc="lower right", fontsize=7)

    order = aurc_df.sort_values("aurc")
    ypos = np.arange(len(order))
    ax2.barh(ypos, order["aurc"], color=[ORANGE if "absolute" in g else
                                         BLUE if "quantile-ref" in g else
                                         GOLD if "spread" in g else INK_MUT
                                         for g in order["gate"]], height=0.6)
    ax2.errorbar(order["aurc"], ypos,
                 xerr=[order["aurc"] - order["ci_lo"], order["ci_hi"] - order["aurc"]],
                 fmt="none", ecolor=INK_MUT, elinewidth=1, capsize=2.5)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([g.replace(" (deployed)", "\n(deployed)").replace(" (new)", "\n(new)")
                         for g in order["gate"]], fontsize=7)
    ax2.set_xlabel("AURC [cycles], lower is better")
    ax2.set_title(f"AURC over {AURC_RANGE[0]:.0%}–{AURC_RANGE[1]:.0%} retention\n"
                  f"(95% CI, {N_BOOTSTRAP} fold bootstraps)", pad=4, fontsize=8.5)
    ax2.grid(axis="y", alpha=0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"in_study_risk_coverage.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _report(m, aurc_df, sp, checks, deployed, q_curve, dists) -> None:
    L, a = [], None
    a = L.append
    ok = checks["pass"].all()
    a("# Task 2 — quantile-referenced abstention, in-study (Severson)")
    a("")
    a("The gate's rule changes from `coverage >= 4.08` to `coverage >= the q-th "
      "percentile of the bank's own leave-one-out coverage distribution`. Same "
      "66-fold leave-one-policy-group-out protocol, same predictions, same seed, "
      "same pinned publication edge set — only the abstention rule is swapped. "
      "Each fold's percentile comes from that fold's training bank alone.")
    a("")
    a(f"## (a) In-study results preserved: **{'YES' if ok else 'NO'}**")
    a("")
    a("| check | this run | published | tol | pass |")
    a("|---|---|---|---|---|")
    for _, r in checks.iterrows():
        name = str(r["check"]).replace("|", "\\|")
        want = (f"{r['published']:.4g}" if isinstance(r["published"], (int, float))
                else r["published"])
        a(f"| {name} | {r['value']} | {want} | {r['tol']} | "
          f"{'✅' if r['pass'] else '❌'} |")
    a("")
    if not ok:
        a("> **A published number moved.** Inspect before using this gate.")
        a("")
    a("| gate | AURC (cyc) | 95% CI |")
    a("|---|---|---|")
    for _, r in aurc_df.iterrows():
        a(f"| {r['gate']} | {r['aurc']:.1f} | [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] |")
    a("")
    a("| Spearman(signal, \\|error\\|) | rho | p |")
    a("|---|---|---|")
    for name, s in sp.items():
        a(f"| {name} | {s.statistic:.3f} | {s.pvalue:.3g} |")
    a("")
    a("Raw coverage values are untouched by the swap, so the published Spearman "
      "figures are reproduced exactly. The within-fold percentile rank is the "
      "quantile gate's own ordering; it is reported alongside because that is what "
      "the new rule actually sorts on.")
    a("")
    a("## (b) Deployed percentile and how it maps to 4.08")
    a("")
    a(f"- **q = {deployed['q_star']:g}th percentile** of the bank's LOO coverage "
      f"distribution, selected in-study only (no HUST information).")
    a(f"- Retention {deployed['achieved_retention']:.1%} "
      f"({deployed['n_retained']} of 120 cells), retained RMSE "
      f"{deployed['rmse_retained_cycles']:.1f} cycles — the paper's operating point.")
    a(f"- Across the 66 folds this percentile lands at absolute coverage "
      f"**{deployed['median_fold_threshold_at_q_star']:.3f}** (median; range "
      f"{deployed['min_fold_threshold']:.3f}–{deployed['max_fold_threshold']:.3f}), "
      f"against the old fixed **{deployed['old_absolute_threshold']:.3f}**.")
    a("")
    a("In-study the two rules are therefore near-indistinguishable — which is the "
      "point. The folds' training banks are 118 of the same 120 Severson cells, so "
      "their LOO coverage distributions barely differ and a fixed percentile maps to "
      "a near-fixed absolute value. The rules only diverge when the bank itself "
      "changes character, which is exactly the cross-study case.")
    a("")
    a("### Percentile → absolute threshold → retention")
    a("")
    a("| q (pct) | median fold threshold | retention | retained RMSE (cyc) |")
    a("|---|---|---|---|")
    for qq in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90):
        r = q_curve[q_curve["q"] == qq]
        if len(r):
            r = r.iloc[0]
            a(f"| {int(qq)} | {r['median_fold_threshold']:.3f} | "
              f"{r['frac_retained']:.1%} | {r['rmse_retained']:.1f} |")
    a("")
    loo_all = pd.concat(list(dists.values()))
    a(f"Bank LOO coverage across all folds: min {loo_all.min():.2f}, median "
      f"{loo_all.median():.2f}, max {loo_all.max():.2f} (n={len(loo_all)} "
      "cell-fold pairs).")
    a("")
    a("## Files")
    a("")
    a("| file | contents |")
    a("|---|---|")
    a("| `in_study_absolute_sweep.csv` | risk–coverage for the absolute gate (+ random) |")
    a("| `in_study_quantile_sweep.csv` | risk–coverage for the quantile gate, by q |")
    a("| `in_study_aurc.csv` | AURC + bootstrap CIs, all four gates |")
    a("| `in_study_fold_thresholds.csv` | per-fold threshold at q\\* and LOO summary |")
    a("| `in_study_preservation_checks.csv` | machine-checkable headline preservation |")
    a("| `deployed_gate.json` | the setting to deploy |")
    a("| `in_study_risk_coverage.pdf/.png` | figure |")
    a("")
    a("Reproduce: `python -m experiments.exp06_quantile_gate.in_study`")
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print(f"[exp06] report -> {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
