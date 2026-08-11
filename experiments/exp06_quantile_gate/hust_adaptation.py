"""Task 3 — the HUST small-sample adaptation, re-run under the quantile gate.

Same design as experiment 05 (n in {0,5,10,20,40} labelled HUST cells added to
the bank, 5 seeded repeats, bank-only and bank+retrain, evaluated on the
remaining 77-n cells) with one change: the gate is no longer `coverage >= 4.08`
but `coverage >= the q*-th percentile of the CURRENT bank's leave-one-out
coverage distribution`. q* = 41 was fixed in-study (Task 2) and is not re-tuned
here.

Because the reference distribution is re-derived from whatever bank it is
given, it moves as HUST cells enter — which is the entire point.

Reference-distribution variants (both reported):
  same_policy_excluded  each bank cell's coverage against the rest of the bank
                        with its own policy group removed. This is the Task-2
                        definition, and it mirrors the leave-one-policy-group-out
                        condition the in-study threshold was derived under.
  self_only             only the cell itself removed.
They differ sharply here for a structural reason worth stating plainly: all 77
HUST cells share one charge policy (5C(80%)-1C), so "exclude my own policy
group" means "exclude the entire HUST study" for a HUST bank cell, while the
held-out HUST cells being judged DO get to use HUST neighbours. The primary
number is the Task-2 definition; the sensitivity is reported beside it.

Run:  python -m experiments.exp06_quantile_gate.hust_adaptation
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.common import load_artifacts
from experiments.exp05_hust_adaptation.core import (
    GraphIndex, ape, augmented_bank, design_matrix, hust_table, mape, rmse, severson_bank)
from experiments.exp05_hust_adaptation.run import (
    APE_CUTOFF_DEFAULT, MIN_N_FOR_SPEARMAN, N_GRID, N_REPEATS)
from src.config import OUTPUTS
from src.models.common import SEED, make_model
from src.models.dataset import TARGET

OUT = OUTPUTS / "experiment_06_quantile_gate"
DEPLOYED_GATE = OUT / "deployed_gate.json"
REF_VARIANTS = ("same_policy_excluded", "self_only", "study_stratified")
MIN_REFERENCE_CELLS = 5      # below this a same-study percentile is meaningless


def bank_loo_coverage(idx: GraphIndex, bank: pd.DataFrame, variant: str) -> np.ndarray:
    """The bank's own leave-one-out behaviour coverage distribution."""
    eg = np.ones(len(bank), bool) if variant == "same_policy_excluded" else None
    gf = idx.graph_features(bank, exclude_self=True, exclude_own_group=eg)
    return gf["behavior_coverage_train"].to_numpy(float)


def reference_threshold(idx: GraphIndex, bank: pd.DataFrame, variant: str,
                        q_star: float, query_study: str,
                        held_out_scope: str = "none") -> tuple[float, str]:
    """Gate threshold from the bank, under one reference definition.

    `study_stratified` is the coherent rule, and it has two parts:

    1. *Population* — judge a cell against bank members of ITS OWN study. A
       global percentile over a Severson-dominated bank keeps answering a
       question about Severson no matter how much HUST data arrives, which is
       why the plain quantile swap does not restore utility.
    2. *Scoring* — score those reference cells under the same neighbour
       availability the query itself faces, given by `held_out_scope`:
         "policy_group"  the query's whole policy group is held out (in-study
                         leave-one-policy-group-out CV) -> reference excludes
                         same-policy edges. With an all-Severson bank this is
                         exactly the Task-2 definition, so q* transfers.
         "none"          nothing is held out; the deployed cell sees the whole
                         bank -> reference excludes only self.
       Getting this wrong biases the gate: scoring the reference with less
       information than the query has makes the gate too permissive, and with
       more information makes it too strict.

    Falls back to the whole bank when the query's study has too few members to
    form a percentile — which is what keeps the zero-shot case abstaining.
    """
    if variant != "study_stratified":
        return float(np.percentile(bank_loo_coverage(idx, bank, variant), q_star)), variant
    scope_rule = {"policy_group": "same_policy_excluded", "none": "self_only"}[held_out_scope]
    same = (bank["study"] == query_study).to_numpy()
    if same.sum() < MIN_REFERENCE_CELLS:
        return (float(np.percentile(bank_loo_coverage(idx, bank, "same_policy_excluded"),
                                    q_star)), "fallback:whole_bank")
    loo = bank_loo_coverage(idx, bank, scope_rule)[same]
    return float(np.percentile(loo, q_star)), f"same_study/{scope_rule}"


def run_draw(meta, models, sev, hust, n, seed, q_star, ape_cutoff) -> list[dict]:
    rng = np.random.default_rng(seed)
    add_ids = ([] if n == 0 else
               sorted(rng.choice(hust["cell_id"].to_numpy(), size=n, replace=False).tolist()))
    bank = augmented_bank(sev, hust, add_ids)
    idx = GraphIndex(meta, bank)

    ev = hust[~hust["cell_id"].isin(add_ids)].reset_index(drop=True)
    gf_eval = idx.graph_features(ev)              # eval cells are not in the bank
    X_eval = design_matrix(meta, ev, gf_eval)
    coverage = gf_eval["behavior_coverage_train"].to_numpy()
    actual = ev["cycle_life_nominal"].to_numpy(float)

    # gate thresholds, re-derived from THIS bank
    # deployed HUST cells are judged against a bank that holds nothing back
    thresholds = {v: reference_threshold(idx, bank, v, q_star, "hust",
                                         held_out_scope="none")
                  for v in REF_VARIANTS}

    # predictors
    is_sev = (bank["study"] == "severson").to_numpy()
    gf_train = idx.graph_features(bank, exclude_self=True, exclude_own_group=is_sev)
    retrained = make_model()
    retrained.fit(design_matrix(meta, bank, gf_train), bank[TARGET].to_numpy(float))
    preds = {"bank_only": 10.0 ** models["point"].predict(X_eval),
             "bank_retrain": 10.0 ** retrained.predict(X_eval)}

    rows = []
    for ref in REF_VARIANTS:
        thr, ref_source = thresholds[ref]
        keep = coverage >= thr
        drop = ~keep
        for variant, pred in preds.items():
            a = ape(actual, pred)
            sp = float("nan")
            if keep.sum() >= MIN_N_FOR_SPEARMAN:
                sp = float(spearmanr(coverage[keep], np.abs(pred - actual)[keep]).statistic)
            rows.append({
                "n_added": n, "variant": variant, "ref_distribution": ref, "seed": seed,
                "q_star": q_star, "gate_threshold": thr, "reference_source": ref_source,
                "n_eval": len(ev), "n_retained": int(keep.sum()),
                "retention_pct": 100.0 * keep.mean(),
                "rmse_retained": rmse(actual[keep], pred[keep]),
                "mape_retained": mape(actual[keep], pred[keep]),
                "rmse_ungated_all": rmse(actual, pred),
                "mape_ungated_all": mape(actual, pred),
                "rmse_abstained": rmse(actual[drop], pred[drop]),
                "mape_abstained": mape(actual[drop], pred[drop]),
                "false_acceptance": int((keep & (a > ape_cutoff)).sum()),
                "unnecessary_rejection": int((drop & (a <= ape_cutoff)).sum()),
                "spearman_cov_vs_abserr": sp,
                "cov_min": float(coverage.min()), "cov_median": float(np.median(coverage)),
                "cov_max": float(coverage.max()), "cov_mean": float(coverage.mean()),
                "added_cells": ";".join(add_ids),
            })
    return rows


AGG_COLS = ["retention_pct", "n_retained", "gate_threshold", "rmse_retained",
            "mape_retained", "rmse_ungated_all", "mape_ungated_all", "rmse_abstained",
            "false_acceptance", "unnecessary_rejection", "spearman_cov_vs_abserr",
            "cov_min", "cov_median", "cov_max"]


def aggregate(per_repeat: pd.DataFrame) -> pd.DataFrame:
    g = per_repeat.groupby(["n_added", "ref_distribution", "variant"], sort=False)
    out = g[AGG_COLS].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out = out.reset_index()
    out["n_eval"] = g["n_eval"].first().to_numpy()
    out["n_repeats"] = g.size().to_numpy()
    return out


def figure(agg: pd.DataFrame, q_star: float) -> None:
    import matplotlib.pyplot as plt
    from src.viz.paper_style import BLUE, GREEN, INK_MUT, ORANGE, W_FULL, apply_style
    apply_style()
    primary = agg[agg.ref_distribution == "study_stratified"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_FULL, 2.9))

    ret = primary[primary.variant == "bank_retrain"]     # gate is bank-determined
    ax1.errorbar(ret["n_added"], ret["retention_pct_mean"], yerr=ret["retention_pct_std"],
                 marker="s", ms=4.5, lw=2.1, color=GREEN, capsize=2.5)
    ax1.set_xlabel("labelled HUST cells in the bank ($n$)")
    ax1.set_ylabel("retention on held-out cells [%]")
    ax1.set_title("Retention recovers", pad=4, fontsize=8.5)
    ax1.set_ylim(-4, 104)
    ax1.set_xticks(N_GRID)
    ax1.annotate("identical for both variants:\nthe gate reads the bank,\nnot the predictor",
                 xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                 fontsize=6.8, color=INK_MUT)

    for variant, color, mk, lw, z, alpha in [("bank_retrain", GREEN, "s", 2.1, 3, 1.0),
                                             ("bank_only", ORANGE, "o", 1.3, 2, 0.55)]:
        s = primary[primary.variant == variant]
        msk = s["mape_retained_mean"].notna()
        ax2.errorbar(s.loc[msk, "n_added"], s.loc[msk, "mape_retained_mean"],
                     yerr=s.loc[msk, "mape_retained_std"], marker=mk, ms=4.5, lw=lw,
                     color=color, capsize=2.5, zorder=z, alpha=alpha,
                     label=variant.replace("_", "-") + (" (deployed)" if variant ==
                                                        "bank_retrain" else ""))
    ung = primary[(primary.variant == "bank_only") & (primary.n_added == 0)][
        "mape_ungated_all_mean"].iloc[0]
    ax2.axhline(ung, ls=":", lw=1.2, color=INK_MUT,
                label=f"zero-shot ungated ({ung:.0f}%)")
    ax2.axhline(30, ls="--", lw=1.0, color=BLUE, label="30% APE bar")
    ax2.set_xlabel("labelled HUST cells in the bank ($n$)")
    ax2.set_ylabel("MAPE on retained cells [%]")
    ax2.set_title("Accuracy on what is served", pad=4, fontsize=8.5)
    ax2.set_xticks(N_GRID)
    ax2.set_yscale("log")
    ax2.set_ylim(7, 130)
    ax2.legend(loc="lower left", fontsize=6.8)

    fig.suptitle(f"HUST adaptation under the study-stratified quantile gate "
                 f"($q^*={q_star:g}$th percentile, fixed in-study, never re-tuned)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"hust_recovery_quantile.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ape-cutoff", type=float, default=APE_CUTOFF_DEFAULT)
    p.add_argument("--repeats", type=int, default=N_REPEATS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not DEPLOYED_GATE.exists():
        raise SystemExit("run experiments.exp06_quantile_gate.in_study first "
                         "(it selects q* in-study)")
    q_star = float(json.loads(DEPLOYED_GATE.read_text())["q_star"])

    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta, models = art["meta"], art["models"]
    sev, hust = severson_bank(), hust_table()

    rows = []
    for n in N_GRID:
        for rep in range(args.repeats):
            rows.extend(run_draw(meta, models, sev, hust, n, SEED + rep,
                                 q_star, args.ape_cutoff))
        print(f"[exp06-hust] n={n} done")
    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(OUT / "hust_results_per_repeat.csv", index=False)
    agg = aggregate(per_repeat)
    agg.to_csv(OUT / "hust_results_aggregated.csv", index=False)
    figure(agg, q_star)

    pd.set_option("display.width", 250)
    show = ["n_added", "ref_distribution", "variant", "gate_threshold_mean",
            "retention_pct_mean", "mape_retained_mean", "false_acceptance_mean",
            "unnecessary_rejection_mean", "mape_ungated_all_mean"]
    print("\n=== HUST adaptation under the quantile gate ===")
    print(agg[show].round(2).to_string(index=False))
    return per_repeat, agg, q_star


if __name__ == "__main__":
    main()
