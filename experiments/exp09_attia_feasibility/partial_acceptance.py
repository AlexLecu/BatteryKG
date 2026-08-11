"""The reviewer's partial-acceptance scenario, on Attia et al. 2020.

The first external population where the deployed gate retains a non-empty
subset zero-shot (17 of 45). This evaluates what it actually bought.

Deployed configuration applied UNCHANGED: the Severson-trained point and
quantile models, the Severson-only neighbour bank, the stratified percentile
gate at q*=41 falling back to the whole-bank percentile (no Attia cell is in the
bank). Nothing is retrained, nothing is added to the bank.

FEATURE COMPLETENESS — read the error figures with this in mind. The deployed
model takes 19 features. For an Attia cell, 8 of them are undefined:

  c_rate_1, c_rate_2, soc_transition_pct   the bank's policy grammar describes a
                                           TWO-step fast charge; Attia's CLO
                                           protocols are four-step CC (e.g.
                                           "4.8-5.2-5.2-4.16"), which that
                                           grammar cannot express
  batch                                    Severson-specific covariate
  condition_* (4 graph features)           follow from the policy features, so
                                           the condition similarity view is
                                           undefined for these cells

The 11 that ARE defined are the ones that matter most: all 7 early-cycle
features and all 4 behaviour-view graph features. XGBoost consumes the missing
ones as NaN natively, exactly as the deployed path does for HUST's `batch`. But
a model asked to predict with 8/19 inputs absent is working harder than it does
in-study, and any error figure below inherits that.

Run:  python -m experiments.exp09_attia_feasibility.partial_acceptance
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from app.common import load_artifacts
from experiments.exp09_attia_feasibility.attia import (OUT, Q_STAR, behavior_coverage,
                                                       build_cycles, build_features)
from src.config import ROOT
from src.ingestion.qc import cycle_life_table
from src.models.common import make_model
from src.models.dataset import TARGET

TABLE = ROOT / "paper" / "tables" / "tab_attia_partial.tex"
APE_CUTOFF_DEFAULT = 30.0
NOMINAL_BAND = 0.68            # q16..q84
CONDITION_FEATURES = ["condition_nbr_wmean_log_life", "condition_nbr_wstd_log_life",
                      "condition_nbr_mean_weight", "condition_coverage_train"]


def _rmse(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def _mape(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - t) / t) * 100) if len(t) else float("nan")


def _ape(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return np.abs(p - t) / t * 100


def behavior_graph_features(meta: dict, bank: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """The four behaviour-view graph features, served-path maths."""
    from app.common import _zscore_with
    sc = meta["scaler"]["behavior"]
    cols, k = sc["columns"], meta["k_neighbors"]
    B = _zscore_with(bank[cols].to_numpy(float), sc["mean"], sc["std"])
    Q = _zscore_with(q[cols].to_numpy(float), sc["mean"], sc["std"])
    W = 1.0 / (1.0 + np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2))
    W = np.where(q["cell_id"].to_numpy()[:, None] == bank["cell_id"].to_numpy()[None, :],
                 -np.inf, W)
    idx = np.argsort(-W, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(W, idx, axis=1)
    y = bank[TARGET].to_numpy(float)[idx]
    wsum = w.sum(1)
    wmean = (w * y).sum(1) / wsum
    wstd = np.sqrt((w * (y - wmean[:, None]) ** 2).sum(1) / wsum)
    return pd.DataFrame({"behavior_nbr_wmean_log_life": wmean,
                         "behavior_nbr_wstd_log_life": wstd,
                         "behavior_nbr_mean_weight": w.mean(1),
                         "behavior_coverage_train": wsum}, index=q.index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ape-cutoff", type=float, default=APE_CUTOFF_DEFAULT)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta, models, bank = art["meta"], art["models"], art["bank"].copy()

    cycles = build_cycles()
    life = cycle_life_table(cycles)[["cell_id", "cycle_life_nominal", "reached_eol_nominal"]]
    feats, skipped, tolerant = build_features()
    d = feats.merge(life, on="cell_id", how="inner").reset_index(drop=True)
    actual = d["cycle_life_nominal"].to_numpy(float)

    # --- deployed serving path, unchanged --------------------------------------
    gf = behavior_graph_features(meta, bank, d)
    X = pd.concat([d.reindex(columns=meta["base_features"]), gf], axis=1)
    for c in CONDITION_FEATURES:                 # undefined for Attia (see docstring)
        X[c] = np.nan
    X = X[meta["features"]]
    n_missing = int(X.isna().all(axis=0).sum())

    pred = 10.0 ** models["point"].predict(X)
    band_lo = 10.0 ** models["q16"].predict(X)
    band_hi = 10.0 ** models["q84"].predict(X)
    lo, hi = np.minimum(band_lo, band_hi), np.maximum(band_lo, band_hi)

    cov = behavior_coverage(meta, bank, d, "cell_id")
    bank_loo = behavior_coverage(meta, bank, bank, "cell_id", exclude_own_group=True)
    thr = float(np.percentile(bank_loo, Q_STAR))
    keep = cov >= thr
    drop = ~keep
    ape = _ape(actual, pred)

    # --- graph-free GBM baseline, same seed/config, trained on the bank --------
    base = make_model()
    base.fit(bank[meta["base_features"]], bank[TARGET])
    pred_base = 10.0 ** base.predict(d.reindex(columns=meta["base_features"]))

    # --- calibration of the nominal q16-q84 band ------------------------------
    inside = (actual >= lo) & (actual <= hi)

    def calib(mask):
        if mask.sum() == 0:
            return {"n": 0}
        return {"n": int(mask.sum()),
                "empirical_coverage": float(inside[mask].mean()),
                "nominal_coverage": NOMINAL_BAND,
                "median_band_width_cycles": float(np.median((hi - lo)[mask])),
                # direction of the miss, named for what the MODEL did
                "actual_above_band_model_underpredicted":
                    int((actual[mask] > hi[mask]).sum()),
                "actual_below_band_model_overpredicted":
                    int((actual[mask] < lo[mask]).sum())}

    # --- correlations ----------------------------------------------------------
    sp_ret = (stats.spearmanr(cov[keep], np.abs(pred - actual)[keep])
              if keep.sum() >= 10 else None)
    sp_all = stats.spearmanr(cov, np.abs(pred - actual))
    proxy = stats.spearmanr(cov, actual)          # mirrors experiment 01's confound test

    res = {
        "config": {"model": "deployed Severson-trained artifacts, unchanged",
                   "bank_cells": int(len(bank)), "attia_added_to_bank": 0,
                   "models_retrained": 0, "q_star": Q_STAR,
                   "gate_rule": "study-stratified percentile, whole-bank fallback",
                   "threshold": thr, "ape_cutoff_pct": args.ape_cutoff},
        "feature_completeness": {
            "n_features": len(meta["features"]),
            "fully_missing_for_attia": n_missing,
            "missing": sorted(X.columns[X.isna().all(axis=0)].tolist()),
            "note": "policy grammar cannot express Attia's four-step CC protocols; "
                    "the condition similarity view follows from those features"},
        "population": {"n": int(len(d)), "reached_eol": int(d["reached_eol_nominal"].sum()),
                       "cycle_life": {"min": float(actual.min()),
                                      "median": float(np.median(actual)),
                                      "max": float(actual.max())}},
        "gate": {"retained": int(keep.sum()), "abstained": int(drop.sum()),
                 "retention_pct": round(100 * float(keep.mean()), 1)},
        "retained": {"n": int(keep.sum()), "rmse": _rmse(actual[keep], pred[keep]),
                     "mape": _mape(actual[keep], pred[keep]),
                     "false_acceptance": int((ape[keep] > args.ape_cutoff).sum()),
                     "worst_ape": float(ape[keep].max()) if keep.any() else float("nan")},
        "abstained": {"n": int(drop.sum()),
                      "rmse_had_it_answered": _rmse(actual[drop], pred[drop]),
                      "mape_had_it_answered": _mape(actual[drop], pred[drop]),
                      "unnecessary_rejection": int((ape[drop] <= args.ape_cutoff).sum())},
        "ungated_graph_model_all": {"n": int(len(d)), "rmse": _rmse(actual, pred),
                                    "mape": _mape(actual, pred),
                                    "would_be_false_acceptance":
                                        int((ape > args.ape_cutoff).sum())},
        "ungated_graph_free_baseline_all": {
            "n": int(len(d)), "rmse": _rmse(actual, pred_base),
            "mape": _mape(actual, pred_base)},
        "calibration_q16_q84": {"retained": calib(keep), "abstained": calib(drop),
                                "all": calib(np.ones(len(d), bool))},
        "correlations": {
            "spearman_coverage_vs_abs_error_retained":
                {"rho": float(sp_ret.statistic), "p": float(sp_ret.pvalue)}
                if sp_ret is not None else None,
            "spearman_coverage_vs_abs_error_all":
                {"rho": float(sp_all.statistic), "p": float(sp_all.pvalue)},
            "lifetime_proxy_coverage_vs_cycle_life":
                {"rho": float(proxy.statistic), "p": float(proxy.pvalue),
                 "in_study_reference_rho": 0.055,
                 "note": "experiment 01 ran this confound check in-study and found "
                         "rho=0.055 (coverage is not a lifetime proxy); this is the "
                         "same test on Attia"},
        },
        "data_quality": {"length_mismatch_handled": tolerant, "feature_skipped": skipped},
    }

    per_cell = d[["cell_id", "cycle_life_nominal"]].copy()
    per_cell["coverage"] = cov
    per_cell["retained"] = keep
    per_cell["prediction"] = pred
    per_cell["ape_pct"] = ape
    per_cell["band_lo"] = lo
    per_cell["band_hi"] = hi
    per_cell["inside_band"] = inside
    per_cell["prediction_graph_free"] = pred_base
    per_cell.sort_values("coverage", ascending=False).to_csv(
        OUT / "attia_partial_acceptance_per_cell.csv", index=False)
    (OUT / "attia_partial_acceptance.json").write_text(json.dumps(res, indent=1, default=str))
    _tex(res)
    _markdown(res, per_cell, args.ape_cutoff)

    r, ab = res["retained"], res["abstained"]
    print(f"retained {r['n']}: RMSE {r['rmse']:.1f}  MAPE {r['mape']:.1f}%  "
          f"FA {r['false_acceptance']}  worst APE {r['worst_ape']:.1f}%")
    print(f"abstained {ab['n']}: RMSE(had it answered) {ab['rmse_had_it_answered']:.1f}  "
          f"MAPE {ab['mape_had_it_answered']:.1f}%  UR {ab['unnecessary_rejection']}")
    u, b = res["ungated_graph_model_all"], res["ungated_graph_free_baseline_all"]
    print(f"ungated graph model (45): RMSE {u['rmse']:.1f}  MAPE {u['mape']:.1f}%")
    print(f"ungated graph-free  (45): RMSE {b['rmse']:.1f}  MAPE {b['mape']:.1f}%")
    c = res["calibration_q16_q84"]["retained"]
    print(f"band calibration on retained: {c['empirical_coverage']:.0%} inside a "
          f"{NOMINAL_BAND:.0%} nominal band (median width "
          f"{c['median_band_width_cycles']:.0f} cycles)")
    co = res["correlations"]
    print(f"Spearman(coverage, |err|) retained: "
          f"{co['spearman_coverage_vs_abs_error_retained']['rho']:+.3f}")
    print(f"lifetime proxy Spearman(coverage, life): "
          f"{co['lifetime_proxy_coverage_vs_cycle_life']['rho']:+.3f} "
          f"(p={co['lifetime_proxy_coverage_vs_cycle_life']['p']:.3g})")
    return res


def _tex(r: dict) -> None:
    g, ret, ab = r["gate"], r["retained"], r["abstained"]
    u, bl = r["ungated_graph_model_all"], r["ungated_graph_free_baseline_all"]
    cut = r["config"]["ape_cutoff_pct"]
    tex = [
        "% Auto-generated by experiments/exp09_attia_feasibility/partial_acceptance.py",
        "% — do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Partial acceptance on a third external study: Attia et al.\ 2020",
        r"(closed-loop optimization; 45 cells of the same A123 APR18650M1A, same",
        r"laboratory and equipment as the training study, 30\,\textdegree C, 4C",
        r"discharge, four-step constant-current fast-charge protocols). The deployed",
        r"configuration is applied unchanged --- Severson-trained model, Severson-only",
        r"neighbor bank, stratified percentile gate at $q^{*}=41$ falling back to the",
        r"whole-bank percentile because no Attia cell is in the bank. Nothing is",
        r"retrained and nothing is added to the bank. Unlike the HUST and SNL",
        r"populations, the 4C discharge is shared with the bank, so the behaviour-view",
        r"features are commensurable and the gate retains a non-empty subset. False",
        f"acceptances are served predictions with APE above {cut:g}\\%; unnecessary",
        r"rejections are abstentions on cells the model would have called within that",
        r"bound.\label{tab:attia-partial}}",
        r"\begin{tabularx}{\textwidth}{lCCCCC}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{n} & \textbf{RMSE (cycles)} & "
        r"\textbf{MAPE (\%)} & \textbf{FA} & \textbf{UR} \\",
        r"\midrule",
        f"Attia zero-shot, gate-retained & {ret['n']} & {ret['rmse']:.1f} & "
        f"{ret['mape']:.1f} & {ret['false_acceptance']} & --- \\\\",
        f"Attia zero-shot, gate-abstained & {ab['n']} & {ab['rmse_had_it_answered']:.1f} & "
        f"{ab['mape_had_it_answered']:.1f} & --- & {ab['unnecessary_rejection']} \\\\",
        r"\midrule",
        f"Attia zero-shot, graph model (ungated) & {u['n']} & {u['rmse']:.1f} & "
        f"{u['mape']:.1f} & {u['would_be_false_acceptance']} & --- \\\\",
        f"Attia zero-shot, graph-free GBM (ungated) & {bl['n']} & {bl['rmse']:.1f} & "
        f"{bl['mape']:.1f} & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(tex) + "\n")
    print(f"[tex] -> {TABLE}")


def _markdown(r: dict, per_cell: pd.DataFrame, cut: float) -> None:
    ret, ab = r["retained"], r["abstained"]
    u, bl = r["ungated_graph_model_all"], r["ungated_graph_free_baseline_all"]
    cal = r["calibration_q16_q84"]
    co = r["correlations"]
    L, a = [], None
    a = L.append
    a("# Attia 2020 — partial acceptance under the deployed gate")
    a("")
    a(f"The first external population the gate does not refuse outright: "
      f"**{ret['n']} of {r['population']['n']} cells retained** "
      f"({r['gate']['retention_pct']}%) at threshold {r['config']['threshold']:.4f}. "
      "Deployed artifacts applied unchanged; nothing retrained, nothing added to the "
      "bank.")
    a("")
    a("| configuration | n | RMSE (cyc) | MAPE (%) | FA | UR |")
    a("|---|---|---|---|---|---|")
    a(f"| gate-retained | {ret['n']} | {ret['rmse']:.1f} | {ret['mape']:.1f} | "
      f"{ret['false_acceptance']} | — |")
    a(f"| gate-abstained (had it answered) | {ab['n']} | {ab['rmse_had_it_answered']:.1f} | "
      f"{ab['mape_had_it_answered']:.1f} | — | {ab['unnecessary_rejection']} |")
    a(f"| graph model, ungated | {u['n']} | {u['rmse']:.1f} | {u['mape']:.1f} | "
      f"{u['would_be_false_acceptance']} | — |")
    a(f"| graph-free GBM, ungated | {bl['n']} | {bl['rmse']:.1f} | {bl['mape']:.1f} | — | — |")
    a("")
    a(f"FA = served with APE > {cut:g}%. UR = abstained though APE ≤ {cut:g}%.")
    a("")
    a("## Does the gate select the cells it should?")
    a("")
    rmse_better = ret["rmse"] < ab["rmse_had_it_answered"]
    mape_better = ret["mape"] < ab["mape_had_it_answered"]
    a(f"Retained RMSE {ret['rmse']:.1f} vs abstained {ab['rmse_had_it_answered']:.1f} "
      f"(ungated {u['rmse']:.1f}); retained MAPE {ret['mape']:.1f}% vs abstained "
      f"{ab['mape_had_it_answered']:.1f}%.")
    a("")
    if rmse_better and not mape_better:
        a("**The two metrics disagree, and the disagreement is the finding.** The gate "
          "retains cells with lower absolute error but *higher* percentage error. That "
          "is what happens when the retained set is systematically shorter-lived: the "
          "same absolute error is a larger fraction of a smaller lifetime. Read with "
          "the lifetime-proxy result below, the RMSE advantage is substantially a "
          "lifetime effect rather than evidence that the gate found the predictable "
          "cells.")
    elif rmse_better:
        a("The gate separates on both metrics: the cells it serves carry lower error "
          "than the ones it refuses.")
    else:
        a("**The gate does not separate**: the cells it serves carry higher error than "
          "the ones it refuses.")
    a("")
    sr = co["spearman_coverage_vs_abs_error_retained"]
    if sr:
        a(f"Spearman(coverage, |error|) on the retained set: **ρ = {sr['rho']:+.3f}** "
          f"(p = {sr['p']:.3g}). In-study the same correlation is −0.247: more coverage, "
          "less error.")
    sa = co["spearman_coverage_vs_abs_error_all"]
    a(f"Across all 45: ρ = {sa['rho']:+.3f} (p = {sa['p']:.3g}).")
    a("")
    a("### Lifetime-proxy check")
    a("")
    lp = co["lifetime_proxy_coverage_vs_cycle_life"]
    a(f"Spearman(coverage, cycle life) = **{lp['rho']:+.3f}** (p = {lp['p']:.3g}), "
      f"against {lp['in_study_reference_rho']:+.3f} in-study (experiment 01). "
      + ("Coverage is behaving as a lifetime proxy on this population, which the "
         "in-study test explicitly ruled out for Severson — the retained subset is "
         "partly being selected for how long the cells live, not only for how "
         "predictable they are."
         if abs(lp["rho"]) > 0.3 else
         "Consistent with the in-study finding: coverage is not acting as a lifetime "
         "proxy here either."))
    a("")
    a("## Calibration of the q16–q84 band")
    a("")
    a("| set | n | empirical coverage | nominal | median width (cyc) | model over-predicted | model under-predicted |")
    a("|---|---|---|---|---|---|---|")
    for k in ("retained", "abstained", "all"):
        c = cal[k]
        if c.get("n"):
            a(f"| {k} | {c['n']} | {c['empirical_coverage']:.0%} | "
              f"{c['nominal_coverage']:.0%} | {c['median_band_width_cycles']:.0f} | "
              f"{c['actual_below_band_model_overpredicted']} | "
              f"{c['actual_above_band_model_underpredicted']} |")
    a("")
    allc = cal["all"]
    over, under = (allc["actual_below_band_model_overpredicted"],
                   allc["actual_above_band_model_underpredicted"])
    a(f"The band is badly miscalibrated out of distribution — "
      f"{allc['empirical_coverage']:.0%} empirical against a "
      f"{allc['nominal_coverage']:.0%} nominal band — and the misses are "
      f"**one-sided**: {over} of {allc['n']} cells fall below the band and {under} "
      "above it. The model does not merely have wide uncertainty here, it "
      "systematically predicts these cells to live longer than they do, and its "
      "own uncertainty band does not admit the possibility.")
    a("")
    a("## Feature completeness caveat")
    a("")
    fc = r["feature_completeness"]
    a(f"{fc['fully_missing_for_attia']} of {fc['n_features']} model inputs are "
      f"undefined for every Attia cell: `{'`, `'.join(fc['missing'])}`. "
      f"{fc['note']}. The 11 that remain are all 7 early-cycle features and all 4 "
      "behaviour-view graph features — the informative ones — but the error figures "
      "above are produced with 8 inputs absent, which is not the condition the "
      "in-study numbers were measured under.")
    a("")
    a("## Artifacts")
    a("")
    a("| file | contents |")
    a("|---|---|")
    a("| `attia_partial_acceptance.json` | full result record |")
    a("| `attia_partial_acceptance_per_cell.csv` | per-cell coverage, prediction, APE, band |")
    a("| `paper/tables/tab_attia_partial.tex` | paper table |")
    (OUT / "PARTIAL_ACCEPTANCE.md").write_text("\n".join(L) + "\n")
    print(f"[md]  -> {OUT / 'PARTIAL_ACCEPTANCE.md'}")


if __name__ == "__main__":
    main()
