"""Minimal adaptation on Attia: n=5 labelled cells into the bank, then re-evaluate.

Mirrors the HUST design (experiments/exp06_quantile_gate/hust_adaptation.py) at a
single point: n=5, bank+retrain only, 5 seeded repeats, the stratified percentile
gate at q*=41 unchanged, evaluated on the remaining 40 cells.

Two departures from the HUST module, both forced by the data:

1. BEHAVIOUR-VIEW ONLY. Attia's four-step CC protocols cannot be expressed in the
   bank's two-step policy grammar, so its condition features are null and the
   condition similarity view is undefined for these cells. Condition-view graph
   features are computed for Severson rows (against Severson bank rows) and left
   NaN for Attia rows, which is what the data supports. HUST had parseable policy
   features and did not need this.

2. QUANTILE MODELS ARE RETRAINED TOO. The zero-shot run found the q16-q84 band
   covering 22% of cells against a 68% nominal, one-sided (35 of 45 below the
   band). Re-checking that after adaptation requires refitting q16/q84 alongside
   the point model, which the HUST experiment did not need to do.

Retention and the gate depend only on the bank, so the two variants of the HUST
study would coincide here; only bank+retrain is run, as specified.

Run:  python -m experiments.exp09_attia_feasibility.adaptation
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

from app.common import _zscore_with, load_artifacts
from experiments.exp09_attia_feasibility.attia import OUT, Q_STAR, build_cycles, build_features
from experiments.exp09_attia_feasibility.partial_acceptance import (CONDITION_FEATURES,
                                                                    NOMINAL_BAND, _ape,
                                                                    _mape, _rmse)
from src.config import ROOT
from src.ingestion.qc import cycle_life_table
from src.models.common import SEED, XGB_PARAMS, make_model
from src.models.dataset import TARGET

TABLE = ROOT / "paper" / "tables" / "tab_attia_adaptation.tex"
REPORT = OUT / "PARTIAL_ACCEPTANCE.md"
MARKER = "\n## Minimal adaptation (n=5)"
N_ADD = 5
N_REPEATS = 5
APE_CUTOFF = 30.0
MIN_REFERENCE_CELLS = 5          # same rule as exp06: below this, fall back
ATTIA_GROUP = "pgATT00"          # one cross-study policy group, as HUST uses pgH00


def _view_features(meta: dict, view: str) -> list[str]:
    return meta["scaler"][view]["columns"]


def graph_features(meta: dict, bank: pd.DataFrame, q: pd.DataFrame, view: str,
                   exclude_self: bool = False,
                   exclude_own_group: np.ndarray | None = None) -> pd.DataFrame:
    """Served-path graph features for one view.

    Bank rows and query rows with incomplete features for the view are excluded /
    returned as NaN respectively — the honest handling for a study whose policy
    features the bank's grammar cannot express.
    """
    cols, k = _view_features(meta, view), meta["k_neighbors"]
    sc = meta["scaler"][view]
    ok_bank = bank[cols].notna().all(axis=1).to_numpy()
    B = _zscore_with(bank.loc[ok_bank, cols].to_numpy(float), sc["mean"], sc["std"])
    yb = bank.loc[ok_bank, TARGET].to_numpy(float)
    ids_b = bank.loc[ok_bank, "cell_id"].to_numpy()
    grp_b = bank.loc[ok_bank, "policy_group_id"].to_numpy()

    out = pd.DataFrame(index=q.index, columns=[
        f"{view}_nbr_wmean_log_life", f"{view}_nbr_wstd_log_life",
        f"{view}_nbr_mean_weight", f"{view}_coverage_train"], dtype=float)
    ok_q = q[cols].notna().all(axis=1).to_numpy()
    if not ok_q.any() or B.shape[0] < k:
        return out

    Q = _zscore_with(q.loc[ok_q, cols].to_numpy(float), sc["mean"], sc["std"])
    W = 1.0 / (1.0 + np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2))
    if exclude_self:
        W = np.where(q.loc[ok_q, "cell_id"].to_numpy()[:, None] == ids_b[None, :], -np.inf, W)
    if exclude_own_group is not None:
        eg = np.asarray(exclude_own_group, bool)[ok_q]
        W = np.where(eg[:, None] & (q.loc[ok_q, "policy_group_id"].to_numpy()[:, None]
                                    == grp_b[None, :]), -np.inf, W)
    idx = np.argsort(-W, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(W, idx, axis=1)
    y = yb[idx]
    wsum = w.sum(1)
    wmean = (w * y).sum(1) / wsum
    wstd = np.sqrt((w * (y - wmean[:, None]) ** 2).sum(1) / wsum)
    out.loc[ok_q, :] = np.column_stack([wmean, wstd, w.mean(1), wsum])
    return out


def attia_bank_rows(d: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Attia cells in neighbour-bank schema (policy features null by construction)."""
    r = d.copy()
    r["policy_group_id"] = ATTIA_GROUP
    r["charge_policy_norm"] = r.get("charge_policy_norm", pd.NA)
    r["batch"] = np.nan
    for c in ("c_rate_1", "c_rate_2", "soc_transition_pct"):
        r[c] = np.nan
    r[TARGET] = np.log10(r["cycle_life_nominal"].astype(float))
    r["study"] = "attia"
    return r


def stratified_threshold(meta: dict, bank: pd.DataFrame, query_study: str) -> tuple[float, str]:
    """exp06's stratified percentile rule, behaviour view.

    Same-study reference cells scored self-only (a deployed cell sees the whole
    bank); fall back to the whole bank scored with same-policy edges excluded
    when the study has too few members to form a percentile.
    """
    same = (bank["study"] == query_study).to_numpy()
    if same.sum() >= MIN_REFERENCE_CELLS:
        loo = graph_features(meta, bank, bank, "behavior",
                             exclude_self=True)["behavior_coverage_train"].to_numpy()
        return float(np.percentile(loo[same], Q_STAR)), "same_study/self_only"
    loo = graph_features(meta, bank, bank, "behavior", exclude_self=True,
                         exclude_own_group=np.ones(len(bank), bool)
                         )["behavior_coverage_train"].to_numpy()
    return float(np.percentile(loo, Q_STAR)), "fallback:whole_bank"


def design_matrix(meta: dict, rows: pd.DataFrame, bank: pd.DataFrame,
                  exclude_self: bool, is_severson: np.ndarray | None) -> pd.DataFrame:
    beh = graph_features(meta, bank, rows, "behavior", exclude_self=exclude_self)
    con = graph_features(meta, bank, rows, "condition", exclude_self=exclude_self,
                         exclude_own_group=is_severson)
    X = pd.concat([rows.reindex(columns=meta["base_features"]).reset_index(drop=True),
                   beh.reset_index(drop=True), con.reset_index(drop=True)], axis=1)
    for c in CONDITION_FEATURES:
        if c not in X:
            X[c] = np.nan
    return X[meta["features"]]


def run_draw(meta, sev_bank, attia, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    add = sorted(rng.choice(attia["cell_id"].to_numpy(), size=N_ADD, replace=False).tolist())
    added = attia[attia["cell_id"].isin(add)]
    bank = pd.concat([sev_bank, added[sev_bank.columns.intersection(added.columns)]],
                     ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    ev = attia[~attia["cell_id"].isin(add)].reset_index(drop=True)

    thr, src = stratified_threshold(meta, bank, "attia")
    gf = graph_features(meta, bank, ev, "behavior")
    cov = gf["behavior_coverage_train"].to_numpy()
    keep, drop = cov >= thr, cov < thr
    actual = ev["cycle_life_nominal"].to_numpy(float)

    # retrain point + quantile models on the augmented bank
    is_sev = (bank["study"] == "severson").to_numpy()
    Xtr = design_matrix(meta, bank, bank, exclude_self=True, is_severson=is_sev)
    ytr = bank[TARGET].to_numpy(float)
    point = make_model()
    point.fit(Xtr, ytr)
    quant = {}
    for alpha, name in [(0.16, "q16"), (0.84, "q84")]:
        m = XGBRegressor(**XGB_PARAMS, objective="reg:quantileerror", quantile_alpha=alpha)
        m.fit(Xtr, ytr)
        quant[name] = m

    Xev = design_matrix(meta, ev, bank, exclude_self=False, is_severson=None)
    pred = 10.0 ** point.predict(Xev)
    lo = 10.0 ** quant["q16"].predict(Xev)
    hi = 10.0 ** quant["q84"].predict(Xev)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    inside = (actual >= lo) & (actual <= hi)
    ape = _ape(actual, pred)

    def sp(x, y, mask):
        if mask.sum() < 10:
            return None
        r = stats.spearmanr(x[mask], y[mask])
        return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(mask.sum())}

    return {
        "seed": seed, "added": add, "threshold": thr, "reference_source": src,
        "n_eval": int(len(ev)), "n_retained": int(keep.sum()),
        "retention_pct": 100.0 * float(keep.mean()),
        "rmse_retained": _rmse(actual[keep], pred[keep]),
        "mape_retained": _mape(actual[keep], pred[keep]),
        "rmse_abstained": _rmse(actual[drop], pred[drop]),
        "mape_abstained": _mape(actual[drop], pred[drop]),
        "rmse_ungated": _rmse(actual, pred), "mape_ungated": _mape(actual, pred),
        "false_acceptance": int((keep & (ape > APE_CUTOFF)).sum()),
        "unnecessary_rejection": int((drop & (ape <= APE_CUTOFF)).sum()),
        "band_coverage_retained": float(inside[keep].mean()) if keep.any() else np.nan,
        "band_coverage_all": float(inside.mean()),
        "band_overpredicted_all": int((actual < lo).sum()),
        "band_underpredicted_all": int((actual > hi).sum()),
        "spearman_cov_vs_abserr_retained": sp(cov, np.abs(pred - actual), keep),
        "spearman_cov_vs_life_retained": sp(cov, actual, keep),
        "spearman_cov_vs_life_all": sp(cov, actual, np.ones(len(ev), bool)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ape-cutoff", type=float, default=APE_CUTOFF)
    args = p.parse_args()
    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta = art["meta"]
    sev_bank = art["bank"].copy()
    sev_bank["study"] = "severson"

    cycles = build_cycles()
    life = cycle_life_table(cycles)[["cell_id", "cycle_life_nominal"]]
    feats, _, _ = build_features()
    attia = attia_bank_rows(feats.merge(life, on="cell_id"), meta)

    rows = [run_draw(meta, sev_bank, attia, SEED + i) for i in range(N_REPEATS)]
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "attia_adaptation_per_repeat.csv", index=False)

    def ms(col):
        v = per[col].astype(float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}

    def sp_agg(col):
        vals = [r[col]["rho"] for r in rows if r[col]]
        return ({"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                 "n_repeats": len(vals)} if vals else None)

    agg = {
        "design": {"n_added": N_ADD, "repeats": N_REPEATS,
                   "seeds": [SEED + i for i in range(N_REPEATS)],
                   "variant": "bank+retrain (point + q16/q84)",
                   "q_star": Q_STAR, "gate": "stratified percentile, unchanged",
                   "n_eval": int(per["n_eval"].iloc[0]),
                   "reference_source": sorted(set(per["reference_source"])),
                   "ape_cutoff_pct": args.ape_cutoff},
        "threshold": ms("threshold"),
        "retention_pct": ms("retention_pct"), "n_retained": ms("n_retained"),
        "rmse_retained": ms("rmse_retained"), "mape_retained": ms("mape_retained"),
        "rmse_abstained": ms("rmse_abstained"), "mape_abstained": ms("mape_abstained"),
        "rmse_ungated": ms("rmse_ungated"), "mape_ungated": ms("mape_ungated"),
        "false_acceptance": ms("false_acceptance"),
        "unnecessary_rejection": ms("unnecessary_rejection"),
        "band_coverage_retained": ms("band_coverage_retained"),
        "band_coverage_all": ms("band_coverage_all"),
        "band_overpredicted_all": ms("band_overpredicted_all"),
        "band_underpredicted_all": ms("band_underpredicted_all"),
        "spearman_cov_vs_abserr_retained": sp_agg("spearman_cov_vs_abserr_retained"),
        "spearman_cov_vs_life_retained": sp_agg("spearman_cov_vs_life_retained"),
        "spearman_cov_vs_life_all": sp_agg("spearman_cov_vs_life_all"),
        "zero_shot_reference": {"retention_pct": 37.8, "rmse_retained": 141.7,
                                "mape_retained": 21.5, "false_acceptance": 4,
                                "unnecessary_rejection": 21,
                                "band_coverage_retained": 0.06,
                                "spearman_cov_vs_life_all": -0.643},
    }
    (OUT / "attia_adaptation.json").write_text(json.dumps(agg, indent=1, default=str))
    _tex(agg)
    _append_section(agg)

    print(f"n={N_ADD}, {N_REPEATS} repeats, eval on {agg['design']['n_eval']} cells "
          f"(reference: {agg['design']['reference_source']})")
    print(f"  threshold      {agg['threshold']['mean']:.3f} ± {agg['threshold']['std']:.3f}")
    print(f"  retention      {agg['retention_pct']['mean']:.1f} ± {agg['retention_pct']['std']:.1f} %")
    print(f"  retained RMSE  {agg['rmse_retained']['mean']:.1f} ± {agg['rmse_retained']['std']:.1f}")
    print(f"  retained MAPE  {agg['mape_retained']['mean']:.1f} ± {agg['mape_retained']['std']:.1f} %")
    print(f"  FA / UR        {agg['false_acceptance']['mean']:.1f} / "
          f"{agg['unnecessary_rejection']['mean']:.1f}")
    print(f"  band (retained){agg['band_coverage_retained']['mean']:6.1%} "
          f"(nominal {NOMINAL_BAND:.0%}); over-pred "
          f"{agg['band_overpredicted_all']['mean']:.1f}/{agg['design']['n_eval']}")
    for k in ("spearman_cov_vs_abserr_retained", "spearman_cov_vs_life_retained",
              "spearman_cov_vs_life_all"):
        v = agg[k]
        print(f"  {k:34s} {v['mean']:+.3f} ± {v['std']:.3f}" if v else f"  {k}: n/a")
    return agg


def _tex(a: dict) -> None:
    z = a["zero_shot_reference"]
    def pm(k, f="{:.1f}"):
        return f"${f.format(a[k]['mean'])} \\pm {f.format(a[k]['std'])}$"
    tex = [
        "% Auto-generated by experiments/exp09_attia_feasibility/adaptation.py",
        "% — do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Minimal adaptation on Attia et al.\ 2020: five labelled cells added",
        r"to the neighbor bank at random and the predictor (point and quantile models)",
        r"refitted on Severson $+\,5$, evaluated on the remaining 40. The stratified",
        r"percentile gate is unchanged at $q^{*}=41$; with five same-study cells in the",
        r"bank the rule now takes its reference from Attia's own coverage distribution",
        r"rather than falling back to the whole bank. Mean $\pm$ standard deviation over",
        r"five seeded repeats; the zero-shot row is the unadapted deployed configuration",
        r"on all 45 cells, shown for contrast. FA counts served predictions with APE",
        r"above 30\%, UR abstentions on cells the model would have called within that",
        r"bound. $\rho_{\mathrm{life}}$ is Spearman(coverage, cycle life) over the",
        r"evaluated population --- the confound check that reads $+0.055$ in-study.",
        r"\label{tab:attia-adaptation}}",
        r"\begin{tabularx}{\textwidth}{lCCCCCC}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Retention (\%)} & \textbf{RMSE (cycles)} & "
        r"\textbf{MAPE (\%)} & \textbf{FA} & \textbf{UR} & "
        r"$\boldsymbol{\rho_{\mathrm{life}}}$ \\",
        r"\midrule",
        f"Zero-shot ($n=0$, 45 cells) & {z['retention_pct']:.1f} & "
        f"{z['rmse_retained']:.1f} & {z['mape_retained']:.1f} & "
        f"{z['false_acceptance']} & {z['unnecessary_rejection']} & "
        f"{z['spearman_cov_vs_life_all']:+.3f} \\\\",
        f"Adapted ($n=5$, 40 cells) & {pm('retention_pct')} & {pm('rmse_retained')} & "
        f"{pm('mape_retained')} & {pm('false_acceptance')} & "
        f"{pm('unnecessary_rejection')} & "
        f"${a['spearman_cov_vs_life_all']['mean']:+.3f} \\pm "
        f"{a['spearman_cov_vs_life_all']['std']:.3f}$ \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(tex) + "\n")
    print(f"[tex] -> {TABLE}")


def _append_section(a: dict) -> None:
    """Idempotently append the adaptation section to the zero-shot report."""
    z = a["zero_shot_reference"]
    L, ap = [], None
    ap = L.append
    ap(MARKER.strip())
    ap("")
    ap(f"Five labelled Attia cells added to the bank at random, predictor (point and "
       f"q16/q84) refitted on Severson + 5, gate unchanged at q*={Q_STAR:g}, evaluated "
       f"on the remaining {a['design']['n_eval']}. Mean ± std over "
       f"{a['design']['repeats']} seeded repeats. With five same-study cells present "
       f"the stratified rule now draws its reference from Attia itself "
       f"(`{'`, `'.join(a['design']['reference_source'])}`) instead of the whole-bank "
       "fallback.")
    ap("")
    ap("| | zero-shot (n=0, 45 cells) | adapted (n=5, 40 cells) |")
    ap("|---|---|---|")
    ap(f"| gate threshold | 4.070 | {a['threshold']['mean']:.3f} ± {a['threshold']['std']:.3f} |")
    ap(f"| retention % | {z['retention_pct']:.1f} | "
       f"{a['retention_pct']['mean']:.1f} ± {a['retention_pct']['std']:.1f} |")
    ap(f"| retained RMSE | {z['rmse_retained']:.1f} | "
       f"{a['rmse_retained']['mean']:.1f} ± {a['rmse_retained']['std']:.1f} |")
    ap(f"| retained MAPE % | {z['mape_retained']:.1f} | "
       f"{a['mape_retained']['mean']:.1f} ± {a['mape_retained']['std']:.1f} |")
    ap(f"| false acceptances | {z['false_acceptance']} | "
       f"{a['false_acceptance']['mean']:.1f} ± {a['false_acceptance']['std']:.1f} |")
    ap(f"| unnecessary rejections | {z['unnecessary_rejection']} | "
       f"{a['unnecessary_rejection']['mean']:.1f} ± {a['unnecessary_rejection']['std']:.1f} |")
    ap(f"| ungated RMSE (all evaluated) | 178.7 | "
       f"{a['rmse_ungated']['mean']:.1f} ± {a['rmse_ungated']['std']:.1f} |")
    ap("")
    ap("### The two diagnostics after adaptation")
    ap("")
    for key, label, ref in [
            ("spearman_cov_vs_life_all",
             "Spearman(coverage, cycle life), all evaluated", "-0.643 zero-shot, +0.055 in-study"),
            ("spearman_cov_vs_life_retained",
             "Spearman(coverage, cycle life), retained only", "—"),
            ("spearman_cov_vs_abserr_retained",
             "Spearman(coverage, |error|), retained only", "-0.127 zero-shot, -0.247 in-study")]:
        v = a[key]
        ap(f"- **{label}**: "
           + (f"{v['mean']:+.3f} ± {v['std']:.3f} (reference: {ref})" if v
              else f"not computable — fewer than 10 retained cells in every repeat "
                   f"(reference: {ref})"))
    ap("")
    lp = a["spearman_cov_vs_life_all"]
    if lp:
        ap("The lifetime-proxy confound "
           + ("**persists** after adaptation" if abs(lp["mean"]) > 0.3 else
              "**weakens substantially** after adaptation")
           + f" ({lp['mean']:+.3f} vs {z['spearman_cov_vs_life_all']:+.3f} zero-shot). "
           + ("Five cells are not enough to stop coverage tracking lifetime on this "
              "population; the retained subset is still being chosen partly for how "
              "long its cells live."
              if abs(lp["mean"]) > 0.3 else
              "Giving the gate same-study reference points changes what it selects for."))
    ap("")
    ap("### Band calibration after retraining")
    ap("")
    ap(f"| set | empirical coverage | nominal | model over-predicted (of "
       f"{a['design']['n_eval']}) |")
    ap("|---|---|---|---|")
    ap(f"| retained | {a['band_coverage_retained']['mean']:.0%} ± "
       f"{a['band_coverage_retained']['std']:.0%} | {NOMINAL_BAND:.0%} | — |")
    ap(f"| all evaluated | {a['band_coverage_all']['mean']:.0%} ± "
       f"{a['band_coverage_all']['std']:.0%} | {NOMINAL_BAND:.0%} | "
       f"{a['band_overpredicted_all']['mean']:.1f} ± "
       f"{a['band_overpredicted_all']['std']:.1f} |")
    ap("")
    ap(f"Zero-shot the band covered {z['band_coverage_retained']:.0%} of retained cells "
       f"with all 35 misses on one side (over-prediction). ")
    ap("")
    ap("Artifacts: `attia_adaptation.json`, `attia_adaptation_per_repeat.csv`, "
       "`paper/tables/tab_attia_adaptation.tex`.")

    text = REPORT.read_text() if REPORT.exists() else "# Attia 2020\n"
    text = text.split(MARKER)[0].rstrip() + "\n"
    REPORT.write_text(text + "\n" + "\n".join(L) + "\n")
    print(f"[md]  appended -> {REPORT}")


if __name__ == "__main__":
    main()
