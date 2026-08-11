"""Experiment 04 — cross-study external validation on HUST (zero-shot).

The deployed configuration is tested AS DEPLOYED: the frozen serving artifacts
(model trained on all 120 Severson cells, Severson-only neighbor bank,
Severson-tuned abstention threshold) are applied to the 77 HUST cells with no
re-tuning of any kind. Same commercial cell (A123 APR18650M1A), mirrored
protocol design (HUST varies discharge at fixed 5C(80%)-1C charge; Severson
varied charge at fixed 4C discharge), and longer-lived cells (median 1875 vs
736 cycles) — a genuine distribution-shift test.

Rows produced: (a) graph model ungated on all HUST cells; (b) the deployed
gate's retained vs abstained split at the transferred threshold; (c) the
graph-free GBM baseline ("what everyone else does"); with the in-study
(experiment 01) numbers alongside.

Run:  python -m src.models.experiment_04
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.common import load_artifacts, predict_with_gate
from src.config import OUTPUTS, PROCESSED, ROOT
from src.ingestion.qc import cycle_life_table
from src.models.common import make_model

REPORT = OUTPUTS / "experiment_04_cross_study.md"
PAPER_TABLE = ROOT / "paper" / "tables" / "tab_cross_study.tex"

# In-study reference (experiment 01 / serving meta; canonical outputs).
#
# `retained_rmse` is the DEPLOYED gate's in-study number. The deployed gate is
# now the quantile-referenced one (retain when coverage reaches the q*=41st
# percentile of the bank's own leave-one-out coverage distribution), so this is
# 79.0 and not the 72.0 of the superseded absolute 4.08 threshold. Both retain
# exactly 72 of 120 cells; they disagree on 2 of them, and one of those carries
# a large error. Source: outputs/experiment_06_quantile_gate/deployed_gate.json
# (`rmse_retained_cycles`, and `operating_point_comparison` for the delta).
IN_STUDY = {"rmse_cycles": 135.2, "mape_pct": 9.69,
            "retained_frac": 0.60, "retained_n": 72, "retained_rmse": 79.0}

HUST_POLICY = {"c_rate_1": 5.0, "c_rate_2": 1.0, "soc_transition_pct": 80.0}


def build_hust_eval() -> pd.DataFrame:
    feats = pd.read_parquet(PROCESSED / "hust_features.parquet")
    cycles = pd.read_parquet(PROCESSED / "hust_cycles.parquet")
    life = cycle_life_table(cycles)[["cell_id", "cycle_life_nominal", "flags"]]
    df = feats.merge(life, on="cell_id", how="inner")
    excluded = df.loc[df["flags"] != "", "cell_id"].tolist()
    if excluded:
        print(f"[exp04] excluding QC-flagged HUST cells: {excluded}")
    return df[df["flags"] == ""].reset_index(drop=True)


def _rmse(t, p):
    return float(np.sqrt(np.mean((np.asarray(t) - np.asarray(p)) ** 2)))


def _mape(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - t) / t) * 100)


def _cell(v, fmt="{:.1f}", dash="---"):
    """Render a metric for a table cell.

    A metric over an empty set (the gate retains nothing, so there is no
    retained RMSE) is undefined, not zero and not a number — print a dash
    rather than letting a float NaN reach the page as the string 'nan'.
    """
    if v is None or not np.isfinite(v):
        return dash
    return fmt.format(v)


def main() -> None:
    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta, bank = art["meta"], art["bank"]
    thr = meta["abstention"]["threshold"]
    hust = build_hust_eval()
    n = len(hust)

    # --- deployed graph model + gate, zero-shot -------------------------------
    rows = []
    for _, r in hust.iterrows():
        query = {c: float(r[c]) for c in meta["behavior_features"]}
        for c in meta["base_features"]:
            if c in HUST_POLICY:
                query[c] = HUST_POLICY[c]
            elif c == "batch":
                query[c] = float("nan")       # Severson-specific covariate
            elif c not in query:
                query[c] = float(r[c])
        res = predict_with_gate(art, query, exclude_group=None)
        rows.append({"cell_id": r["cell_id"], "actual": float(r["cycle_life_nominal"]),
                     "pred": res["prediction_cycles"], "coverage": res["coverage"],
                     "abstain": res["abstain"]})
    g = pd.DataFrame(rows)

    # --- graph-free GBM baseline (same seed/config, trained on the bank) ------
    base_model = make_model()
    base_model.fit(bank[meta["base_features"]], bank["log10_cycle_life"])
    Xh = pd.DataFrame([{c: (HUST_POLICY.get(c, float("nan")) if c in HUST_POLICY
                            or c == "batch" else float(r[c]))
                        for c in meta["base_features"]}
                       for _, r in hust.iterrows()])
    g["pred_baseline"] = 10 ** base_model.predict(Xh)

    retained, abstained = g[~g["abstain"]], g[g["abstain"]]
    abst_rate = len(abstained) / n

    stats = {
        "baseline_rmse": _rmse(g["actual"], g["pred_baseline"]),
        "baseline_mape": _mape(g["actual"], g["pred_baseline"]),
        "graph_rmse": _rmse(g["actual"], g["pred"]),
        "graph_mape": _mape(g["actual"], g["pred"]),
        "retained_rmse": _rmse(retained["actual"], retained["pred"]) if len(retained) else float("nan"),
        "retained_mape": _mape(retained["actual"], retained["pred"]) if len(retained) else float("nan"),
        "abstained_rmse": _rmse(abstained["actual"], abstained["pred"]) if len(abstained) else float("nan"),
        "abstention_rate": abst_rate,
        "n_retained": int(len(retained)),
    }

    # --- risk-coverage on HUST with the fixed threshold marked ------------------
    from src.viz.paper_style import BLUE, INK_MUT, ORANGE, W_FULL, apply_style, save_fig
    import matplotlib.pyplot as plt
    apply_style()
    sq = (g["actual"] - g["pred"]) ** 2
    thrs = np.unique(g["coverage"])
    fr, rk = [], []
    for t in thrs:
        keep = g["coverage"] >= t
        if keep.sum() == 0:
            continue
        fr.append(keep.mean())
        rk.append(float(np.sqrt(sq[keep].mean())))
    fig, ax = plt.subplots(figsize=(W_FULL, 2.9))
    ax.plot(fr, rk, lw=1.6, color=ORANGE, label="coverage-gated (HUST, zero-shot)")
    ax.axhline(stats["graph_rmse"], ls=":", lw=1.2, color=INK_MUT,
               label=f"ungated RMSE ({stats['graph_rmse']:.0f} cyc)")
    ax.axvline(1 - abst_rate, ls="--", lw=1.4, color=BLUE,
               label=f"deployed threshold ({thr:.2f}) -> retains {1-abst_rate:.0%}")
    ax.set_xlabel("fraction of HUST cells retained")
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    save_fig(fig, "risk_coverage_cross_study")
    for ext in ("png", "pdf"):
        (OUTPUTS / f"risk_coverage_cross_study.{ext}").write_bytes(
            (ROOT / "paper" / "figures" / f"risk_coverage_cross_study.{ext}").read_bytes())

    # --- report ------------------------------------------------------------------
    L = ["# Experiment 04 — cross-study external validation (HUST, zero-shot)", ""]
    a = L.append
    a(f"{n} HUST cells (A123 APR18650M1A — same commercial cell as training; "
      "discharge-varied protocol vs Severson's charge-varied; cycle life median "
      f"{hust['cycle_life_nominal'].median():.0f} vs 736 in-study). Deployed "
      "artifacts applied unchanged: Severson-trained model, Severson-only "
      f"neighbor bank, transferred threshold {thr:.2f}. No re-tuning.\n")
    a("| configuration | n | RMSE (cycles) | MAPE (%) |")
    a("|---|---|---|---|")
    a(f"| in-study CV (graph model, exp 01) | 120 | {IN_STUDY['rmse_cycles']:.1f} | {IN_STUDY['mape_pct']:.2f} |")
    a(f"| in-study CV, gate-retained {IN_STUDY['retained_frac']:.0%} | "
      f"{IN_STUDY['retained_n']} | {IN_STUDY['retained_rmse']:.1f} | — |")
    a(f"| HUST zero-shot, graph-free GBM (no gate) | {n} | {stats['baseline_rmse']:.1f} | {stats['baseline_mape']:.1f} |")
    a(f"| HUST zero-shot, graph model (ungated) | {n} | {stats['graph_rmse']:.1f} | {stats['graph_mape']:.1f} |")
    a(f"| HUST zero-shot, gate-RETAINED | {stats['n_retained']} | "
      f"{_cell(stats['retained_rmse'], dash='—')} | "
      f"{_cell(stats['retained_mape'], dash='—')} |")
    a(f"| HUST zero-shot, gate-ABSTAINED (error had it answered) | {n - stats['n_retained']} | "
      f"{_cell(stats['abstained_rmse'], dash='—')} | — |")
    a(f"\nAbstention rate on HUST: **{abst_rate:.1%}** (in-study at the same "
      f"threshold: 40%).\n")
    a("## Verdict\n")
    if stats["n_retained"] == 0:
        a("The gate refuses the ENTIRE cross-study population — and the numbers "
          "say it is right to. Had the system answered, every prediction would "
          f"have been catastrophically wrong (MAPE {stats['graph_mape']:.0f}%; "
          "the model, trained where high delta-Q variance means short life, "
          "predicts a few hundred cycles for cells that actually last "
          f"{hust['cycle_life_nominal'].min():.0f}-"
          f"{hust['cycle_life_nominal'].max():.0f}). Cross-study coverage "
          f"(max {g['coverage'].max():.2f}) sits far below the deployed "
          f"threshold ({thr:.2f}): the multi-stage-discharge protocol places "
          "HUST cells in a region of behavior space the Severson bank simply "
          "does not cover, and the gate detects that BEFORE any label is seen. "
          "The graph-free baseline — the standard deployment — answers every "
          f"cell with ~{stats['baseline_mape']:.0f}% error and no warning. "
          "The honest cost: zero utility on the new study until its data "
          "enters the graph; that is the designed behavior, and precisely what "
          "the self-updating ingestion path exists to fix.")
    elif abst_rate > 0.40 and stats["retained_rmse"] < stats["graph_rmse"]:
        a("The gate behaves honestly out of distribution: it abstains on a far "
          "larger fraction of cross-study cells than in-study, and the cells it "
          "does retain carry materially lower error than the ungated set.")
    else:
        a("TODO-review: automated verdict — inspect the numbers above; the gate "
          "did not clearly separate retained from abstained error on this run.")
    a("\nCoverage distribution: HUST coverage min/median/max = "
      f"{g['coverage'].min():.2f}/{g['coverage'].median():.2f}/{g['coverage'].max():.2f} "
      f"(threshold {thr:.2f}).")
    REPORT.write_text("\n".join(L) + "\n")
    print(f"[exp04] report -> {REPORT}")

    # --- paper table ----------------------------------------------------------------
    tex = [
        "% Auto-generated by src/models/experiment_04.py — do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Cross-study external validation: the deployed configuration",
        "(Severson-trained model, Severson-only neighbor bank, transferred",
        "abstention threshold) applied zero-shot to the 77 HUST cells --- the",
        "same commercial cell under a mirrored protocol design (varied discharge,",
        "fixed fast charge) with substantially longer lifetimes. No re-tuning of",
        "any component.\\label{tab:crossstudy}}",
        "\\begin{tabularx}{\\textwidth}{lCCC}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{n} & \\textbf{RMSE (cycles)} & \\textbf{MAPE (\\%)} \\\\",
        "\\midrule",
        f"In-study CV (graph model) & 120 & {IN_STUDY['rmse_cycles']:.1f} & {IN_STUDY['mape_pct']:.2f} \\\\",
        f"In-study CV, gate-retained {IN_STUDY['retained_frac'] * 100:.0f}\\% & "
        f"{IN_STUDY['retained_n']} & {IN_STUDY['retained_rmse']:.1f} & --- \\\\",
        "\\midrule",
        f"HUST zero-shot, graph-free GBM & {n} & {stats['baseline_rmse']:.1f} & {stats['baseline_mape']:.1f} \\\\",
        f"HUST zero-shot, graph model (ungated) & {n} & {stats['graph_rmse']:.1f} & {stats['graph_mape']:.1f} \\\\",
        f"HUST zero-shot, gate-retained & {stats['n_retained']} & "
        f"{_cell(stats['retained_rmse'])} & {_cell(stats['retained_mape'])} \\\\",
        f"HUST zero-shot, gate-abstained & {n - stats['n_retained']} & "
        f"{_cell(stats['abstained_rmse'])} & --- \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}",
    ]
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text("\n".join(tex) + "\n")
    print(f"[exp04] table -> {PAPER_TABLE}")

    print("\n=== headline ===")
    for k, v in stats.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 2)}")


if __name__ == "__main__":
    main()
