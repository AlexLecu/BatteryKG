"""Task 3 — sensitivity of the cycle-life target to its definitional choices.

The EOL label (cycles to 80% of nominal, Section 3.3) is produced by three
choices in src/ingestion/qc.py:

  smoothing window     rolling median over the capacity trace before the
                       threshold test          (current: 5, centered, min_periods=1)
  threshold-proximity  a cell whose smoothed curve never dips below the
                       threshold still counts as reaching EOL at its last cycle
                       if its final measured capacity is within EOL_TOL of the
                       threshold                (current: 1%)
  interpolation        the crossing cycle is interpolated linearly between the
                       bracketing cycles        (current: linear)

Each variant re-labels the 124 Severson cells; nothing else changes. The
existing harness (leave-one-policy-group-out CV, the pinned publication edge
set, the risk-coverage machinery) is re-run on the new labels.

The paper configuration is reproduced first and asserted equal to
`src.ingestion.qc.cycle_life_table()`, so the re-implementation cannot drift
from the shipped one.

Run:  python -m experiments.exp07_revision_extras.target_sensitivity
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUTS, PROCESSED, ROOT, processed_parquet
from src.ingestion.qc import EOL_FRACTION, cycle_life_table, is_anomalous
from src.models.abstention import aurc, risk_coverage_sweep
from src.models.baseline import run_baseline
from src.models.common import metric_table
from src.models.dataset import BASE_FEATURES, TARGET
from src.models.graph_model import run_graph_model
from src.viz.make_paper_figures import publication_edges

OUT = OUTPUTS / "experiment_07_revision_extras"
TABLE = ROOT / "paper" / "tables" / "tab_target_sensitivity.tex"

PAPER = {"smooth_window": 5, "eol_tol": 0.01, "interp": "linear"}
VARIANTS = (
    [{"axis": "paper", "label": "paper configuration", **PAPER}]
    + [{"axis": "smoothing", "label": f"smoothing window {w}", **{**PAPER, "smooth_window": w}}
       for w in (1, 3, 7, 9)]
    + [{"axis": "tolerance", "label": f"proximity rule {t:.1%}".replace(".0%", "%"),
        **{**PAPER, "eol_tol": t}} for t in (0.005, 0.02)]
    + [{"axis": "tolerance", "label": "proximity rule none (censor)",
        **{**PAPER, "eol_tol": None}}]
    + [{"axis": "interpolation", "label": "nearest cycle",
        **{**PAPER, "interp": "nearest"}}]
)


# --- re-implementation of the label, with the three choices exposed ------------
def _crossing(cycles, caps, threshold, interp):
    below = caps <= threshold
    if not below.any():
        return None
    j = int(np.argmax(below))
    if j == 0:
        return float(cycles[0])
    c0, c1 = caps[j - 1], caps[j]
    n0, n1 = cycles[j - 1], cycles[j]
    if interp == "nearest":
        # whichever bracketing cycle's capacity sits closer to the threshold
        return float(n0 if abs(c0 - threshold) <= abs(c1 - threshold) else n1)
    if c0 == c1:
        return float(n1)
    return float(n0 + (c0 - threshold) / (c0 - c1) * (n1 - n0))


def cell_life(cell_df, smooth_window, eol_tol, interp):
    d = cell_df.sort_values("cycle_index")
    cycles = d["cycle_index"].to_numpy(float)
    caps = d["discharge_capacity_ah"].to_numpy(float)
    nominal = float(d["nominal_capacity_ah"].iloc[0])
    smoothed = (pd.Series(caps).rolling(smooth_window, center=True, min_periods=1)
                .median().to_numpy())
    thr = EOL_FRACTION * nominal
    life = _crossing(cycles, smoothed, thr, interp)
    if life is None and eol_tol is not None and float(caps[-1]) <= thr * (1 + eol_tol):
        life = float(cycles[-1])
    return life


def life_table(cycles_df, smooth_window, eol_tol, interp) -> pd.DataFrame:
    rows = []
    for cid, g in cycles_df.groupby("cell_id", sort=True):
        rows.append({"cell_id": cid,
                     "cycle_life_nominal": cell_life(g, smooth_window, eol_tol, interp)})
    out = pd.DataFrame(rows)
    out["reached_eol_nominal"] = out["cycle_life_nominal"].notna()
    return out


# --- modeling table on arbitrary labels ---------------------------------------
def build_dataset_with(life: pd.DataFrame, cycles_df: pd.DataFrame) -> pd.DataFrame:
    """`src.models.dataset.build_dataset` with the EOL labels swapped in.

    Cells without a label (no crossing under a variant that has no proximity
    rule) are right-censored and dropped, exactly as the task specifies.
    """
    from src.kg.features import parse_policy
    early = pd.read_parquet(PROCESSED / "severson_features.parquet")
    meta = (cycles_df.groupby("cell_id")
            .agg(batch=("batch", "first"), charge_policy_norm=("charge_policy_norm", "first"),
                 policy_group_id=("policy_group_id", "first")).reset_index())
    # QC anomaly flags do not depend on the EOL label, so they are taken as shipped
    anom = cycle_life_table(cycles_df)[["cell_id", "is_anomalous"]]

    df = (early.merge(meta, on="cell_id", how="inner")
          .merge(life, on="cell_id", how="inner")
          .merge(anom, on="cell_id", how="inner"))
    df = df[df["reached_eol_nominal"]].reset_index(drop=True)
    feats = df["charge_policy_norm"].map(parse_policy)
    df["c_rate_1"] = [f.c_rate_1 if f else np.nan for f in feats]
    df["c_rate_2"] = [f.c_rate_2 if f else np.nan for f in feats]
    df["soc_transition_pct"] = [f.soc_transition_pct if f else np.nan for f in feats]
    df[TARGET] = np.log10(df["cycle_life_nominal"].astype(float))
    return df.sort_values("cell_id").reset_index(drop=True)


def evaluate(df: pd.DataFrame, edges) -> dict:
    """Baseline RMSE + the abstention diagnostics, on whatever labels df carries."""
    d = df[~df["is_anomalous"]].reset_index(drop=True)
    if d["policy_group_id"].nunique() < 3 or len(d) < 20:
        return {"n_cells": len(d), "n_folds": d["policy_group_id"].nunique()}
    base = run_baseline(d)
    graph = run_graph_model(d, edges).sort_values("cell_id").reset_index(drop=True)
    yt, yp = graph["y_true_log"].to_numpy(), graph["y_pred_log"].to_numpy()
    cov = graph["behavior_coverage_train"].to_numpy()
    sweep = risk_coverage_sweep(cov, yt, yp, n_random=0)
    return {
        "n_cells": len(d), "n_folds": int(d["policy_group_id"].nunique()),
        "baseline_rmse_cycles": metric_table(base, "b")["rmse_cycles"],
        "graph_rmse_cycles": metric_table(graph, "g")["rmse_cycles"],
        "graph_mape_pct": metric_table(graph, "g")["mape_pct"],
        "spearman_cov_err_log": float(stats.spearmanr(cov, np.abs(yt - yp)).statistic),
        "aurc": float(aurc(sweep["frac_retained"], sweep["rmse_retained"])),
    }


def main(table_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "target_sensitivity.csv"
    if table_only:
        # re-render the LaTeX from the stored grid; no models, no re-labelling
        sens = pd.read_csv(csv)
        ref = sens[sens.axis == "paper"].iloc[0].to_dict()
        _tex(sens, ref)
        return sens, ref

    cycles = pd.read_parquet(processed_parquet("severson_mit"))
    edges = publication_edges()

    # the paper label, and the assertion that this module reproduces it
    shipped = cycle_life_table(cycles)[["cell_id", "cycle_life_nominal"]]
    mine = life_table(cycles, **PAPER)
    merged = shipped.merge(mine, on="cell_id", suffixes=("_shipped", "_mine"))
    if not np.allclose(merged["cycle_life_nominal_shipped"].astype(float),
                       merged["cycle_life_nominal_mine"].astype(float), atol=1e-9):
        bad = merged[~np.isclose(merged["cycle_life_nominal_shipped"].astype(float),
                                 merged["cycle_life_nominal_mine"].astype(float))]
        raise AssertionError(f"re-implementation differs from qc.cycle_life_table:\n{bad}")
    print(f"[target_sensitivity] paper label reproduced exactly for {len(merged)} cells")

    ref_life = mine.set_index("cell_id")["cycle_life_nominal"]
    ref_eval = None
    rows = []
    for v in VARIANTS:
        cfg = {k: v[k] for k in ("smooth_window", "eol_tol", "interp")}
        lt = life_table(cycles, **cfg)
        s = lt.set_index("cell_id")["cycle_life_nominal"]
        both = ref_life.notna() & s.notna()
        delta = (s[both] - ref_life[both]).abs()
        changed = int((delta > 1e-9).sum())
        res = evaluate(build_dataset_with(lt, cycles), edges)
        if v["axis"] == "paper":
            ref_eval = res
        rows.append({
            "axis": v["axis"], "label": v["label"], **cfg,
            "n_labelled": int(s.notna().sum()),
            "n_lost": int((ref_life.notna() & s.isna()).sum()),
            "n_changed": changed,
            "median_abs_delta": float(delta[delta > 1e-9].median()) if changed else 0.0,
            "max_abs_delta": float(delta.max()) if changed else 0.0,
            **res,
        })
        print(f"  {v['label']:32s} labelled={rows[-1]['n_labelled']:3d} "
              f"changed={changed:3d} baseRMSE={res.get('baseline_rmse_cycles', float('nan')):7.1f}")

    sens = pd.DataFrame(rows)
    for c in ("baseline_rmse_cycles", "graph_rmse_cycles", "spearman_cov_err_log", "aurc"):
        sens[f"d_{c}"] = sens[c] - ref_eval[c]
    sens.to_csv(OUT / "target_sensitivity.csv", index=False)
    _tex(sens, ref_eval)
    print(f"\n[target_sensitivity] -> {OUT / 'target_sensitivity.csv'}")
    return sens, ref_eval


def _tex(sens: pd.DataFrame, ref: dict) -> None:
    def row(r):
        chg = "---" if r["axis"] == "paper" else str(int(r["n_changed"]))
        def d(v):
            if r["axis"] == "paper" or r["n_changed"] == 0:
                return "---"
            return f"{v:.3g}" if v >= 0.01 else r"$<\!0.01$"
        med, mx = d(r["median_abs_delta"]), d(r["max_abs_delta"])
        return " & ".join([
            r["label"].replace("%", r"\%"), str(int(r["n_labelled"])), chg, med, mx,
            f"{r['baseline_rmse_cycles']:.1f}",
            f"{r['spearman_cov_err_log']:.3f}", f"{r['aurc']:.1f}"]) + r" \\"

    body = []
    for axis, head in [("paper", None), ("smoothing", "Smoothing window"),
                       ("tolerance", "Threshold-proximity rule"),
                       ("interpolation", "Crossing interpolation")]:
        sub = sens[sens.axis == axis]
        if head:
            body.append(r"\midrule")
            body.append(r"\multicolumn{8}{l}{\textit{" + head + r"}} \\")
        body += [row(r) for _, r in sub.iterrows()]

    tex = [
        "% Auto-generated by experiments/exp07_revision_extras/target_sensitivity.py",
        "% — do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Sensitivity of the cycle-life target to the three definitional",
        r"choices behind it (Section~\ref{sec:eol}), over the 124 Severson cells. Each",
        r"row re-labels the cells and re-runs the unchanged evaluation harness",
        r"(leave-one-policy-group-out CV, identical features, seed, and similarity",
        r"graph); only the labels differ. \textit{Lab.}\ is the number of cells",
        r"receiving an EOL label, \textit{Chg.}\ the number whose label moves",
        r"relative to the paper configuration, and $|\Delta|$ the shift in cycles",
        r"among those. The proximity rule is what makes the target well defined for",
        r"most of this dataset: the Severson cells are cycled until they reach",
        r"$80\%$ and then stopped, so the smoothed capacity curve of 81 of the 124",
        r"cells never dips strictly below the threshold. Removing the rule censors",
        r"them, and the resulting metrics refer to a different, shorter-lived",
        r"subpopulation rather than to a worse model. The graph model tracks the",
        r"baseline throughout and is omitted for width.\label{tab:target-sensitivity}}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{lCCCCCCC}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Lab.} & \textbf{Chg.} & "
        r"\textbf{med.\ $|\Delta|$} & \textbf{max $|\Delta|$} & "
        r"\textbf{Base RMSE} & \textbf{Spearman} & \textbf{AURC} \\",
        r" & (of 124) & & (cyc) & (cyc) & (cyc) & $\rho$ & (cyc) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(tex) + "\n")
    print(f"[target_sensitivity] -> {TABLE}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table-only", action="store_true",
                   help="re-render the LaTeX table from target_sensitivity.csv "
                        "without re-running the label variants or the CV")
    main(table_only=p.parse_args().table_only)
