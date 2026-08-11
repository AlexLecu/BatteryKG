"""Step 5c — where do the SNL cells sit in the deployed behaviour space?

READ-ONLY in every sense that matters: it computes features, places them
against the frozen neighbour bank, and reports a number. It writes nothing to
the graph, adds nothing to the bank, and retrains nothing.

The honest framing, which the number alone does not convey:

  The bank's behaviour view is defined on Severson's protocol — dQ(V) between
  cycles 10 and 100 of a 4C discharge. SNL provides no such measurement. Its
  ordinary cycling discharges are sampled at 120 s (9-58 points; too sparse to
  invert), and its dense discharges are the 0.5C RPTs, which occur at cycles
  ~1-3 and then only every 200-500 cycles. So SNL cells CANNOT be placed in the
  paper's behaviour space on the paper's own feature definition. Any coverage
  number therefore rests on a SURROGATE definition, and is reported as such.

  Surrogate used here: dQ(V) between the first RPT and the first RPT at least
  150 ageing cycles later, both at 0.5C, with the >60 s gap-split the
  feasibility report requires (some 35 degC RPT cycles file two discharge
  events under one Cycle_Index; unsplit, correlation with the true curve falls
  to 0.32). This is the route that feasibility study established as viable.

Three distortions separate this from a like-for-like number, all in the
direction of understating similarity: discharge rate (0.5C vs 4C), cycle window
(1 -> 200-500 vs 10 -> 100), and the fact that the deployed z-score scaler is
Severson's. The result should be read as "SNL is far outside the bank", not as
a calibrated distance.

Run:  python -m experiments.exp08_snl_ingestion.coverage
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from app.common import load_artifacts
from experiments.exp05_hust_adaptation.core import GraphIndex, severson_bank
from experiments.exp08_snl_ingestion.snl_ingest import (NOMINAL_AH, OUT_PARQUET, ZIP_PATH,
                                                        _parse_name)
from experiments.exp08_snl_ingestion.stage import OUT

V_GRID = np.arange(2.0, 3.50, 0.01)
GAP_S = 60.0                 # feasibility report: split discharge runs on >60 s gaps
MIN_RPT_PTS = 200            # dense (10 s) RPT discharges carry ~730-780 points
MIN_CYCLE_GAP = 150          # second RPT must be >= this many ageing cycles later
TS_COLS = ["Test_Time (s)", "Cycle_Index", "Current (A)", "Voltage (V)",
           "Discharge_Capacity (Ah)"]


def _longest_run(seg: pd.DataFrame) -> pd.DataFrame:
    """Longest contiguous discharge run — the mandatory >60 s gap-split."""
    seg = seg.sort_values("Test_Time (s)")
    gap = (seg["Test_Time (s)"].diff() > GAP_S).cumsum()
    return max((g for _, g in seg.groupby(gap)), key=len)


def _qv(seg: pd.DataFrame) -> np.ndarray | None:
    g = (pd.DataFrame({"v": np.round(seg["Voltage (V)"].to_numpy(float), 3),
                       "q": seg["Discharge_Capacity (Ah)"].to_numpy(float)})
         .groupby("v")["q"].median().sort_index())
    if len(g) < 4:
        return None
    Q = np.full(V_GRID.shape, np.nan)
    m = (V_GRID >= g.index.min()) & (V_GRID <= g.index.max())
    Q[m] = np.interp(V_GRID[m], g.index.values, g.values)
    return Q


def surrogate_features(cycles: pd.DataFrame) -> pd.DataFrame:
    """Behaviour-view features for each SNL cell, on the RPT surrogate."""
    caps = {c: g.set_index("cycle_index")["discharge_capacity_ah"]
            for c, g in cycles.groupby("cell_id")}
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = sorted(n for n in z.namelist() if n.endswith("_timeseries.csv"))
        for name in names:
            if _parse_name(name.replace("_timeseries.csv", "_cycle_data.csv")) is None:
                continue
            stem = name.rsplit("/", 1)[-1].replace("_timeseries.csv", "")
            with z.open(name) as fh:
                ts = pd.read_csv(io.BytesIO(fh.read()), usecols=TS_COLS)
            dis = ts[ts["Current (A)"] < -0.01]
            pc = dis.groupby("Cycle_Index").size()
            dense = sorted(pc[pc > MIN_RPT_PTS].index)
            if len(dense) < 2:
                print(f"  {stem}: <2 dense RPTs — skipped")
                continue
            c1 = dense[0]
            later = [c for c in dense if c - c1 >= MIN_CYCLE_GAP]
            c2 = later[0] if later else dense[-1]
            Q1 = _qv(_longest_run(dis[dis["Cycle_Index"] == c1]))
            Q2 = _qv(_longest_run(dis[dis["Cycle_Index"] == c2]))
            if Q1 is None or Q2 is None:
                continue
            dQ = (Q2 - Q1)[np.isfinite(Q2 - Q1)]
            s = caps[stem]
            early, late = s.index.min(), s.index.max()
            # capacity anchors: first and the RPT nearest c2 (the surrogate's "late")
            cap_e = float(s.iloc[0])
            cap_l = float(s.iloc[(np.abs(s.index.to_numpy() - c2)).argmin()])
            span = s.index[(s.index >= c1) & (s.index <= c2)]
            slope = float(np.polyfit(span, s.loc[span], 1)[0]) if len(span) > 2 else np.nan
            rows.append({
                "cell_id": stem, "rpt_early": int(c1), "rpt_late": int(c2),
                "var_dQ_100_10": float(np.log10(np.var(dQ))),
                "min_dQ_100_10": float(np.log10(np.abs(np.min(dQ)))),
                "slope_2_100": slope,
                "capacity_cycle_2": cap_e, "capacity_cycle_100": cap_l,
                "cap_ratio_100_2": cap_l / cap_e,
                "n_cycles_spanned": int(c2 - c1),
                "cycle_first": int(early), "cycle_last": int(late),
            })
            print(f"  {stem}: RPT {c1}->{c2}", flush=True)
    return pd.DataFrame(rows)


def behavior_coverage(meta: dict, bank: pd.DataFrame, queries: pd.DataFrame,
                      exclude_own_group: bool = False) -> np.ndarray:
    """Sum of the top-k behaviour-view neighbour weights against `bank`.

    Same maths as the served path (frozen z-score scaler, weight 1/(1+d),
    top-k sum), restricted to the behaviour view and with self-exclusion by
    cell_id per the Task-1 fix.
    """
    from app.common import _zscore_with
    sc = meta["scaler"]["behavior"]
    cols, k = sc["columns"], meta["k_neighbors"]
    B = _zscore_with(bank[cols].to_numpy(float), sc["mean"], sc["std"])
    Q = _zscore_with(queries[cols].to_numpy(float), sc["mean"], sc["std"])
    W = 1.0 / (1.0 + np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2))
    qid = queries["cell_id"].to_numpy() if "cell_id" in queries else queries["study_cell_id"].to_numpy()
    W = np.where(qid[:, None] == bank["cell_id"].to_numpy()[None, :], -np.inf, W)
    if exclude_own_group:
        qg = queries["policy_group_id"].to_numpy()
        W = np.where(qg[:, None] == bank["policy_group_id"].to_numpy()[None, :], -np.inf, W)
    return np.sort(W, axis=1)[:, -k:].sum(axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta = art["meta"]
    thr = meta["abstention"]["threshold"]

    cycles = pd.read_parquet(OUT_PARQUET)
    feats = surrogate_features(cycles)

    # Place them against the FROZEN deployed bank; nothing is added to it.
    # Only the BEHAVIOUR view is computed: SNL has no two-step charge policy, so
    # its condition features are null by construction and that view is undefined
    # for these cells (GraphIndex would raise rather than invent a placement).
    bank = severson_bank()
    cov = behavior_coverage(meta, bank, feats)
    feats["behavior_coverage"] = cov
    sev_cov = behavior_coverage(meta, bank, bank, exclude_own_group=True)

    summary = {
        "framing": ("SNL cells cannot be placed in the deployed behaviour space on "
                    "the paper's own feature definition (4C discharge, cycles 10 and "
                    "100). The numbers below use an RPT surrogate (0.5C, first RPT vs "
                    "the first RPT >=150 ageing cycles later) and are therefore not a "
                    "calibrated distance."),
        "n_cells": int(len(feats)),
        "surrogate_cycle_window": [int(feats["rpt_early"].min()),
                                   int(feats["rpt_late"].max())],
        "snl_coverage": {"min": float(cov.min()), "median": float(np.median(cov)),
                         "max": float(cov.max()), "mean": float(cov.mean())},
        "severson_in_study_coverage": {
            "min": float(sev_cov.min()), "median": float(np.median(sev_cov)),
            "max": float(sev_cov.max())},
        "deployed_threshold": thr,
        "snl_cells_above_threshold": int((cov >= thr).sum()),
        "bank_unchanged": {"cells": int(len(bank)),
                           "snl_added_to_bank": 0, "models_retrained": 0},
        "distortions": [
            "discharge rate: 0.5C RPT vs the bank's 4C basis",
            f"cycle window: {int(feats['n_cycles_spanned'].median())} ageing cycles "
            "(median) vs the bank's 10 -> 100",
            "z-score scaler is Severson's, frozen",
        ],
    }
    feats.to_csv(OUT / "snl_behavior_coverage.csv", index=False)
    (OUT / "snl_coverage_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== behaviour-view coverage of SNL against the frozen bank ===")
    print(f"  SNL:      min {cov.min():.2f}  median {np.median(cov):.2f}  max {cov.max():.2f}")
    print(f"  Severson: min {sev_cov.min():.2f}  median {np.median(sev_cov):.2f}  "
          f"max {sev_cov.max():.2f}  (in-study, own group excluded)")
    print(f"  deployed threshold {thr:.3f} -> {int((cov >= thr).sum())}/{len(cov)} "
          "SNL cells would pass")
    print(f"  bank unchanged: {len(bank)} cells, 0 SNL added, 0 models retrained")
    return feats, summary


if __name__ == "__main__":
    main()
