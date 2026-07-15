"""Loader for the HUST 77-cell dataset (Ma et al. 2022, E&ES; CC BY 4.0).

Same commercial cell as Severson
(A123 APR18650M1A, 1.1 Ah); fixed charge (5C to 80% SOC, 1C to 3.6 V, CV) —
expressible in our policy grammar as 5C(80%)-1C — with per-cell varied
multi-stage discharge (rates vendored from BatteryML's preprocessing table,
originally ESI Table S1). Cell 7-5's first two cycles are faulty and skipped
(as in BatteryML).

Outputs (same schema/definitions as the Severson artifacts):
  data/processed/hust_cycles.parquet    per-cycle tidy frame (conform())
  data/processed/hust_features.parquet  early-cycle features, cycles <= 100

Attribution: dataset DOI 10.17632/nsc7hnsg4s.2; paper DOI 10.1039/d2ee01676a.

Download (manual — not covered by src.ingestion.download): from Mendeley Data
https://data.mendeley.com/datasets/nsc7hnsg4s/2 fetch `our_data.zip` (~1.2 GB)
and save it as data/raw/hust/hust_data.zip. The loader streams the per-cell
pickles directly from the zip; no extraction needed.

Run:  python -m src.ingestion.hust
"""
from __future__ import annotations

import json
import pickle
import zipfile

import numpy as np
import pandas as pd

from src.config import DATA, PROCESSED, RAW
from src.ingestion import cell_metadata as cm
from src.ingestion.schema import conform

STUDY = "hust"
RAW_HUST = RAW / "hust"
ZIP_PATH = RAW_HUST / "hust_data.zip"
RATES_JSON = DATA / "hust_discharge_rates.json"   # vendored from BatteryML (MIT)

CHARGE_POLICY = "5C(80%)-1C"          # fixed for all HUST cells (paper Methods)
POLICY_GROUP = "pgH00"                # single cross-study policy group
TEMPERATURE_C = 30.0
SKIP_FIRST_CYCLES = {"7-5": 2}        # faulty first cycles (BatteryML does the same)

DQ_EARLY, DQ_LATE, SLOPE_LO, CYCLE_MAX = 10, 100, 2, 100
V_GRID_N = 500


def _iter_cell_pickles():
    """Yield (cell_id, payload) streamed DIRECTLY from the zip — the archive
    is ~1.2 GB but expands to ~3+ GB, so we never extract to disk."""
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"{ZIP_PATH} missing — manual download: get our_data.zip from "
            "Mendeley Data https://data.mendeley.com/datasets/nsc7hnsg4s/2 "
            f"(DOI 10.17632/nsc7hnsg4s.2) and save it as {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".pkl"))
        for name in names:
            cell_id = name.rsplit("/", 1)[-1][:-4]
            with z.open(name) as fh:
                yield cell_id, pickle.load(fh)


def _discharge_qv(cycle_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(V, Qd) samples of the discharge segment; Qd = capacity discharged so
    far [Ah]. The Capacity column counts down during discharge."""
    dis = cycle_df[cycle_df["Status"].str.contains("discharge", case=False)]
    cap = dis["Capacity (mAh)"].to_numpy(float)
    v = dis["Voltage (V)"].to_numpy(float)
    if len(cap) < 10:
        return np.array([]), np.array([])
    qd = (cap[0] - cap) / 1000.0
    return v, qd


def _qd_of_cycle(cycle_df: pd.DataFrame) -> float:
    dis = cycle_df[cycle_df["Status"].str.contains("discharge", case=False)]
    cap = dis["Capacity (mAh)"].to_numpy(float)
    if len(cap) < 2:
        return float("nan")
    return float((cap[0] - cap[-1]) / 1000.0)


def _charge_time_min(cycle_df: pd.DataFrame) -> float:
    # NB: 'discharge' contains 'charge' — must exclude it explicitly
    status = cycle_df["Status"].str.lower()
    ch = cycle_df[status.str.contains("charge") & ~status.str.contains("discharge")]
    if len(ch) < 2:
        return float("nan")
    t = ch["Time (s)"].to_numpy(float)
    return float((t[-1] - t[0]) / 60.0)


def _interp_qv(v: np.ndarray, qd: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(v)
    return np.interp(grid, v[order], qd[order])


def process_cell(cell_id: str, cycles: dict, rates: list[float]) -> tuple[list[dict], dict | None]:
    """Per-cycle tidy rows + early-cycle feature dict for one cell."""
    skip = SKIP_FIRST_CYCLES.get(cell_id, 0)
    cyc_nums = sorted(cycles.keys())[skip:]

    rows, charge_times = [], {}
    qv = {}
    for n in cyc_nums:
        df = cycles[n]
        qd = _qd_of_cycle(df)
        if not np.isfinite(qd) or qd <= 0:
            continue
        rows.append({"cycle_index": int(n), "discharge_capacity_ah": qd})
        if n <= CYCLE_MAX:
            charge_times[n] = _charge_time_min(df)
        if n in (DQ_EARLY, DQ_LATE):
            qv[n] = _discharge_qv(df)

    feats = None
    if DQ_EARLY in qv and DQ_LATE in qv and all(len(qv[k][0]) > 10 for k in qv):
        (v10, q10), (v100, q100) = qv[DQ_EARLY], qv[DQ_LATE]
        lo = max(v10.min(), v100.min()) + 0.01
        hi = min(v10.max(), v100.max()) - 0.01
        grid = np.linspace(lo, hi, V_GRID_N)
        dq = _interp_qv(v100, q100, grid) - _interp_qv(v10, q10, grid)
        cycs = np.array([r["cycle_index"] for r in rows], float)
        caps = np.array([r["discharge_capacity_ah"] for r in rows], float)
        mask = (cycs >= SLOPE_LO) & (cycs <= CYCLE_MAX)

        def _at(n):
            i = int(np.argmin(np.abs(cycs - n)))
            return float(caps[i])

        cap2, cap100 = _at(SLOPE_LO), _at(DQ_LATE)
        ct = [x for x in charge_times.values() if np.isfinite(x)]
        feats = {
            "cell_id": f"HUST_{cell_id}",
            "var_dQ_100_10": float(np.log10(max(np.var(dq), 1e-12))),
            "min_dQ_100_10": float(np.log10(max(abs(dq.min()), 1e-12))),
            "slope_2_100": float(np.polyfit(cycs[mask], caps[mask], 1)[0]),
            "capacity_cycle_2": cap2,
            "capacity_cycle_100": cap100,
            "cap_ratio_100_2": cap100 / cap2 if cap2 else float("nan"),
            "mean_charge_time_2_100": float(np.mean(ct)) if ct else float("nan"),
        }

    cell = cm.SEVERSON_CELL           # same commercial cell (phase-0 verified)
    s1, s2, s3 = rates
    for r in rows:
        r.update({
            "cell_id": f"HUST_{cell_id}",
            "study": STUDY,
            "chemistry": cell.chemistry,
            "nominal_capacity_ah": cell.nominal_capacity_ah,
            "temperature_c": TEMPERATURE_C,
            "dod_window": "0-100%",
            "c_rate_charge": CHARGE_POLICY,
            "c_rate_discharge": f"{s1:g}C/{s2:g}C/{s3:g}C/1C stages",
        })
    return rows, feats


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = json.loads(RATES_JSON.read_text())
    all_rows, all_feats, skipped = [], [], []
    for i, (cell_id, payload) in enumerate(_iter_cell_pickles(), 1):
        cycles = payload[cell_id]["data"]
        rows, feats = process_cell(cell_id, cycles, rates[cell_id])
        del payload
        if not rows:
            skipped.append(cell_id)
            continue
        all_rows.extend(rows)
        if feats:
            all_feats.append(feats)
        if i % 20 == 0:
            print(f"[hust] {i}/77 cells processed")

    df = pd.DataFrame(all_rows).drop_duplicates(["cell_id", "cycle_index"])
    df = conform(df)
    # study-specific extras (mirroring the Severson parquet's columns)
    df["charge_policy_raw"] = CHARGE_POLICY
    df["charge_policy_norm"] = CHARGE_POLICY
    df["policy_group_id"] = POLICY_GROUP
    out = PROCESSED / "hust_cycles.parquet"
    df.to_parquet(out, index=False)

    fdf = pd.DataFrame(all_feats).sort_values("cell_id")
    fout = PROCESSED / "hust_features.parquet"
    fdf.to_parquet(fout, index=False)
    print(f"[hust] {df['cell_id'].nunique()} cells, {len(df):,} cycle-rows -> {out}")
    print(f"[hust] {len(fdf)} cells with early-cycle features -> {fout}")
    if skipped:
        print(f"[hust] skipped (no usable cycles): {skipped}")
    return df, fdf


if __name__ == "__main__":
    build()
