"""Early-cycle scalar features from the Severson .mat files.

HARD LEAKAGE BOUNDARY: every feature is computed from cycles <= 100 ONLY. No
information from cycle 101 onward may enter a feature (that would leak the label,
since cycle life ranges 147-2236). The boundary is enforced in code
(`compute_early_features` asserts it) and the reader slices inputs to <=100
before they ever reach the math.

Features (per cell), written to data/processed/severson_features.parquet:
  var_dQ_100_10   log10 variance of dQ(V) = Qdlin(cycle100) - Qdlin(cycle10)
                  [the canonical Severson "variance" feature]
  min_dQ_100_10   log10 |min| of the same dQ(V) curve
  slope_2_100     linear slope of discharge capacity over cycles 2-100
  capacity_cycle_2, capacity_cycle_100, cap_ratio_100_2
  mean_charge_time_2_100   mean charge time over cycles 2-100 (if available)

Run:  python -m src.ingestion.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("features needs h5py: pip install h5py") from e

from src.config import PROCESSED
from src.ingestion.severson import (
    BATCH_FILES,
    CONTINUATIONS,
    EXCLUDE_BATCH1,
    EXCLUDE_BATCH3,
)

CYCLE_MAX = 100          # <<< hard leakage boundary >>>
DQ_EARLY, DQ_LATE = 10, 100
SLOPE_LO = 2

FEATURE_COLUMNS = [
    "var_dQ_100_10", "min_dQ_100_10", "slope_2_100",
    "capacity_cycle_2", "capacity_cycle_100", "cap_ratio_100_2",
    "mean_charge_time_2_100",
]


def _at(cyc: np.ndarray, arr: np.ndarray, n: int) -> float:
    """Value of `arr` at cycle number `n` (exact, else nearest)."""
    idx = np.where(cyc == n)[0]
    j = int(idx[0]) if len(idx) else int(np.argmin(np.abs(cyc - n)))
    return float(arr[j])


def compute_early_features(
    cyc, qd, ct, qdlin_early, qdlin_late,
    early: int = DQ_EARLY, late: int = DQ_LATE, slope_lo: int = SLOPE_LO,
) -> dict:
    """Compute the scalar features from a single cell's early-cycle arrays.

    `cyc`, `qd`, `ct` may be full arrays; only cycles in [slope_lo, 100] are used.
    `qdlin_early` / `qdlin_late` are the interpolated discharge curves at cycles
    `early` / `late`. Raises AssertionError on any attempt to use a cycle > 100.
    """
    # --- hard leakage boundary (tripwire) ---
    assert early <= CYCLE_MAX and late <= CYCLE_MAX, (
        f"leakage: dQ cycles ({early},{late}) exceed boundary {CYCLE_MAX}")

    cyc = np.asarray(cyc, dtype=float)
    qd = np.asarray(qd, dtype=float)
    mask = (cyc >= slope_lo) & (cyc <= CYCLE_MAX)
    used = cyc[mask]
    assert used.size > 0 and used.max() <= CYCLE_MAX, (
        "leakage: a cycle beyond the boundary reached the feature computation")

    dQ = np.asarray(qdlin_late, dtype=float) - np.asarray(qdlin_early, dtype=float)
    var_dQ = float(np.log10(np.var(dQ)))
    min_dQ = float(np.log10(np.abs(np.min(dQ))))

    cap2 = _at(cyc, qd, slope_lo)
    cap100 = _at(cyc, qd, late)
    ratio = cap100 / cap2 if cap2 else float("nan")
    slope = float(np.polyfit(used, qd[mask], 1)[0])

    if ct is not None:
        ct = np.asarray(ct, dtype=float)
        mct = float(np.mean(ct[mask]))
    else:
        mct = float("nan")

    return {
        "var_dQ_100_10": var_dQ,
        "min_dQ_100_10": min_dQ,
        "slope_2_100": slope,
        "capacity_cycle_2": cap2,
        "capacity_cycle_100": cap100,
        "cap_ratio_100_2": ratio,
        "mean_charge_time_2_100": mct,
    }


def _read_cell(f, batch, i: int) -> dict | None:
    """Extract the <=100-cycle arrays for cell index `i` (None if incomplete)."""
    summ = f[batch["summary"][i, 0]]
    cyc = np.array(summ["cycle"]).squeeze().astype(float)
    qd = np.array(summ["QDischarge"]).squeeze().astype(float)
    ct = (np.array(summ["chargetime"]).squeeze().astype(float)
          if "chargetime" in summ else None)

    cycles = f[batch["cycles"][i, 0]]

    def qdlin(n: int):
        idx = np.where(cyc == n)[0]
        if not len(idx):
            return None
        j = int(idx[0])
        assert cyc[j] <= CYCLE_MAX, "leakage: Qdlin read beyond cycle 100"
        return np.array(f[cycles["Qdlin"][j, 0]]).squeeze().astype(float)

    q_early, q_late = qdlin(DQ_EARLY), qdlin(DQ_LATE)
    if q_early is None or q_late is None or q_early.size < 2 or q_late.size < 2:
        return None

    # slice to the boundary BEFORE anything downstream can see later cycles
    keep = cyc <= CYCLE_MAX
    return {
        "cyc": cyc[keep],
        "qd": qd[keep],
        "ct": ct[keep] if ct is not None else None,
        "qdlin_early": q_early,
        "qdlin_late": q_late,
    }


def build_features() -> pd.DataFrame:
    """Compute features for the canonical 124 cells; write the Parquet."""
    exclude = EXCLUDE_BATCH1 | EXCLUDE_BATCH3 | set(CONTINUATIONS.values())
    rows, skipped = [], []
    for bnum, path in BATCH_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}; run: python -m src.ingestion.download severson")
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            for i in range(batch["summary"].shape[0]):
                cell_id = f"b{bnum}c{i}"
                if cell_id in exclude:
                    continue
                rec = _read_cell(f, batch, i)
                if rec is None:
                    skipped.append(cell_id)
                    continue
                feats = compute_early_features(**rec)
                feats["cell_id"] = cell_id
                rows.append(feats)

    df = pd.DataFrame(rows)[["cell_id"] + FEATURE_COLUMNS].sort_values("cell_id")
    out = PROCESSED / "severson_features.parquet"
    df.to_parquet(out, index=False)
    print(f"[features] {len(df)} cells -> {out}")
    if skipped:
        print(f"[features] skipped (no cycle 10/100): {skipped}")
    return df


if __name__ == "__main__":
    build_features()
