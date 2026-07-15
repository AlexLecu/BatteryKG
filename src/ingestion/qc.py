"""Dataset-agnostic quality-control and cycle-life analysis.

Operates purely on the common tidy schema (see schema.py), so it works
identically for every study. Used by the exploration notebook to compute
cycle life to 80 % of nominal capacity and to flag anomalous / incomplete
cells.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# End-of-life threshold: 80 % of nominal capacity (standard EOL definition).
EOL_FRACTION = 0.80
# Cells are typically cycled *until* they hit 80 %, so the last recorded point
# sits right at the threshold (e.g. 0.880-0.883 Ah for a 1.1 Ah cell). Allow a
# small relative tolerance so a cell terminated at ~80 % counts as reaching EOL
# rather than being mislabelled "incomplete".
EOL_TOL = 0.01  # 1 %


# Flags that indicate a genuine data anomaly (as opposed to a series that is
# merely short / incomplete). Used to propagate an `is_anomalous` boolean.
ANOMALY_FLAGS = frozenset({
    "capacity_gt_nominal", "low_initial_capacity",
    "capacity_spike_down", "non_finite_capacity",
})


def is_anomalous(flags: str) -> bool:
    """True if a cell carries at least one genuine-anomaly flag."""
    return bool(set(flags.split(",")) & ANOMALY_FLAGS) if flags else False


@dataclass
class CellLife:
    cell_id: str
    study: str
    chemistry: str | None
    nominal_capacity_ah: float | None
    n_cycles: int
    initial_capacity_ah: float          # median of first few cycles (robust to cycle-1 noise)
    final_capacity_ah: float
    eol_basis: str                      # which definition the primary columns use
    # --- cycle life under BOTH end-of-life definitions ---
    cycle_life_nominal: float | None    # cycle crossing 0.8 * nominal capacity
    reached_eol_nominal: bool
    cycle_life_initial: float | None    # cycle crossing 0.8 * initial measured capacity
    reached_eol_initial: bool
    # --- primary (nominal if known, else initial) for backward compatibility ---
    eol_threshold_ah: float | None
    cycle_life_80pct: float | None
    reached_eol: bool
    flags: str                          # comma-separated QC flags ("" if clean)
    is_anomalous: bool                  # any genuine-anomaly flag present


def _first_crossing(cycles: np.ndarray, caps: np.ndarray, threshold: float) -> float | None:
    """Cycle index where a (noisy, non-monotonic) capacity curve first falls
    below `threshold`, linearly interpolated between the bracketing cycles.

    Returns None if the curve never crosses (cell has not reached EOL).
    """
    below = caps <= threshold
    if not below.any():
        return None
    j = int(np.argmax(below))          # first index below threshold
    if j == 0:
        return float(cycles[0])        # already below at first cycle
    c0, c1 = caps[j - 1], caps[j]
    n0, n1 = cycles[j - 1], cycles[j]
    if c0 == c1:
        return float(n1)
    frac = (c0 - threshold) / (c0 - c1)
    return float(n0 + frac * (n1 - n0))


def _life_at_threshold(
    cycles: np.ndarray, smoothed: np.ndarray, final_cap: float, threshold: float
) -> float | None:
    """Interpolated crossing of `threshold`, with a termination-tolerance
    fallback: a cell cycled to ~80 % may not dip strictly below on the smoothed
    curve, but if its last *measured* capacity is within EOL_TOL of threshold,
    EOL is reached at the final cycle."""
    life = _first_crossing(cycles, smoothed, threshold)
    if life is None and final_cap <= threshold * (1 + EOL_TOL):
        life = float(cycles[-1])
    return life


def compute_cell_life(
    cell_df: pd.DataFrame,
    smooth_window: int = 5,
    init_window: int = 5,
) -> CellLife:
    """Compute cycle-life summary for a single cell's per-cycle dataframe.

    - Capacity is lightly smoothed (rolling median) before threshold crossing
      so a single noisy dip does not prematurely trigger EOL.
    - EOL threshold prefers 0.8 * nominal_capacity; if nominal is unknown it
      falls back to 0.8 * initial measured capacity.
    """
    d = cell_df.sort_values("cycle_index")
    cycles = d["cycle_index"].to_numpy(dtype=float)
    caps = d["discharge_capacity_ah"].to_numpy(dtype=float)

    cell_id = str(d["cell_id"].iloc[0])
    study = str(d["study"].iloc[0])
    chem = d["chemistry"].iloc[0]
    chem = None if pd.isna(chem) else str(chem)
    nominal = d["nominal_capacity_ah"].iloc[0]
    nominal = None if pd.isna(nominal) else float(nominal)

    n = len(caps)
    init_cap = float(np.nanmedian(caps[: min(init_window, n)]))
    final_cap = float(caps[-1])
    max_cap = float(np.nanmax(caps))

    # rolling-median smoothing (centered) suppresses isolated spikes so a single
    # noisy point neither triggers nor hides the EOL crossing.
    smoothed = (
        pd.Series(caps)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # --- both EOL definitions ---
    # (a) 0.8 * nominal datasheet capacity  (b) 0.8 * initial measured capacity
    thr_initial = EOL_FRACTION * init_cap
    life_initial = _life_at_threshold(cycles, smoothed, final_cap, thr_initial)

    if nominal is not None and nominal > 0:
        thr_nominal = EOL_FRACTION * nominal
        life_nominal = _life_at_threshold(cycles, smoothed, final_cap, thr_nominal)
        basis = "nominal"
        primary_life, primary_thr = life_nominal, thr_nominal
    else:
        thr_nominal = None
        life_nominal = None
        basis = "initial_capacity"
        primary_life, primary_thr = life_initial, thr_initial

    flags = flag_cell(cycles, caps, nominal, init_cap, max_cap, primary_life)
    flag_str = ",".join(flags)

    return CellLife(
        cell_id=cell_id,
        study=study,
        chemistry=chem,
        nominal_capacity_ah=nominal,
        n_cycles=int(n),
        initial_capacity_ah=init_cap,
        final_capacity_ah=final_cap,
        eol_basis=basis,
        cycle_life_nominal=life_nominal,
        reached_eol_nominal=life_nominal is not None,
        cycle_life_initial=life_initial,
        reached_eol_initial=life_initial is not None,
        eol_threshold_ah=float(primary_thr) if primary_thr is not None else None,
        cycle_life_80pct=primary_life,
        reached_eol=primary_life is not None,
        flags=flag_str,
        is_anomalous=is_anomalous(flag_str),
    )


def flag_cell(
    cycles: np.ndarray,
    caps: np.ndarray,
    nominal: float | None,
    init_cap: float,
    max_cap: float,
    life: float | None,
) -> list[str]:
    """Heuristic anomaly / incompleteness flags for a single cell.

    Thresholds are deliberately loose so that *normal* variation — including the
    periodic low-rate diagnostic cycles present in the Severson data, which read
    a few percent higher than the fast-cycling capacity — is not flagged; only
    physically implausible values (gross spikes, sub-nominal starts) are.
    """
    flags: list[str] = []
    n = len(caps)

    if n < 50:
        flags.append("short_series")             # too few cycles to characterise fade
    if not np.all(np.isfinite(caps)):
        flags.append("non_finite_capacity")

    # capacity should be physically plausible vs nominal
    if nominal is not None and nominal > 0:
        if max_cap > 1.2 * nominal:
            flags.append("capacity_gt_nominal")  # gross spike — unit / parse / meas. error
        if init_cap < 0.5 * nominal:
            flags.append("low_initial_capacity")

    # a big single-cycle drop then recovery = outlier, not true fade
    if n > 2 and init_cap > 0:
        if np.nanmin(np.diff(caps)) < -0.15 * init_cap:
            flags.append("capacity_spike_down")

    if life is None:
        flags.append("no_eol_reached")           # incomplete: never reached 80 %

    return flags


def cycle_life_table(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Per-cell cycle-life summary table for a whole study dataframe."""
    rows = [
        asdict(compute_cell_life(g, **kwargs))
        for _, g in df.groupby("cell_id", sort=True)
    ]
    return pd.DataFrame(rows)
