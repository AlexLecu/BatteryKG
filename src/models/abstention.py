"""Coverage-gated abstention: risk-coverage analysis after CV.

Sweep an abstention threshold over `coverage_train` (behavior view): at each
threshold, keep only cells whose graph coverage into their training fold meets
it, and measure prediction error on the retained set vs the fraction retained.

The curve only means something if it beats RANDOM abstention: at each retention
fraction we also compute the mean error of many random subsets of the same
size (seeded). If coverage-gating tracks the random curve, coverage carries no
error signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.common import SEED

N_RANDOM_DRAWS = 500


def risk_coverage_sweep(coverage: np.ndarray, y_true_log: np.ndarray,
                        y_pred_log: np.ndarray,
                        n_random: int = N_RANDOM_DRAWS,
                        seed: int = SEED) -> pd.DataFrame:
    """Risk-coverage table over all distinct coverage thresholds.

    Risk = RMSE in cycles (back-transformed) on the retained cells.
    Columns: threshold, n_retained, frac_retained, rmse_retained, rmse_random.
    """
    coverage = np.asarray(coverage, float)
    t = 10.0 ** np.asarray(y_true_log, float)
    p = 10.0 ** np.asarray(y_pred_log, float)
    sq = (t - p) ** 2
    n = len(sq)
    rng = np.random.default_rng(seed)

    rows = []
    # thresholds: each distinct coverage value (ascending) -> retention shrinks
    for thr in np.unique(coverage):
        keep = coverage >= thr
        m = int(keep.sum())
        if m == 0:
            continue
        rmse_kept = float(np.sqrt(sq[keep].mean()))
        # random baseline at the same retention size (skipped when n_random=0,
        # e.g. inside bootstrap loops)
        if n_random > 0:
            draws = np.empty(n_random)
            for j in range(n_random):
                idx = rng.choice(n, size=m, replace=False)
                draws[j] = np.sqrt(sq[idx].mean())
            rmse_rand = float(draws.mean())
        else:
            rmse_rand = float("nan")
        rows.append({
            "threshold": float(thr),
            "n_retained": m,
            "frac_retained": m / n,
            "rmse_retained": rmse_kept,
            "rmse_random": rmse_rand,
        })
    return pd.DataFrame(rows)


# AURC is integrated over a COMMON retention range so gates with different
# threshold grids are comparable; below ~20 % retention the retained-set RMSE
# is noise from a handful of cells, so that region is excluded by default.
AURC_RANGE = (0.20, 1.0)


def aurc(frac_retained, risk, lo: float = AURC_RANGE[0], hi: float = AURC_RANGE[1],
         grid_points: int = 161) -> float:
    """Area under the risk-coverage curve, normalised to a mean risk.

    Interpolates risk(frac) onto a fixed grid over [lo, hi] and averages, so
    the value reads as 'average RMSE across retention levels' and different
    gates integrate over the same support.
    """
    frac = np.asarray(frac_retained, float)
    risk = np.asarray(risk, float)
    order = np.argsort(frac)
    frac, risk = frac[order], risk[order]
    grid = np.linspace(lo, hi, grid_points)
    # step-interpolate; outside the observed range clamp to the edge values
    r = np.interp(grid, frac, risk)
    return float(np.trapezoid(r, grid) / (hi - lo))
