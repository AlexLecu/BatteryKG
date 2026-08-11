"""Small-sample adaptation on HUST — shared machinery.

Reuses the deployed serving path (app/artifacts + app.common maths, the
experiment-04 HUST eval table, src.models.common's fixed XGBoost config). The
only thing reimplemented here is neighbour lookup, and only because the served
version needs one change to be correct under bank augmentation — see
`GraphIndex` below.

Nothing in this module writes to app/artifacts, the KG, or src/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.common import _zscore_with, load_artifacts
from src.config import OUTPUTS
from src.models.dataset import BASE_FEATURES, TARGET
from src.models.experiment_04 import HUST_POLICY, build_hust_eval

# HUST is one cross-study policy group at a fixed charge policy (src.ingestion.hust)
HUST_GROUP = "pgH00"
HUST_CHARGE_POLICY = "5C(80%)-1C"

VIEWS = ("condition", "behavior")
OUT = OUTPUTS / "experiment_05_hust_adaptation"


# --- HUST as bank-shaped rows -------------------------------------------------
def hust_table() -> pd.DataFrame:
    """The 77 QC-clean HUST cells in neighbour-bank schema.

    `batch` is a Severson-specific covariate and is NaN for HUST, exactly as
    experiment 04 passes it at serve time.
    """
    h = build_hust_eval().copy()
    h["policy_group_id"] = HUST_GROUP
    h["charge_policy_norm"] = HUST_CHARGE_POLICY
    h["batch"] = np.nan
    for c, v in HUST_POLICY.items():
        h[c] = v
    h[TARGET] = np.log10(h["cycle_life_nominal"].astype(float))
    h["study"] = "hust"
    return h.sort_values("cell_id").reset_index(drop=True)


def severson_bank() -> pd.DataFrame:
    """The deployed 120-cell neighbour bank."""
    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    b = art["bank"].copy()
    b["study"] = "severson"
    return b.sort_values("cell_id").reset_index(drop=True)


BANK_COLS = (["cell_id", "policy_group_id", "charge_policy_norm", "batch",
              "cycle_life_nominal", TARGET, "study"] + BASE_FEATURES)


def augmented_bank(sev: pd.DataFrame, hust: pd.DataFrame,
                   add_ids: list[str]) -> pd.DataFrame:
    """Severson bank + the labelled HUST cells named in `add_ids`.

    Sorted by cell_id so that neighbour ties break on cell_id ascending, which
    is the tie-break the served `view_neighbors` uses.
    """
    cols = [c for c in dict.fromkeys(BANK_COLS) if c in sev.columns or c in hust.columns]
    add = hust[hust["cell_id"].isin(add_ids)]
    out = pd.concat([sev[cols], add[cols]], ignore_index=True)
    return out.sort_values("cell_id").reset_index(drop=True)


# --- neighbour lookup ----------------------------------------------------------
class GraphIndex:
    """Vectorised neighbour/graph-feature computation against one bank.

    Identical maths to `app.common.view_neighbors` (same z-score using the
    FROZEN deployed scaler, same weight 1/(1+d), same top-k, same tie-break),
    vectorised over many queries at once and exposing self-exclusion by
    cell_id.

    This class originally differed from the served path: that path used to drop
    any neighbour with weight >= 0.999999, treating "distance 0" as a proxy for
    "this is the query cell". The proxy breaks under bank augmentation — all 77
    HUST cells share one charge policy, so in the CONDITION view every HUST bank
    cell sits at distance 0 from a HUST query and would be discarded as 'self'.
    That bug is now fixed in app/common.py (explicit `exclude_cell_id`, covered
    by tests/test_self_exclusion.py); the two implementations agree by
    construction, and `check_matches_served_path` asserts it at n=0.
    """

    def __init__(self, meta: dict, bank: pd.DataFrame):
        self.meta, self.bank = meta, bank.reset_index(drop=True)
        self.k = meta["k_neighbors"]
        self.ids = self.bank["cell_id"].to_numpy()
        self.groups = self.bank["policy_group_id"].to_numpy()
        self.y = self.bank[TARGET].to_numpy(float)
        self._Z = {}
        for v in VIEWS:
            sc = meta["scaler"][v]
            self._Z[v] = (_zscore_with(self.bank[sc["columns"]].to_numpy(float),
                                       sc["mean"], sc["std"]), sc["columns"])

    def _weights(self, view: str, queries: pd.DataFrame) -> np.ndarray:
        sc = self.meta["scaler"][view]
        B, cols = self._Z[view]
        Q = _zscore_with(queries[cols].to_numpy(float), sc["mean"], sc["std"])
        d = np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2)
        return 1.0 / (1.0 + d)

    def graph_features(self, queries: pd.DataFrame,
                       exclude_self: bool = False,
                       exclude_own_group: np.ndarray | None = None) -> pd.DataFrame:
        """Graph features for each row of `queries` (needs the bank feature columns).

        exclude_self        drop the bank row whose cell_id equals the query's
        exclude_own_group   boolean per query: also drop same-policy-group rows
        """
        n = len(queries)
        qid = queries["cell_id"].to_numpy()
        qgrp = queries["policy_group_id"].to_numpy()
        mask = np.zeros((n, len(self.bank)), bool)          # True = excluded
        if exclude_self:
            mask |= qid[:, None] == self.ids[None, :]
        if exclude_own_group is not None:
            eg = np.asarray(exclude_own_group, bool)
            mask |= eg[:, None] & (qgrp[:, None] == self.groups[None, :])

        out = {}
        for view in VIEWS:
            W = self._weights(view, queries)
            W = np.where(mask, -np.inf, W)
            if (np.isfinite(W).sum(1) < self.k).any():
                raise ValueError(f"fewer than k={self.k} eligible neighbours in {view} view")
            # stable sort on the cell_id-ordered bank == (weight desc, cell_id asc)
            idx = np.argsort(-W, axis=1, kind="stable")[:, :self.k]
            w = np.take_along_axis(W, idx, axis=1)
            yy = self.y[idx]
            wsum = w.sum(1)
            wmean = (w * yy).sum(1) / wsum
            wstd = np.sqrt((w * (yy - wmean[:, None]) ** 2).sum(1) / wsum)
            out[f"{view}_nbr_wmean_log_life"] = wmean
            out[f"{view}_nbr_wstd_log_life"] = wstd
            out[f"{view}_nbr_mean_weight"] = w.mean(1)
            out[f"{view}_coverage_train"] = wsum
        return pd.DataFrame(out, index=queries.index)


def design_matrix(meta: dict, rows: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model input in the exact column order the artifacts expect."""
    X = pd.concat([rows[meta["base_features"]].reset_index(drop=True),
                   gf.reset_index(drop=True)], axis=1)
    return X[meta["features"]]


# --- metrics ------------------------------------------------------------------
def rmse(t, p) -> float:
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def mape(t, p) -> float:
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - t) / t) * 100) if len(t) else float("nan")


def ape(t, p) -> np.ndarray:
    t, p = np.asarray(t, float), np.asarray(p, float)
    return np.abs(p - t) / t * 100


def check_matches_served_path(meta, bank, hust) -> None:
    """Assert GraphIndex reproduces app.common's served coverage at n=0."""
    from app.common import predict_with_gate
    art, _ = load_artifacts()
    idx = GraphIndex(meta, bank)
    mine = idx.graph_features(hust)["behavior_coverage_train"].to_numpy()
    served = []
    for _, r in hust.iterrows():
        q = {c: float(r[c]) for c in meta["behavior_features"]}
        for c in meta["base_features"]:
            q.setdefault(c, float(r[c]))
        served.append(predict_with_gate(art, q, exclude_group=None)["coverage"])
    if not np.allclose(mine, np.asarray(served), rtol=0, atol=1e-9):
        raise AssertionError("GraphIndex diverges from the served path at n=0")
