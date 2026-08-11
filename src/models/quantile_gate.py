"""Quantile-referenced abstention: the gate as a percentile of its own bank.

The original gate compared a test cell's `behavior_coverage_train` against an
absolute constant (4.08) chosen on Severson. That constant is a distance in
Severson's z-scored feature scaling, so it does not transfer: experiment 05
showed HUST cannot reach it at any bank size, because HUST's cells are ~2.6x
more spread out in those units than Severson's are.

The deployed rule is therefore not "is coverage above 4.08?" but "is this cell
as well covered as the bank's own members are?". Concretely:

    threshold = q-th percentile of the bank's leave-one-out coverage
                distribution — each bank cell's coverage against the rest of
                the bank, with same-policy edges and the cell itself excluded

which is scale-free (it re-derives itself from whatever bank it is given) and
reduces to the old behaviour whenever the bank looks like Severson.

This module was experiments/exp06_quantile_gate/gate.py until the rule became
the deployed one; it lives in src/models/ now because src.models.train_final
writes its output into the serving artifacts, and src must not depend on
experiments/.

Self-exclusion here is by cell id, per the fix in app/common.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.graph_model import K_GRAPH, select_train_neighbors

BEHAVIOR_VIEW = "behavior"

# The percentile that reproduces the paper's 60% in-study retention. Selected on
# Severson in-study CV alone, no HUST information used; see
# outputs/experiment_06_quantile_gate/deployed_gate.json.
Q_STAR = 41.0


def loo_coverage(bank_ids, groups: dict, edges_view: dict,
                 k: int = K_GRAPH) -> pd.Series:
    """Leave-one-out coverage of every bank cell against the rest of the bank.

    For each cell: restrict its stored neighbour list to bank members that are
    neither itself nor in its own policy group, keep the top k, sum the weights.

    Excluding the cell's own policy group mirrors what a test cell sees under
    leave-one-policy-group-out CV, where the whole group is held out together —
    without it the bank's own cells would look systematically better covered
    than any test cell could be, and the percentile would be biased high.
    """
    bank = list(bank_ids)
    bank_set = set(bank)
    out = {}
    for cid in bank:
        eligible = {j for j in bank_set if j != cid and groups[j] != groups[cid]}
        sel = select_train_neighbors(edges_view.get(cid, []), eligible, k)
        out[cid] = float(sum(w for _, w in sel))
    return pd.Series(out, name="loo_coverage")


def threshold_at_q(loo_cov, q: float) -> float:
    """Absolute coverage value sitting at the q-th percentile of the bank's LOO
    distribution. q=0 keeps everything; q=100 keeps almost nothing."""
    return float(np.percentile(np.asarray(loo_cov, float), q))


def gate_decisions(test_coverage, loo_cov, q: float) -> np.ndarray:
    """True = retain. Abstains when coverage falls below the bank's q-th percentile."""
    return np.asarray(test_coverage, float) >= threshold_at_q(loo_cov, q)


def fold_thresholds(df: pd.DataFrame, edges_view: dict, q: float,
                    k: int = K_GRAPH, group_col: str = "policy_group_id",
                    id_col: str = "cell_id") -> dict:
    """Per-fold quantile thresholds under leave-one-policy-group-out CV.

    Each fold's threshold is derived from that fold's TRAINING bank only — no
    test cell contributes to the percentile that judges it.
    """
    from src.models.common import lopgo_folds
    groups = dict(zip(df[id_col], df[group_col]))
    out = {}
    for gid, train, _test in lopgo_folds(df, group_col):
        train_ids = df.iloc[train][id_col].tolist()
        out[gid] = threshold_at_q(loo_coverage(train_ids, groups, edges_view, k), q)
    return out


def fold_loo_distributions(df: pd.DataFrame, edges_view: dict, k: int = K_GRAPH,
                           group_col: str = "policy_group_id",
                           id_col: str = "cell_id") -> dict:
    """{fold_id: LOO coverage Series of that fold's training bank}. Computed once
    and reused across the whole q sweep."""
    from src.models.common import lopgo_folds
    groups = dict(zip(df[id_col], df[group_col]))
    return {gid: loo_coverage(df.iloc[train][id_col].tolist(), groups, edges_view, k)
            for gid, train, _t in lopgo_folds(df, group_col)}


def keep_mask_for_q(pred: pd.DataFrame, dists: dict, q: float,
                    coverage_col: str = "behavior_coverage_train") -> np.ndarray:
    """Retain decisions for every row of `pred` at percentile q, fold by fold."""
    keep = np.zeros(len(pred), bool)
    cov = pred[coverage_col].to_numpy(float)
    folds = pred["fold"].to_numpy()
    for gid, loo in dists.items():
        m = folds == gid
        keep[m] = cov[m] >= threshold_at_q(loo, q)
    return keep
