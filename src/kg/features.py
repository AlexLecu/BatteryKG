"""Pure, unit-testable helpers for KG feature engineering.

No Neo4j dependency here — everything is plain Python/NumPy so it can be tested
without a database. Covers: charge-policy parsing, z-scoring, similarity, and
top-k coverage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Batch-3 data-format tag; not part of the charging protocol.
_NEWSTRUCTURE = "-newstructure"

# Two-step fast-charge policy: "<c1>C(<soc>%)-<c2>C"
#   e.g. 4.8C(80%)-4.8C  =  charge at 4.8C to 80% SOC, then 4.8C to the cutoff.
_POLICY_RE = re.compile(r"^(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C$")


@dataclass(frozen=True)
class PolicyFeatures:
    c_rate_1: float             # first-step C-rate
    c_rate_2: float             # second-step C-rate
    soc_transition_pct: float   # SOC at which the step changes


def normalize_policy(policy: str | None) -> str | None:
    """Strip the '-newstructure' data-format tag (protocol-irrelevant)."""
    if not isinstance(policy, str):
        return None
    return policy.replace(_NEWSTRUCTURE, "").strip() or None


def parse_policy(policy: str | None) -> PolicyFeatures | None:
    """Parse a two-step charge policy into numeric features.

    Accepts raw or normalized strings (the '-newstructure' tag is stripped
    first). Returns None for anything that does not match the two-step format
    (e.g. the malformed 'C4(31%)-5' missing its second 'C') — callers must
    treat None as "unparseable", never guess.
    """
    norm = normalize_policy(policy)
    if norm is None:
        return None
    m = _POLICY_RE.match(norm)
    if not m:
        return None
    return PolicyFeatures(
        c_rate_1=float(m.group(1)),
        c_rate_2=float(m.group(3)),
        soc_transition_pct=float(m.group(2)),
    )


def zscore(matrix: np.ndarray) -> np.ndarray:
    """Column-wise standardization; zero-variance columns are left unscaled."""
    X = np.asarray(matrix, dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def similarity(a, b) -> float:
    """1 / (1 + Euclidean distance). 1.0 for identical vectors, →0 as they diverge."""
    d = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return 1.0 / (1.0 + d)


def topk_coverage(weights, k: int) -> float:
    """Sum of the k largest weights (fewer than k available → sum of all)."""
    w = sorted((float(x) for x in weights), reverse=True)
    return float(sum(w[:k]))


def coverage_for_edges(edges, k: int) -> tuple[float, float]:
    """Coverage under both variants from a list of (weight, same_group) edges.

    Returns (coverage_all, coverage_xgroup): the first over all neighbours, the
    second excluding same-policy-group neighbours (the honest grouped-CV value).
    """
    cov_all = topk_coverage([w for w, _ in edges], k)
    cov_xgroup = topk_coverage([w for w, same in edges if not same], k)
    return cov_all, cov_xgroup
