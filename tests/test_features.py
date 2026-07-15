"""Unit tests for pure KG feature helpers (no database needed)."""
import numpy as np
import pytest

from src.kg.features import (
    PolicyFeatures,
    coverage_for_edges,
    normalize_policy,
    parse_policy,
    similarity,
    topk_coverage,
    zscore,
)


# --- policy-string parsing, including -newstructure variants ---------------
@pytest.mark.parametrize("s, expected", [
    ("4.8C(80%)-4.8C", PolicyFeatures(4.8, 4.8, 80.0)),
    ("1C(4%)-6C", PolicyFeatures(1.0, 6.0, 4.0)),
    ("2C(7%)-5.5C", PolicyFeatures(2.0, 5.5, 7.0)),
    # -newstructure tag must be stripped before parsing
    ("3.7C(31%)-5.9C-newstructure", PolicyFeatures(3.7, 5.9, 31.0)),
    ("5.6C(19%)-4.6C-newstructure", PolicyFeatures(5.6, 4.6, 19.0)),
    ("4.8C(80%)-4.8C-newstructure", PolicyFeatures(4.8, 4.8, 80.0)),
])
def test_parse_policy_valid(s, expected):
    assert parse_policy(s) == expected


@pytest.mark.parametrize("s", [
    "4C(31%)-5",      # the real malformed Severson string (second 'C' missing)
    "",
    None,
    "garbage",
    "4C-4C",          # no SOC transition
    "4C(80%)",        # single step only
])
def test_parse_policy_unparseable(s):
    assert parse_policy(s) is None


def test_normalize_policy_strips_suffix():
    assert normalize_policy("4.8C(80%)-4.8C-newstructure") == "4.8C(80%)-4.8C"
    assert normalize_policy("4.8C(80%)-4.8C") == "4.8C(80%)-4.8C"
    assert normalize_policy(None) is None


# --- similarity on hand-checkable vectors ----------------------------------
def test_similarity_identical_is_one():
    assert similarity([0, 0, 0], [0, 0, 0]) == 1.0
    assert similarity([2.5, -1, 3], [2.5, -1, 3]) == 1.0


def test_similarity_known_distance():
    # distance (0,0)->(3,4) = 5  ->  1/(1+5) = 1/6
    assert similarity([0, 0], [3, 4]) == pytest.approx(1 / 6)
    # distance 1 -> 0.5
    assert similarity([0], [1]) == pytest.approx(0.5)


def test_zscore_known():
    Z = zscore([[0.0], [10.0]])          # mean 5, population std 5
    assert Z.flatten() == pytest.approx([-1.0, 1.0])


def test_zscore_zero_variance_column():
    Z = zscore([[3.0, 1.0], [3.0, 5.0]])  # col0 constant -> left unscaled (0)
    assert Z[:, 0] == pytest.approx([0.0, 0.0])
    assert Z[:, 1] == pytest.approx([-1.0, 1.0])


# --- top-k coverage / coverage variants on a tiny fixed "graph" ------------
def test_topk_coverage():
    assert topk_coverage([0.1, 0.9, 0.5, 0.7], 2) == pytest.approx(1.6)
    assert topk_coverage([0.5], 3) == pytest.approx(0.5)   # fewer than k
    assert topk_coverage([], 3) == 0.0


def test_coverage_for_edges_all_and_xgroup():
    # (weight, same_policy_group)
    edges = [(1.0, True), (0.8, False), (0.6, False), (0.4, True), (0.2, False)]
    cov_all, cov_xgroup = coverage_for_edges(edges, k=2)
    assert cov_all == pytest.approx(1.0 + 0.8)      # top-2 over all
    assert cov_xgroup == pytest.approx(0.8 + 0.6)   # top-2 excluding same-group


def test_coverage_xgroup_empty_when_all_same_group():
    edges = [(1.0, True), (0.9, True)]
    cov_all, cov_xgroup = coverage_for_edges(edges, k=5)
    assert cov_all == pytest.approx(1.9)
    assert cov_xgroup == 0.0
