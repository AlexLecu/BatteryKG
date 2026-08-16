"""Krippendorff's alpha (nominal) — hand-computable cases.

The agreement figures in experiments/exp14_annotator_agreement are quoted in the
revision, so the formula gets pinned here rather than trusted.
"""
import pytest

from experiments.exp14_annotator_agreement.alpha import (
    ANNOTATED, NOT_ANNOTATED, alpha_nominal, alphas, build_codings,
)


def test_perfect_agreement():
    assert alpha_nominal([("a", "a"), ("b", "b"), ("a", "a")])["alpha"] == 1.0


def test_known_value():
    # 4 items, A = a a b b, B = a b b b
    #   D_o = 1/4 = 0.25
    #   marginals over 2*4 = 8 codings: a = 3, b = 5  ->  sum of squares = 34
    #   D_e = (8^2 - 34) / (8 * 7) = 30/56 = 0.535714
    #   alpha = 1 - 0.25 / 0.535714 = 0.5333
    r = alpha_nominal([("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")])
    assert r["n_items"] == 4
    assert r["observed"] == 0.75
    assert r["alpha"] == pytest.approx(0.533, abs=1e-3)


def test_systematic_disagreement_is_not_positive():
    r = alpha_nominal([("a", "b")] * 5)
    assert r["alpha"] <= 0.0


def test_empty_is_none():
    assert alpha_nominal([])["alpha"] is None


class _C:
    """Minimal stand-in for common.Claim — only `property` is read."""
    def __init__(self, prop):
        self.property = prop


def test_build_codings_marks_unmatched_sides():
    pairs = [(_C("nominal_capacity_ah"), _C("nominal_capacity_ah")),
             (_C("cycle_life_cycles"), _C("cycle_life_cycles"))]
    ref_only, sec_only = [_C("mass_g")], [_C("diameter_mm"), _C("height_mm")]
    c = build_codings(pairs, ref_only, sec_only)

    assert len(c["property"]) == 5              # 2 matched + 1 + 2 unmatched
    assert len(c["naming"]) == 2                # naming is matched pairs only
    assert ("mass_g", NOT_ANNOTATED) in c["property"]
    assert (NOT_ANNOTATED, "diameter_mm") in c["property"]
    assert c["presence"].count((ANNOTATED, ANNOTATED)) == 2


def test_alphas_naming_ignores_unmatched():
    pairs = [(_C("mass_g"), _C("mass_g")), (_C("cycle_life_cycles"),
                                            _C("cycle_life_cycles"))]
    out = alphas(pairs, [_C("x")], [_C("y")])
    assert out["n_items"] == 4
    assert out["alpha_naming"]["n_items"] == 2
    assert out["alpha_naming"]["alpha"] == 1.0   # two categories, both agreed
    assert out["alpha_property"]["alpha"] < 1.0  # the two unmatched cost it


def test_single_category_is_undefined_not_one():
    """No variance -> D_e = 0 -> alpha is 0/0. Report it as undefined rather
    than as perfect reliability, which is what a single category cannot show."""
    r = alpha_nominal([("mass_g", "mass_g")] * 4)
    assert r["observed"] == 1.0
    assert r["d_e"] == 0.0
    assert r["alpha"] is None
