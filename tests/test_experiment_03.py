"""Tests for experiment 03: question generation and scoring logic."""
import pytest

from src.agents.experiment_03 import (
    classify_wrong,
    conditions_stated,
    generate_questions,
    score_value,
    source_attributed,
    uncertainty_expressed,
)


# --- question generation from the gold YAMLs -------------------------------------
def test_one_question_per_gold_claim():
    qs = generate_questions()
    assert len(qs) == 62
    assert len({q["question_id"] for q in qs}) == 62
    for q in qs:
        assert q["cell_model"] in q["question"]
        assert q["gold"]["value"] is not None
        assert q["gold"]["source_document"]


def test_multi_claim_questions_are_disambiguated():
    qs = generate_questions()
    lg_cycle = [q for q in qs if q["cell_model"] == "LG Chem 18650HG2"
                and q["property"] == "cycle_life_cycles"]
    assert len(lg_cycle) == 2
    texts = {q["question"] for q in lg_cycle}
    assert len(texts) == 2                       # distinct via the qualifier
    assert any("10 A" in t for t in texts) and any("20 A" in t for t in texts)


# --- scoring: value ---------------------------------------------------------------
def test_score_value_correct_and_unit_normalized():
    assert score_value("The nominal capacity is 3.35 Ah.", 3.35, "Ah") == "correct"
    assert score_value("It is rated at 3350 mAh.", 3.35, "Ah") == "correct"
    assert score_value("Cycle life is 1,000 cycles.", 1000, "cycles") == "correct"
    assert score_value("Range is -20 to 60 °C.", [-20, 60], "degC") == "correct"


def test_score_value_wrong_fabricated():
    assert score_value("Approximately 2500 mAh.", 3.35, "Ah") == "wrong"


def test_score_value_unit_confused():
    # right digits, wrong magnitude claimed as the big unit
    assert score_value("The capacity is 3350 Ah.", 3.35, "Ah") == "unit_confused"


def test_score_value_no_value():
    assert score_value("I do not know this cell.", 3.35, "Ah") == "no_value"


# --- scoring: conditions / source / uncertainty ------------------------------------
def test_conditions_stated():
    parsed = {"temperature_c": 25, "discharge_current_a": 10.0}
    assert conditions_stated("300 cycles at 25°C with 10A discharge", parsed) is True
    assert conditions_stated("300 cycles", parsed) is False
    assert conditions_stated("whatever", {}) is None          # gold states none


def test_source_attribution_requires_specificity():
    assert source_attributed("per the LG product specification PS-HG2, p. 6")
    assert source_attributed("see page 4 of the spec")
    assert not source_attributed("according to the datasheet")   # not specific
    assert not source_attributed("3000 mAh nominal")


def test_uncertainty_detection():
    assert uncertainty_expressed("I'm not certain, but around 3000 mAh")
    assert uncertainty_expressed("I do not have reliable information on this")
    assert not uncertainty_expressed("The capacity is 3000 mAh.")


# --- taxonomy -----------------------------------------------------------------------
def test_classify_wrong_taxonomy():
    variants = [(5.0, "A")]                     # the lygte echo's 5A variant
    # matches the OTHER document's value -> blended/variant
    assert classify_wrong("The fast charge current is 5 A.", 4.0, "A",
                          variants) == "variant_value"
    # matches nothing we hold -> fabricated_or_stale
    assert classify_wrong("It is 2.0 A.", 4.0, "A", variants) == "fabricated_or_stale"
    # refusal with uncertainty
    assert classify_wrong("I am not certain about this cell.", 4.0, "A",
                          variants) == "refused"
    # correct passes through
    assert classify_wrong("4 A fast charge.", 4.0, "A", variants) == "correct"
