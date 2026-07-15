"""Unit tests for the LLM extraction harness (no API calls)."""
import pytest

from src.agents.evaluation import (
    GoldClaim,
    conditions_correct,
    evaluate_document,
    value_in_document,
    values_match,
)
from src.agents.extractor import _parse_json_claims


# --- response parsing -----------------------------------------------------------
def test_parse_known_good_response():
    raw = """```json
[
  {"property": "nominal_capacity_ah", "value": 1.1, "unit": "Ah", "page": 1,
   "stated_conditions": {"text": "unspecified"}},
  {"property": "charge_temp_range_c", "value": [10, 45], "unit": "degC", "page": 1,
   "stated_conditions": {"text": "Ambient Temperature, Charge: +10 ~ +45degC"}}
]
```"""
    claims = _parse_json_claims(raw)
    assert len(claims) == 2
    assert claims[0]["property"] == "nominal_capacity_ah"
    assert claims[1]["value"] == [10, 45]


def test_parse_fills_missing_conditions_and_rejects_garbage():
    claims = _parse_json_claims('[{"property": "mass_g", "value": 47.5}]')
    assert claims[0]["stated_conditions"]["text"] == "unspecified"
    with pytest.raises(ValueError):
        _parse_json_claims("I could not find any claims in the document.")
    with pytest.raises(ValueError):
        _parse_json_claims('[{"value": 1.0}]')     # missing property


# --- value matching ----------------------------------------------------------------
def test_values_match_exact_and_tolerance():
    assert values_match(1.1, "Ah", 1.1, "Ah")
    assert values_match(3.35, "Ah", 3350, "mAh")       # unit-scaled
    assert values_match(1000, "cycles", 1000, "cycles")
    assert not values_match(1000, "cycles", 736, "cycles")
    assert values_match([10, 45], "degC", [10, 45], "degC")
    assert not values_match([10, 45], "degC", [0, 40], "degC")


def _gold(prop, value, unit, cond_text="unspecified", parsed=None):
    return GoldClaim(doc="d", cell_model="m", property=prop, value=value,
                     unit=unit, page=1, conditions_text=cond_text,
                     parsed_conditions=parsed or {})


# --- conditions scoring ----------------------------------------------------------------
def test_conditions_unspecified_must_not_be_guessed():
    g = _gold("nominal_voltage_v", 3.6, "V")           # gold: unspecified
    assert conditions_correct(g, {"text": "unspecified"})
    assert not conditions_correct(g, {"text": "at 25degC", "temperature_c": 25})


def test_conditions_numeric_fields_must_match():
    g = _gold("cycle_life_cycles", 300, "cycles",
              cond_text="25degC, 10A discharge",
              parsed={"temperature_c": 25, "discharge_current_a": 10.0})
    ok = {"text": "at 25 degC with 10A discharge", "temperature_c": 25,
          "discharge_current_a": 10}
    assert conditions_correct(g, ok)
    wrong = {"text": "at 45 degC", "temperature_c": 45}
    assert not conditions_correct(g, wrong)


# --- full document evaluation: TP / value-tol / cond mismatch / hallucination -----
def test_evaluate_document_all_cases():
    doc_text = "=== PAGE 1 ===\nCapacity 3350 mAh typ. Cycle life 300 cycles at 25degC 10A."
    gold = [
        _gold("nominal_capacity_ah", 3.35, "Ah"),                      # TP via unit-scale
        _gold("cycle_life_cycles", 300, "cycles",
              cond_text="25degC 10A", parsed={"temperature_c": 25}),   # TP, cond wrong
        _gold("mass_g", 47.0, "g"),                                    # FN
    ]
    for g in gold:
        g.doc = "d"
    preds = [
        {"property": "nominal_capacity_ah", "value": 3350, "unit": "mAh",
         "stated_conditions": {"text": "unspecified"}},
        {"property": "cycle_life_cycles", "value": 300, "unit": "cycles",
         "stated_conditions": {"text": "at 45degC", "temperature_c": 45}},
        {"property": "internal_impedance_mohm", "value": 999, "unit": "mOhm",
         "stated_conditions": {"text": "unspecified"}},                # hallucination
    ]
    ev = evaluate_document("d", gold, preds, doc_text)
    assert (ev.tp, ev.fp, ev.fn) == (2, 1, 1)
    assert ev.conditions_ok == 1                 # capacity ok, cycle-life cond wrong
    assert ev.hallucinations == 1                # 999 not in text
    assert ev.precision == pytest.approx(2 / 3)
    assert ev.recall == pytest.approx(2 / 3)


def test_value_in_document_scaling():
    text = "Nominal Capacity: 3350mAh"
    assert value_in_document(3.35, "Ah", text)       # x1000 rescale
    assert value_in_document(3350, "mAh", text)
    assert not value_in_document(999, "mOhm", text)
