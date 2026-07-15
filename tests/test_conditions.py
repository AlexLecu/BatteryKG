"""Unit tests for the conditions-focused extraction stage (no API calls)."""
from src.agents.conditions_extractor import (
    conditions_consensus,
    page_context,
    split_pages,
    validate_conditions,
)
from src.agents.evaluation import GoldClaim, condition_over_extracted


# --- field-level consensus ------------------------------------------------------
def test_conditions_consensus_majority_and_tie():
    runs = [
        {"temperature_c": 25, "discharge_current_a": 10, "charge_mode": "CC-CV",
         "text": "at 25degC, 10A"},
        {"temperature_c": 25, "discharge_current_a": 10, "dod_pct": 100,
         "text": "25degC 10A discharge"},
        {"temperature_c": 25, "discharge_current_a": 3.25, "charge_mode": "CC-CV",
         "text": "25degC"},
    ]
    out = conditions_consensus(runs)
    assert out["temperature_c"] == 25              # 3/3
    assert out["discharge_current_a"] == 10        # 2/3 majority
    assert out["charge_mode"] == "CC-CV"           # 2/3
    assert "dod_pct" not in out                    # 1/3 -> dropped (conservative)


def test_conditions_consensus_no_majority_gives_unspecified():
    runs = [{"temperature_c": 25}, {"temperature_c": 45}, {"temperature_c": 0}]
    out = conditions_consensus(runs)
    assert "temperature_c" not in out              # three-way tie -> dropped
    assert out["text"] == "unspecified"


def test_conditions_consensus_treats_unspecified_as_absent():
    runs = [{"temperature_c": 25}, {"temperature_c": "unspecified"},
            {"temperature_c": 25}]
    out = conditions_consensus(runs)
    assert out["temperature_c"] == 25              # 2 real votes


# --- condition-side validation ------------------------------------------------------
def test_validate_conditions_rejects_numbers_not_in_source():
    page = "6.3 Cycle Life: 300 cycles at 25degC, discharge 3.25A to 2.5V."
    cond = {"temperature_c": 25, "discharge_current_a": 3.25,
            "charge_time_min": 240,               # NOT on this page -> rejected
            "text": "at 25degC"}
    kept, rejections = validate_conditions(cond, page, doc="d", claim_property="p")
    assert kept["temperature_c"] == 25 and kept["discharge_current_a"] == 3.25
    assert "charge_time_min" not in kept
    assert len(rejections) == 1
    assert rejections[0].reason == "condition_not_in_source"
    assert rejections[0].field == "charge_time_min"


def test_validate_conditions_all_rejected_becomes_unspecified():
    kept, rejections = validate_conditions({"temperature_c": 45}, "no numbers here")
    assert kept["text"] == "unspecified"
    assert len(rejections) == 1


# --- over-extraction detector ---------------------------------------------------------
def test_over_extraction_when_gold_unspecified():
    gold = GoldClaim(doc="d", cell_model="m", property="nominal_voltage_v",
                     value=3.6, unit="V", page=1,
                     conditions_text="unspecified", parsed_conditions={})
    invented = {"temperature_c": 25, "text": "at 25 degC"}    # model invents 25degC
    assert condition_over_extracted(gold, invented)
    honest = {"text": "unspecified"}
    assert not condition_over_extracted(gold, honest)


def test_over_extraction_allows_numbers_from_gold_quote():
    gold = GoldClaim(doc="d", cell_model="m", property="std_charge_current_a",
                     value=1.5, unit="A", page=1,
                     conditions_text="1.5A to 3.6V CCCV, 45 min",
                     parsed_conditions={"charge_voltage_v": 3.6, "charge_time_min": 45})
    echo = {"charge_voltage_v": 3.6, "charge_time_min": 45, "text": "to 3.6V, 45 min"}
    assert not condition_over_extracted(gold, echo)
    assert condition_over_extracted(gold, {"temperature_c": 25})   # 25 not in quote


def test_over_extraction_is_unit_blind():
    # a prediction of 0.6 A is grounded by a gold quote saying "600mA"
    gold = GoldClaim(doc="d", cell_model="m", property="std_discharge_current_a",
                     value=0.6, unit="A", page=1,
                     conditions_text="CC 600mA, end voltage (cut off) 2.0V",
                     parsed_conditions={"discharge_cutoff_v": 2.0})
    pred = {"discharge_current_a": 0.6, "discharge_cutoff_v": 2.0}
    assert not condition_over_extracted(gold, pred)


# --- page context -----------------------------------------------------------------------
def test_split_pages_and_context():
    doc = "=== PAGE 1 ===\nshort\n\n=== PAGE 2 ===\n" + ("x " * 600) + "\n\n=== PAGE 3 ===\ntail"
    pages = split_pages(doc)
    assert set(pages) == {1, 2, 3}
    # page 1 is short -> gets neighbours; page 2 is long -> stands alone
    assert "x x" in page_context(pages, 1)
    assert "tail" not in page_context(pages, 2)
    # unknown page -> whole document fallback
    assert "tail" in page_context(pages, 99)
