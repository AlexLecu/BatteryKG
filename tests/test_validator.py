"""Unit tests for the deterministic Validator stage."""
from src.agents.validator import (
    consensus,
    normalize_tolerance,
    schema_gate,
    unit_sane,
    validate_claims,
    value_in_source,
    _doc_numbers,
)


def _c(prop, value, unit=None, **kw):
    return {"property": prop, "value": value, "unit": unit,
            "stated_conditions": {"text": "unspecified"}, **kw}


# --- stage 1: schema gate -----------------------------------------------------
def test_schema_gate_passes_scalar_and_range():
    assert schema_gate(_c("mass_g", 47.0))[0][0]["value"] == 47.0
    assert schema_gate(_c("charge_temp_range_c", [10, 45]))[1] is None


def test_schema_gate_unnests_nested_ranges():
    claims, reason = schema_gate(_c("storage_temp_range_c",
                                    [[-20, 60], [-20, 45], [-20, 20]]))
    assert reason is None
    assert [c["value"] for c in claims] == [[-20, 60], [-20, 45], [-20, 20]]
    assert all(c["validator_note"] == "unnested" for c in claims)


def test_schema_gate_unnests_flat_multivalue():
    claims, _ = schema_gate(_c("cycle_life_cycles", [300, 200, 100]))
    assert [c["value"] for c in claims] == [300, 200, 100]


def test_schema_gate_rejects_malformed():
    assert schema_gate({"value": 1.0})[1] == "missing_field"
    assert schema_gate(_c("mass_g", "forty-seven"))[1] == "malformed_value"
    assert schema_gate(_c("mass_g", None))[1] == "malformed_value"


# --- stage 2: value-in-source ----------------------------------------------------
def test_value_in_source_formatting_and_conversion():
    nums = _doc_numbers("Cycle life Over 1,000 cycles. Charge 45 min. 3350mAh.")
    assert value_in_source(_c("cycle_life_cycles", 1000), nums)     # "1,000" == 1000
    assert value_in_source(_c("nominal_capacity_ah", 3.35, "Ah"), nums)  # x1000 prefix
    # silent unit conversion 45 min -> 0.75 h: 0.75 not in source -> rejected
    assert not value_in_source(_c("std_charge_time_h", 0.75, "h"), nums)


def test_value_in_source_negative_values():
    # regression: -50 in "range -50°C to +60°C" must be found (abs-matching),
    # including when the PDF uses a Unicode minus
    nums = _doc_numbers("Storage temperature range -50°C to +60°C")
    assert value_in_source(_c("storage_temp_range_c", [-50, 60], "degC"), nums)
    nums_u = _doc_numbers("Storage temperature range −50°C to +60°C")  # U+2212
    assert value_in_source(_c("storage_temp_range_c", [-50, 60], "degC"), nums_u)


# --- stage 3: unit sanity -----------------------------------------------------------
def test_unit_sanity_bounds():
    assert unit_sane(_c("nominal_capacity_ah", 3.35, "Ah"))
    assert unit_sane(_c("nominal_capacity_ah", 3350, "mAh"))    # unit-scaled
    assert not unit_sane(_c("nominal_capacity_ah", 99, "Ah"))   # implausible
    assert not unit_sane(_c("nominal_voltage_v", 12, "V"))
    assert unit_sane(_c("cycle_life_cycles", 300, "cycles"))
    assert not unit_sane(_c("cycle_life_cycles", 50000, "cycles"))
    assert unit_sane(_c("unknown_new_property", 1e9))           # no bounds -> pass


# --- stage 4: tolerance normalizer -----------------------------------------------------
def test_normalize_tolerance():
    doc = "Max. Charge Voltage: 4.20 ± 0.05V"
    claim = _c("max_charge_voltage_v", [4.15, 4.25], "V")
    assert normalize_tolerance(claim, doc)
    assert claim["value"] == 4.2
    # a genuine range (not a tolerance in the source) is left alone
    claim2 = _c("charge_temp_range_c", [10, 45], "degC")
    assert not normalize_tolerance(claim2, doc)
    assert claim2["value"] == [10, 45]


# --- stage 5: consensus -----------------------------------------------------------------
def test_consensus_keeps_majority_drops_singletons():
    r1 = [_c("mass_g", 47.0, "g"), _c("nominal_voltage_v", 3.6, "V")]
    r2 = [_c("mass_g", 47.0, "g")]
    r3 = [_c("mass_g", 47.0, "g"), _c("cycle_life_cycles", 300, "cycles")]
    kept = consensus([r1, r2, r3])
    props = {c["property"]: c["consensus_level"] for c in kept}
    assert props == {"mass_g": 3}          # voltage and cycle life: 1 run each

def test_consensus_matches_across_units():
    r1 = [_c("nominal_capacity_ah", 3.35, "Ah")]
    r2 = [_c("nominal_capacity_ah", 3350, "mAh")]
    kept = consensus([r1, r2])
    assert len(kept) == 1 and kept[0]["consensus_level"] == 2


# --- false-rejection guard: a fully-correct set passes untouched ----------------------------
def test_correct_claims_pass_untouched():
    doc = ("=== PAGE 1 ===\nNominal capacity 1.1Ah, 3.3 V. Maximum continuous "
           "discharge 30A. Cycle life Over 1,000 cycles. Weight 39 grams. "
           "Operating temperature -30°C to +60°C.")
    claims = [
        _c("nominal_capacity_ah", 1.1, "Ah"),
        _c("nominal_voltage_v", 3.3, "V"),
        _c("max_cont_discharge_a", 30, "A"),
        _c("cycle_life_cycles", 1000, "cycles"),
        _c("mass_g", 39, "g"),
        _c("operating_temp_range_c", [-30, 60], "degC"),
    ]
    res = validate_claims(claims, doc, doc="synthetic", run="run1")
    assert res.rejected == []
    assert res.normalized == 0
    assert [c["value"] for c in res.accepted] == [c["value"] for c in claims]
