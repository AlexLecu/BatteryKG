"""Tests for claim-vs-claim reconciliation (verdicts, units, parsing, loading)."""
import pytest

from src.kg.spec_consistency import (
    SpecEntry,
    _pair_entries,
    normalize_value,
    parse_spec_echo_line,
    verdict_for,
)


def _e(prop, value, frag="", conds=None, src="test"):
    return SpecEntry(property=prop, value=value, source_id=src,
                     fragment=frag, conditions=conds or {}, bound_text=frag)


# --- unit normalization -----------------------------------------------------------
def test_unit_normalization():
    assert normalize_value(1500, "mA", "std_charge_current_a") == pytest.approx(1.5)
    assert normalize_value(3400, "mAh", "nominal_capacity_ah") == pytest.approx(3.4)
    assert normalize_value(45, "min", "std_charge_time_h") == pytest.approx(0.75)
    assert normalize_value(4.2, "V", "charge_voltage_v") == pytest.approx(4.2)
    assert normalize_value([10, 45], "degC", "charge_temp_range_c") == [10.0, 45.0]


# --- verdict logic -------------------------------------------------------------------
def test_verdict_consistent_within_tolerance():
    v, _ = verdict_for(_e("nominal_capacity_ah", 3.35), _e("nominal_capacity_ah", 3.4))
    assert v == "consistent"                       # 1.5% < 2% tolerance


def test_verdict_conflict():
    v, d = verdict_for(_e("fast_charge_current_a", 4.0),
                       _e("fast_charge_current_a", 5.0))
    assert v == "document_variant_conflict"
    assert "4.0 vs 5.0" in d


def test_verdict_condition_mismatch_distinct_from_value_mismatch():
    a = _e("cycle_life_cycles", 1000.0, conds={"discharge_c_rate": 5.0})
    b = _e("cycle_life_cycles", 1000.0, conds={"discharge_c_rate": 10.0})
    v, d = verdict_for(a, b)
    assert v == "condition_mismatch"
    assert "discharge_c_rate" in d


def test_verdict_range_values():
    a = _e("charge_temp_range_c", [10.0, 45.0])
    b = _e("charge_temp_range_c", [0.0, 40.0])
    assert verdict_for(a, b)[0] == "document_variant_conflict"
    assert verdict_for(a, _e("charge_temp_range_c", [10.0, 45.0]))[0] == "consistent"


def test_bound_satisfaction_one_sided_only():
    # point 45 g satisfies a stated upper bound 'Max. 47.5g'
    bound = _e("mass_g", 47.5, frag="Weight (Max.): 47.5g")
    point = _e("mass_g", 45.0, frag="Netweight: 45g")
    assert verdict_for(bound, point)[0] == "consistent"
    # but TWO rival 'Max.' values are a conflict, not satisfaction
    bound2 = _e("mass_g", 48.5, frag="Weight: less than 48.5g")
    assert verdict_for(bound, bound2)[0] == "document_variant_conflict"


def test_minutes_do_not_trigger_lower_bound():
    # regression: '15 min' (minutes) must not read as 'minimum'
    a = _e("fast_charge_current_a", 4.0, frag="fast charge current: 4A to 3.6V CCCV, 15 min")
    b = _e("fast_charge_current_a", 5.0, frag="fast charge current: 5A to 3.6v CCCV, 15min")
    assert verdict_for(a, b)[0] == "document_variant_conflict"


def test_pairing_and_gaps():
    ea = [_e("cycle_life_cycles", 300.0), _e("cycle_life_cycles", 200.0)]
    eb = [_e("cycle_life_cycles", 200.0)]
    pairs, gaps = _pair_entries(ea, eb)
    # value-proximity pairing: 200<->200; 300 left as an a_only gap
    assert len(pairs) == 1 and pairs[0][0].value == 200.0
    assert len(gaps) == 1 and gaps[0][0] == "a_only" and gaps[0][1].value == 300.0


# --- spec-echo line parsing -----------------------------------------------------------
def test_parse_a123_fast_charge_line():
    entries = parse_spec_echo_line(
        "Recommended fast charge current: 5A to 3.6v CCCV, 15min", "lygte_x")
    assert len(entries) == 1
    assert entries[0].property == "fast_charge_current_a"
    assert entries[0].value == pytest.approx(5.0)


def test_parse_lg_dual_cycle_life_line():
    line = ("Cycle life: 300 at 10A, 200 at 20A both with 4A charge, "
            "ramaning capacity minimum 70%")
    entries = parse_spec_echo_line(line, "lygte_x")
    assert [(e.value, e.conditions["discharge_current_a"]) for e in entries] == \
        [(300.0, 10.0), (200.0, 20.0)]
    assert all(e.conditions["retention_pct"] == 70.0 for e in entries)


def test_parse_a123_cycle_life_line():
    entries = parse_spec_echo_line(
        "Cycle life at 10C discharge, 100% DOD: Over 1,000 cycles", "lygte_x")
    assert entries[0].value == pytest.approx(1000.0)
    assert entries[0].conditions == {"discharge_c_rate": 10.0, "dod_pct": 100.0}


def test_parse_panasonic_label_line():
    entries = parse_spec_echo_line(
        "Cut off at 2.75V, full charge 4.35V (Wrong value)", "lygte_x")
    props = {e.property: e.value for e in entries}
    assert props == {"discharge_cutoff_v": 2.75, "charge_voltage_v": 4.35}


def test_parse_standard_charge_with_ma():
    entries = parse_spec_echo_line("Standard charge: 1500mA, 4.2V, 50mA", "lygte_x")
    assert entries[0].property == "std_charge_current_a"
    assert entries[0].value == pytest.approx(1.5)          # mA -> A
    assert entries[0].conditions["charge_voltage_v"] == pytest.approx(4.2)


# --- integration: mandated findings + idempotent loading -------------------------------
@pytest.mark.integration
def test_known_cases_and_idempotency(neo4j_driver):
    from src.kg.spec_consistency import compute_spec_consistency
    rows1 = compute_spec_consistency(driver=neo4j_driver, write_outputs=False)

    def has(model, prop, verdict, needle=""):
        return any(r["model"] == model and r["property"] == prop
                   and r["verdict"] == verdict and needle in r["detail"]
                   for r in rows1)

    assert has("A123 APR18650M1A", "fast_charge_current_a",
               "document_variant_conflict")                          # 4A vs 5A
    assert has("A123 APR18650M1A", "cycle_life_cycles",
               "condition_mismatch", "discharge_c_rate")             # 5C vs 10C
    assert has("Panasonic NCR18650B", "charge_temp_range_c",
               "document_variant_conflict")                          # +10..45 vs 0..40
    assert has("LG Chem 18650HG2", "cycle_life_cycles",
               "condition_mismatch", "retention_pct")                # 60% vs 70%
    assert has("Panasonic NCR18650B", "cycle_life_cycles",
               "coverage_gap", "VERIFIED omission")                  # marketing sheet

    def count():
        with neo4j_driver.session() as s:
            return s.run("MATCH (d:Discrepancy {kind:'claim_vs_claim'}) "
                         "RETURN count(d) AS c").single()["c"]

    n1 = count()
    compute_spec_consistency(driver=neo4j_driver, write_outputs=False)
    assert count() == n1 == len(rows1)             # clean-rebuild -> stable count
