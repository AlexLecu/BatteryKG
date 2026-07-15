"""Tests for the lygte-info.dk source: parser (on a committed fixture),
conditions-matching logic, and KG loader idempotency (synthetic, namespaced)."""
import json
from pathlib import Path

import pytest

from src.ingestion.lygte import parse_review
from src.kg.discrepancy import currents_comparable

FIXTURE = (Path(__file__).parent / "fixtures" / "lygte_fixture.html").read_text(
    encoding="iso-8859-1")


# --- parser -----------------------------------------------------------------
def test_parser_extracts_explicit_statements_with_fragments():
    parsed = parse_review(FIXTURE)
    by_prop = {}
    for m in parsed["measurements"]:
        by_prop.setdefault(m["property"], []).append(m)

    temps = by_prop["measured_cell_temp_c"]
    assert {(t["value"], t["conditions"]["discharge_current_a"]) for t in temps} == \
        {(81.0, 20.0), (90.0, 30.0)}
    assert all("°C" in t["source_fragment"] or "C" in t["source_fragment"]
               for t in temps)

    sust = by_prop["measured_sustained_current_a"][0]
    assert sust["value"] == 30.0 and sust["conditions"]["duration_min"] == 2.0
    assert "two minutes" in sust["source_fragment"]

    cutoff = by_prop["test_discharge_cutoff_v"][0]
    assert cutoff["value"] == 2.8


def test_parser_records_chart_only_not_values():
    parsed = parse_review(FIXTURE)
    chart = [m for m in parsed["measurements"] if m["chart_only"]]
    props = {m["property"] for m in chart}
    assert props == {"measured_capacity_ah", "measured_energy_wh"}
    assert all(m["value"] is None for m in chart)
    assert any("Capacity.png" in m["source_fragment"] for m in chart)


def test_parser_spec_echo():
    parsed = parse_review(FIXTURE)
    assert "Nominal Capacity: 9999mAh" in parsed["spec_echo"]
    assert len(parsed["spec_echo"]) == 3


# --- conditions matching (20% rule) ----------------------------------------------
@pytest.mark.parametrize("a, b, expected", [
    (0.65, 0.65, True),      # exact match
    (0.65, 0.75, True),      # |0.10| <= 0.2 * 0.75 = 0.15 -> within 20%
    (1.0, 1.2, True),        # |0.2| <= 0.2 * 1.2 = 0.24
    (1.0, 1.3, False),       # |0.3| >  0.2 * 1.3 = 0.26
    (0.2, 5.0, False),       # low-rate spec vs high-rate test
    (None, 1.0, False),      # unstated current -> never comparable
    (1.0, None, False),
])
def test_currents_comparable(a, b, expected):
    assert currents_comparable(a, b) is expected


# --- loader idempotency (synthetic source, cleaned up) -----------------------------
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_independent_loader_idempotent(neo4j_driver, tmp_path):
    from src.kg.independent import load_independent
    payload = {
        "retrieved": "2026-01-01",
        "reviews": [{
            "cell_model": "TEST_LYGTE_CELL",
            "source_id": "lygte_test_synthetic",
            "url": "https://example.invalid/test",
            "title": "Test fixture review",
            "retrieved": "2026-01-01",
            "n_specimens": 2,
            "spec_echo": [],
            "measurements": [
                {"property": "measured_cell_temp_c", "value": 81.0, "unit": "degC",
                 "conditions": {"discharge_current_a": 20.0}, "chart_only": False,
                 "source_fragment": "At 20A the cell reaches 81C"},
                {"property": "measured_capacity_ah", "value": None, "unit": None,
                 "conditions": {}, "chart_only": True,
                 "source_fragment": "chart image"},
            ],
        }],
    }
    jf = tmp_path / "lygte_test.json"
    jf.write_text(json.dumps(payload))

    def counts():
        with neo4j_driver.session() as s:
            m = s.run("MATCH (m:Measurement)-[:MEASURED_BY]->"
                      "(:Source {source_id:'lygte_test_synthetic'}) "
                      "RETURN count(m) AS c").single()["c"]
            src = s.run("MATCH (s:Source {source_id:'lygte_test_synthetic'}) "
                        "RETURN count(s) AS c").single()["c"]
            return m, src

    try:
        with neo4j_driver.session() as s:   # the loader MATCHes the Cell
            s.run("MERGE (:Cell {model: 'TEST_LYGTE_CELL'})")
        load_independent(driver=neo4j_driver, json_path=jf)
        first = counts()
        load_independent(driver=neo4j_driver, json_path=jf)
        second = counts()
        assert first == second == (2, 1)
    finally:
        with neo4j_driver.session() as s:
            s.run("MATCH (m:Measurement)-[:MEASURED_BY]->"
                  "(:Source {source_id:'lygte_test_synthetic'}) DETACH DELETE m")
            s.run("MATCH (s:Source {source_id:'lygte_test_synthetic'}) DETACH DELETE s")
            s.run("MATCH (c:Cell {model:'TEST_LYGTE_CELL'}) DETACH DELETE c")
