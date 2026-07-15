"""Integration tests against a live Neo4j (skipped if unreachable).

Isolation strategy (Community edition = single database, so no throwaway DB):
  1. All test entities are namespaced — instance ids `tsynth_*`, cell model
     `TEST_MODEL_XYZ`, source `test_source_xyz`, chemistry `TESTCHEM` — so they
     never collide with real study data.
  2. Every test CellInstance is tagged with a `test_run` property for robust,
     property-based teardown.
  3. `compute_coverage` is called with `only_ids=<test ids>` so it never reads
     or writes the real instances' coverage.
  4. Teardown removes every test entity.
The `test_working_graph_untouched` check (run out-of-band, see README/report)
confirms real coverage values are byte-identical before and after the suite.
"""
import pandas as pd
import pytest

from src.ingestion.cell_metadata import CommercialCell
from src.kg.coverage import compute_coverage
from src.kg.load import SourceSpec, load_dataframe

pytestmark = pytest.mark.integration

TEST_CELL = CommercialCell(
    model="TEST_MODEL_XYZ", manufacturer="TestCo", chemistry="TESTCHEM",
    nominal_capacity_ah=1.0, form_factor="test", source="test-fixture",
)
TEST_SOURCE = SourceSpec("test_source_xyz", "unit_test", "fixture")
PREFIX = "tsynth_"
RUN_ID = "pytest_synthetic"
TEST_IDS = [PREFIX + x for x in "abcd"]


def _row(cid, pol, grp, c1, c2, soc, life_n, r_n, life_i, r_i, cap):
    return dict(
        study_cell_id=cid, batch=1, charge_policy_raw=pol, charge_policy_norm=pol,
        policy_group_id=grp, qc_flag="",
        cycle_life_nominal=life_n, reached_eol_nominal=r_n,
        cycle_life_initial=life_i, reached_eol_initial=r_i,
        initial_capacity_ah=cap, c_rate_1=c1, c_rate_2=c2, soc_transition_pct=soc,
    )


def _synth_df():
    return pd.DataFrame([
        _row(PREFIX + "a", "4C(80%)-4C", "tpg0", 4, 4, 80, 500.0, True, 520.0, True, 1.05),
        _row(PREFIX + "b", "4C(80%)-4C", "tpg0", 4, 4, 80, 510.0, True, 530.0, True, 1.06),
        _row(PREFIX + "c", "6C(40%)-3C", "tpg1", 6, 3, 40, 300.0, True, float("nan"), False, 1.04),
        _row(PREFIX + "d", "5C(50%)-5C", "tpg2", 5, 5, 50, 700.0, True, 720.0, True, 1.07),
    ])


def _counts(driver):
    with driver.session() as s:
        q = lambda c: s.run(c).single()["c"]
        return dict(
            cell=q("MATCH (n:Cell {model:'TEST_MODEL_XYZ'}) RETURN count(n) AS c"),
            chem=q("MATCH (n:Chemistry {name:'TESTCHEM'}) RETURN count(n) AS c"),
            source=q("MATCH (n:Source {source_id:'test_source_xyz'}) RETURN count(n) AS c"),
            instance=q(f"MATCH (n:CellInstance) WHERE n.study_cell_id STARTS WITH '{PREFIX}' RETURN count(n) AS c"),
            measurement=q(f"MATCH (m:Measurement) WHERE m.measurement_id STARTS WITH '{PREFIX}' RETURN count(m) AS c"),
            similar=q(f"MATCH (a:CellInstance)-[r:SIMILAR_TO]->(:CellInstance) WHERE a.study_cell_id STARTS WITH '{PREFIX}' RETURN count(r) AS c"),
            instance_of=q(f"MATCH (a:CellInstance)-[r:INSTANCE_OF]->(:Cell {{model:'TEST_MODEL_XYZ'}}) WHERE a.study_cell_id STARTS WITH '{PREFIX}' RETURN count(r) AS c"),
        )


def _cleanup(driver):
    with driver.session() as s:
        s.run(f"MATCH (m:Measurement) WHERE m.measurement_id STARTS WITH '{PREFIX}' DETACH DELETE m")
        # property-based teardown (test_run marker) + namespace fallback
        s.run("MATCH (n:CellInstance {test_run:$rid}) DETACH DELETE n", rid=RUN_ID)
        s.run(f"MATCH (n:CellInstance) WHERE n.study_cell_id STARTS WITH '{PREFIX}' DETACH DELETE n")
        s.run("MATCH (c:Cell {model:'TEST_MODEL_XYZ'}) DETACH DELETE c")
        s.run("MATCH (s:Source {source_id:'test_source_xyz'}) DETACH DELETE s")
        s.run("MATCH (ch:Chemistry {name:'TESTCHEM'}) DETACH DELETE ch")


def _load(driver):
    load_dataframe(_synth_df(), cell=TEST_CELL, source=TEST_SOURCE,
                   parquet_ref="test.parquet", driver=driver, k_neighbors=3,
                   instance_extra={"test_run": RUN_ID})


def test_loader_idempotent(neo4j_driver):
    _cleanup(neo4j_driver)
    try:
        _load(neo4j_driver)
        first = _counts(neo4j_driver)
        _load(neo4j_driver)                      # second run must not change counts
        second = _counts(neo4j_driver)
        assert first == second, (first, second)
        assert first["instance"] == 4
        assert first["cell"] == 1 and first["chem"] == 1 and first["source"] == 1
        assert first["measurement"] == 12        # 3 measurements per instance
        assert first["instance_of"] == 4
        assert first["similar"] == 12            # k=3 neighbours x 4 instances
    finally:
        _cleanup(neo4j_driver)


def test_coverage_written_and_xgroup_leq_all(neo4j_driver):
    _cleanup(neo4j_driver)
    try:
        _load(neo4j_driver)
        # scope coverage to the test ids only -> never touches real instances
        compute_coverage(driver=neo4j_driver, k=2, only_ids=TEST_IDS)
        with neo4j_driver.session() as s:
            rows = {r["id"]: (r["ca"], r["cx"]) for r in s.run(
                f"MATCH (ci:CellInstance) WHERE ci.study_cell_id STARTS WITH '{PREFIX}' "
                "RETURN ci.study_cell_id AS id, ci.coverage_all AS ca, "
                "ci.coverage_xgroup AS cx")}
        assert set(rows) == {PREFIX + x for x in "abcd"}
        for _cid, (ca, cx) in rows.items():
            assert ca is not None and cx is not None
            assert cx <= ca + 1e-9               # excluding neighbours can only lower it
        # a and b are the only same-group pair (identical features, weight 1.0),
        # so dropping same-group neighbours must strictly lower a's coverage.
        ca_a, cx_a = rows[PREFIX + "a"]
        assert cx_a < ca_a
    finally:
        _cleanup(neo4j_driver)
