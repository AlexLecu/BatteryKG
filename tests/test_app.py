"""Headless smoke tests for the Streamlit app (AppTest), artifact loading,
and env-var config resolution."""
from pathlib import Path

import pytest

from app.common import load_artifacts, predict_with_gate, resolve_config

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP / "pages").glob("*.py"))


# --- config resolution -----------------------------------------------------------
def test_env_var_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MYKEY=from_file\n")
    monkeypatch.setenv("MYKEY", "from_env")
    assert resolve_config("MYKEY", env_file=env_file) == "from_env"


def test_env_file_fallback_and_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MYKEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nMYKEY=from_file\nOTHER=x\n")
    assert resolve_config("MYKEY", env_file=env_file) == "from_file"
    assert resolve_config("ABSENT", default="dflt", env_file=env_file) == "dflt"


def test_missing_env_file_gives_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MYKEY", raising=False)
    assert resolve_config("MYKEY", default=None,
                          env_file=tmp_path / "nope.env") is None


# --- artifacts ----------------------------------------------------------------------
def test_artifact_loading_and_serving_path():
    art, err = load_artifacts()
    if art is None:
        pytest.skip(f"artifacts not built: {err}")
    meta = art["meta"]
    assert meta["abstention"]["threshold"] > 0
    assert set(meta["features"]) >= set(meta["base_features"])
    assert len(art["bank"]) == meta["n_training_cells"]
    # end-to-end serving path on a bank row (self excluded by group)
    row = art["bank"].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}
    res = predict_with_gate(art, query, exclude_group=row["policy_group_id"])
    assert res["prediction_cycles"] > 0
    assert res["band_lo"] <= res["band_hi"]
    assert isinstance(res["abstain"], bool)
    assert res["coverage"] > 0


# --- Who-Is-Lying headline definition (paper Table 8) -------------------------------
def test_who_is_lying_headline_counts():
    """Denominator counts only properties present in BOTH documents; the
    coverage_gap rows are not comparisons. Against the shipped KG snapshot the
    headline must read exactly '14 of 43'."""
    from app.common import SNAPSHOT_PATH, SnapshotDriver

    if not SNAPSHOT_PATH.exists():
        pytest.skip("app_snapshot.json not present")
    sd = SnapshotDriver()
    cells = sd.query("MATCH (c:Cell) RETURN c.model AS model, "
                     "c.manufacturer AS mfr ORDER BY model", {})
    rows = []
    for c in cells:
        rows += sd.query("""
    MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(:Cell {model: $model})
    RETURN d.property AS property, d.value_a AS value_a, d.value_b AS value_b,
           d.source_a AS source_a, d.source_b AS source_b,
           d.verdict AS verdict, d.rationale AS detail,
           d.provenance AS provenance,
           d.fragment_a AS fragment_a, d.fragment_b AS fragment_b
    ORDER BY d.verdict, d.property
""", {"model": c["model"]})
    comparison_verdicts = {"consistent", "condition_mismatch",
                           "document_variant_conflict"}
    denominator = sum(1 for r in rows if r["verdict"] in comparison_verdicts)
    numerator = sum(1 for r in rows if r["verdict"] in
                    ("condition_mismatch", "document_variant_conflict"))
    gaps = sum(1 for r in rows if r["verdict"] == "coverage_gap")
    assert numerator == 14, f"headline numerator {numerator} != 14"
    assert denominator == 43, f"headline denominator {denominator} != 43"
    assert gaps == 37, f"coverage gaps {gaps} != 37"


# --- Promise-vs-Reality: plotted n must equal the stored Discrepancy's n ------------
def test_promise_vs_reality_n_matches_discrepancy():
    """The A123 measurement query spans studies (HUST shares the cell model);
    the page must plot exactly the Severson-only set the Discrepancy node was
    computed from (n_measurements)."""
    import re

    from app.common import SNAPSHOT_PATH, SnapshotDriver

    if not SNAPSHOT_PATH.exists():
        pytest.skip("app_snapshot.json not present")
    sd = SnapshotDriver()
    disc = sd.query("""
    MATCH (d:Discrepancy)-[:ABOUT]->(:Cell {model: $model})
    WHERE d.measured_median IS NOT NULL AND d.claim_value IS NOT NULL
    RETURN d.claim_value AS claim, d.measured_median AS med,
           d.measured_min AS mn, d.measured_max AS mx,
           d.measured_iqr_lo AS q1, d.measured_iqr_hi AS q3,
           d.n_measurements AS n, d.relative_gap AS gap,
           d.conditions_comparable AS comparable, d.rationale AS rationale,
           d.claim_conditions AS claim_conditions,
           d.measured_conditions AS measured_conditions
""", {"model": "A123 APR18650M1A"})
    assert disc, "no stored claim-vs-measured Discrepancy for the A123"
    d = disc[0]

    # the page's client-side reconstruction (same queries, same filter)
    instances = sd.query("""
    MATCH (ci:CellInstance) RETURN ci.study_cell_id AS id ORDER BY id""", {})
    sev_ids = [i["id"] for i in instances if re.match(r"^b[123]c\d+$", i["id"])]
    values = []
    for sid in sev_ids:
        r = sd.query("""
    MATCH (:CellInstance {study_cell_id: $center})<-[:ABOUT]-
          (m:Measurement {metric: 'cycle_life_nominal'})
    RETURN m.value AS life""", {"center": sid})
        if r and r[0]["life"] is not None:
            values.append(r[0]["life"])

    assert len(values) == d["n"] == 124, \
        f"plotted n {len(values)} != stored n_measurements {d['n']}"
    assert min(values) >= d["mn"] - 1 and max(values) <= d["mx"] + 1, \
        "reconstructed set exceeds the stored min/max"


# --- page smoke tests (headless) -------------------------------------------------------
@pytest.mark.parametrize("page", [APP / "main.py"] + PAGES,
                         ids=lambda p: p.stem)
def test_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page.name} raised: {at.exception}"
    # honesty rule: pages must render SOMETHING (content or an explicit
    # unavailable/error state), never an empty page
    rendered = (len(at.title) + len(at.markdown) + len(at.error) + len(at.warning)
                + len(at.dataframe) + len(at.metric))
    assert rendered > 0, f"{page.name} rendered nothing"
