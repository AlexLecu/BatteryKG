"""Tests for the Literature Monitor: parsing, dedup, triage parsing, promotion gate."""
import json
from pathlib import Path

import pytest

from src.agents.literature_monitor import (
    dedup_key,
    load_existing_keys,
    parse_arxiv_response,
    parse_s2_response,
    parse_triage,
    stage_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures"


# --- API response parsing -------------------------------------------------------
def test_parse_s2_fixture_filters_year_and_normalizes():
    payload = json.loads((FIXTURES / "s2_fixture.json").read_text())
    out = parse_s2_response(payload, target="Panasonic NCR18650B", query="NCR18650B")
    assert len(out) == 2                                   # 2019 paper filtered
    doi_rec = next(r for r in out if r["doi"])
    assert doi_rec["candidate_id"] == "doi:10.1000/test.2025.001"
    assert doi_rec["year"] == 2025
    assert doi_rec["matched_target"] == "Panasonic NCR18650B"
    arxiv_rec = next(r for r in out if r["arxiv_id"])
    assert arxiv_rec["candidate_id"] == "arxiv:2401.99999"
    assert arxiv_rec["abstract"] == ""                     # null -> empty string


def test_parse_arxiv_fixture():
    xml = (FIXTURES / "arxiv_fixture.xml").read_text()
    out = parse_arxiv_response(xml, target="datasets", query="18650 degradation data")
    assert len(out) == 1                                   # 2019 entry filtered
    rec = out[0]
    assert rec["arxiv_id"] == "2502.12345v1"
    assert rec["venue"] == "arXiv"
    assert "public degradation dataset" in rec["title"]
    assert rec["authors"] == ["D. Researcher", "E. Researcher"]


def test_dedup_key_priority():
    assert dedup_key("10.1/X", "2401.1", "T").startswith("doi:")
    assert dedup_key(None, "2401.1", "T").startswith("arxiv:")
    k1 = dedup_key(None, None, "Same   Title!")
    k2 = dedup_key(None, None, "same title")
    assert k1 == k2 and k1.startswith("title:")


# --- staging dedup -------------------------------------------------------------------
def test_restaging_is_idempotent(tmp_path):
    jsonl = tmp_path / "candidates.jsonl"
    cands = [{"candidate_id": "doi:10.1/a", "title": "A", "authors": [],
              "venue": "", "year": 2025, "doi": "10.1/a", "arxiv_id": None,
              "abstract": "", "matched_target": "t", "matched_query": "q",
              "api": "semantic_scholar"}]
    first = stage_candidates(cands, jsonl)
    assert len(first) == 1
    second = stage_candidates(cands, jsonl)                # same again
    assert second == []                                    # deduped
    assert len(load_existing_keys(jsonl)) == 1
    # duplicate within one batch also dedups
    third = stage_candidates([{**cands[0], "candidate_id": "doi:10.1/b"},
                              {**cands[0], "candidate_id": "doi:10.1/b"}], jsonl)
    assert len(third) == 1


# --- triage parsing ---------------------------------------------------------------------
@pytest.mark.parametrize("raw, verdict", [
    ("yes: the abstract explicitly mentions a released cycling dataset", "yes"),
    ("Maybe - unclear whether data is public", "maybe"),
    ("no: about electric motor control, not cells", "no"),
    ("The paper seems interesting but I cannot say.", "unparsed"),
])
def test_parse_triage(raw, verdict):
    v, rationale = parse_triage(raw)
    assert v == verdict
    assert rationale


# --- promotion requires the explicit flag ----------------------------------------------
def test_promotion_without_confirm_fails():
    from src.agents.promote import promote
    with pytest.raises(SystemExit, match="Refusing to promote"):
        promote("doi:10.1/whatever", confirm=False)
