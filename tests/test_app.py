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
