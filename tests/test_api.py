"""The web API (api/main.py) against the shipped snapshot + artifacts: same
numbers as the Streamlit pages, and bad input refused instead of guessed."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import PRESETS, app  # noqa: E402

client = TestClient(app)


def _skip_without_sources():
    h = client.get("/api/health").json()
    if h["status"] != "ok":
        pytest.skip(f"KG or artifacts unavailable: {h}")


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_presets_give_their_verified_outcome():
    _skip_without_sources()
    ok = client.post("/api/predict", json=PRESETS["pass"]).json()
    refuse = client.post("/api/predict", json=PRESETS["refuse"]).json()
    assert ok["abstain"] is False and ok["prediction_cycles"] > 0
    assert ok["band_lo"] <= ok["band_hi"]
    assert refuse["abstain"] is True


def test_predict_rejects_out_of_range_and_unknown_profile():
    _skip_without_sources()
    bad = {**PRESETS["pass"], "c_rate_1": 99}
    assert client.post("/api/predict", json=bad).status_code == 400
    assert client.post("/api/predict", json={**PRESETS["pass"], "profile": "nope"}).status_code == 400


def test_promise_uses_the_stored_comparison_set():
    _skip_without_sources()
    models = [c["model"] for c in client.get("/api/cells").json()["cells"]]
    for m in models:
        r = client.get("/api/promise", params={"model": m}).json()
        if r["discrepancy"]:
            assert len(r["values"]) == r["discrepancy"]["n"]
    assert client.get("/api/promise", params={"model": "nope"}).status_code == 404


def test_specs_headline_matches_paper_table_8():
    _skip_without_sources()
    s = client.get("/api/specs").json()
    assert (len(s["problems"]), s["n_comparisons"]) == (14, 43)
    # manufacturer-internal disagreements are listed first
    prov = [p["provenance"] == "manufacturer_internal" for p in s["problems"]]
    assert prov == sorted(prov, reverse=True)


def test_graph_neighbourhood_and_validation():
    _skip_without_sources()
    center = client.get("/api/instances").json()["instances"][0]
    g = client.get("/api/graph", params={"center": center, "view": "condition", "k": 7}).json()
    assert 0 < len(g["neighbors"]) <= 7
    assert client.get("/api/graph", params={"center": center, "k": 11}).status_code == 422
    assert client.get("/api/graph", params={"center": center, "view": "x"}).status_code == 400
    assert client.get("/api/graph", params={"center": "nope"}).status_code == 404
