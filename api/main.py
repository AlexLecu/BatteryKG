"""JSON API behind the BatteryKG web app (web/).

Serves the same numbers as the Streamlit pages in app/pages/, through the same
code: KG answers come from app.common.make_driver() (live Neo4j when NEO4J_* is
set, otherwise the recorded snapshot in data/kg_snapshots/), predictions from
app.common.predict_with_gate() over the artifacts in app/artifacts/. The Cypher
is the exact text the pages issue (imported from scripts/gen_app_snapshot.py),
so every query hits the snapshot.

Run:  uvicorn api.main:app --port 8000      (from the repo root)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.common import is_snapshot, load_artifacts, make_driver, predict_with_gate, run_query
from scripts.gen_app_snapshot import (K_RANGE, Q_CELLS, Q_CENTER_LIFE, Q_CLAIMS, Q_DISC, Q_INSTANCES,
                                      Q_MEAS, Q_NEIGHBORS, Q_SPEC, VIEWS)

CONDITION = ["c_rate_1", "c_rate_2", "soc_transition_pct"]
# presets verified by scripts/verify_presets.py (same values as the Streamlit page)
PRESETS = {
    "pass": {"profile": "b1c29", "c_rate_1": 6.0, "c_rate_2": 3.0, "soc_transition_pct": 50.0},
    "refuse": {"profile": "b2c1", "c_rate_1": 2.0, "c_rate_2": 6.0, "soc_transition_pct": 10.0},
}
NOMINAL_AH = 1.1        # A123 APR18650M1A datasheet nominal (the Severson cell)
COMPARISON_VERDICTS = {"consistent", "condition_mismatch", "document_variant_conflict"}
SEVERSON_ID = re.compile(r"^b[123]c\d+$")

app = FastAPI(title="BatteryKG", docs_url=None, redoc_url=None)
if os.getenv("CORS_ORIGINS"):
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS").split(","),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# --- sources (loaded once; a failure is reported per request, never faked) ------------
@lru_cache(maxsize=1)
def _driver():
    return make_driver()


@lru_cache(maxsize=1)
def _artifacts():
    return load_artifacts()


def kg():
    driver, err = _driver()
    if driver is None:
        raise HTTPException(503, f"Knowledge graph unavailable — {err}")
    return driver


def artifacts():
    art, err = _artifacts()
    if art is None:
        raise HTTPException(503, f"Model artifacts unavailable — {err}")
    return art


def q(query: str, **params) -> list[dict]:
    try:
        return run_query(kg(), query, **params)
    except KeyError as e:                       # not in the snapshot: say so
        raise HTTPException(404, str(e.args[0]))


def _cell_models() -> list[str]:
    return [c["model"] for c in q(Q_CELLS)]


def _instance_ids() -> list[str]:
    return [i["id"] for i in q(Q_INSTANCES)]


# --- endpoints ------------------------------------------------------------------------
@app.get("/api/health")
def health():
    driver, err = _driver()
    art, art_err = _artifacts()
    return {
        "status": "ok" if driver is not None and art is not None else "degraded",
        "kg": ("snapshot" if is_snapshot(driver) else "live") if driver is not None else None,
        "snapshot_generated_at": getattr(driver, "generated_at", None),
        "kg_error": err,
        "artifacts_error": art_err,
    }


@app.get("/api/cells")
def cells():
    return {"cells": q(Q_CELLS)}


@app.get("/api/predictor")
def predictor():
    """Everything the 'Will it last?' page needs before the first prediction."""
    art = artifacts()
    meta, bank = art["meta"], art["bank"]
    return {
        "limits": {c: [float(bank[c].min()), float(bank[c].max())] for c in CONDITION},
        "profiles": [{"id": r["cell_id"], "policy": r["charge_policy_norm"]} for _, r in bank.iterrows()],
        "presets": PRESETS,
        "nominal_ah": NOMINAL_AH,
        "n_training_cells": meta["n_training_cells"],
        "k_neighbors": meta["k_neighbors"],
        "threshold": meta["abstention"]["threshold"],
        "cv_rmse_cycles": meta["cv_metrics"]["rmse_cycles"],
        "cv_mape_pct": meta["cv_metrics"]["mape_pct"],
    }


class PredictRequest(BaseModel):
    profile: str
    c_rate_1: float
    c_rate_2: float
    soc_transition_pct: float


@app.post("/api/predict")
def predict(body: PredictRequest):
    art = artifacts()
    meta, bank = art["meta"], art["bank"]
    rows = bank[bank["cell_id"] == body.profile]
    if rows.empty:
        raise HTTPException(400, f"Unknown cell profile {body.profile!r}.")
    query = {c: float(rows.iloc[0][c]) for c in meta["base_features"]}
    for c in CONDITION:
        lo, hi = float(bank[c].min()), float(bank[c].max())
        v = getattr(body, c)
        if not lo <= v <= hi:
            raise HTTPException(400, f"{c} must be between {lo:g} and {hi:g}.")
        query[c] = v
    res = predict_with_gate(art, query)
    return {k: (bool(v) if k == "abstain" else v) for k, v in res.items() if k != "graph_features"}


@app.get("/api/promise")
def promise(model: str):
    """'Promise vs reality': the measured distribution, the stored discrepancy, the claims."""
    if model not in _cell_models():
        raise HTTPException(404, f"Unknown cell {model!r}.")
    values = [m["v"] for m in q(Q_MEAS, model=model)]
    disc = q(Q_DISC, model=model)
    d = disc[0] if disc else None
    claims = q(Q_CLAIMS, model=model)
    mismatch = None
    if values and d and len(values) != d["n"]:
        # the raw measurement query spans every study of this cell model (HUST
        # shares the A123), but the stored Discrepancy is Severson-only — rebuild
        # exactly its measurement set (Severson instance ids are b<batch>c<n>)
        values = []
        for sid in (i for i in _instance_ids() if SEVERSON_ID.match(i)):
            r = q(Q_CENTER_LIFE, center=sid)
            if r and r[0]["life"] is not None:
                values.append(r[0]["life"])
        if len(values) != d["n"]:
            mismatch = (f"Measurement set ({len(values)}) does not match the stored comparison "
                        f"(n={d['n']}) — showing nothing rather than a mismatched plot.")
            values = []
    return {"model": model, "values": values, "discrepancy": d, "claims": claims, "mismatch": mismatch}


@app.get("/api/specs")
def specs():
    """'Who is lying?': every stored claim-vs-claim row, problems first (paper Table 8)."""
    rows = []
    for model in _cell_models():
        for r in q(Q_SPEC, model=model):
            rows.append({**r, "cell": model})
    comparisons = [r for r in rows if r["verdict"] in COMPARISON_VERDICTS]
    problems = sorted((r for r in comparisons if r["verdict"] != "consistent"),
                      key=lambda r: (r["provenance"] != "manufacturer_internal", r["cell"], r["property"]))
    return {
        "rows": rows,
        "problems": problems,
        "n_comparisons": len(comparisons),
        "n_consistent": sum(r["verdict"] == "consistent" for r in comparisons),
        "n_single_document": sum(r["verdict"] == "coverage_gap" for r in rows),
    }


@app.get("/api/instances")
def instances():
    return {"instances": _instance_ids(), "views": VIEWS, "k": [min(K_RANGE), max(K_RANGE)]}


@app.get("/api/graph")
def graph(center: str, view: str = "behavior", k: int = Query(5, ge=min(K_RANGE), le=max(K_RANGE))):
    """'The graph': the SIMILAR_TO neighbourhood of one cell instance."""
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {VIEWS}.")
    if center not in _instance_ids():
        raise HTTPException(404, f"Unknown cell instance {center!r}.")
    life = q(Q_CENTER_LIFE, center=center)
    return {
        "center": center,
        "center_life": life[0]["life"] if life else None,
        "neighbors": q(Q_NEIGHBORS, center=center, view=view, k=k),
    }
