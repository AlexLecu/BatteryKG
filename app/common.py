"""Shared app plumbing: config resolution, KG access, artifacts, serving math.

Honesty rule: every number shown by the app comes from (a) the knowledge
graph, (b) the model artifacts in app/artifacts/, or (c) a report file in
outputs/. Helpers here return None + a reason when a source is unavailable,
and pages render that state explicitly instead of faking data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                     # make src importable
ARTIFACT_DIR = ROOT / "app" / "artifacts"
OUTPUTS = ROOT / "outputs"

# paper palette (CVD-validated)
BLUE, ORANGE, GREEN, GOLD, PINK = "#2563eb", "#ea580c", "#059669", "#b45309", "#db2777"
INK, INK_MUT = "#1f2937", "#6b7280"


# --- config: env vars first, .env fallback, no hardcoded hosts ---------------
def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def resolve_config(name: str, default: str | None = None,
                   env_file: Path | None = None) -> str | None:
    """os.environ wins; falls back to .env (repo root unless overridden)."""
    if name in os.environ:
        return os.environ[name]
    return _parse_env_file(env_file or ROOT / ".env").get(name, default)


def neo4j_settings() -> dict:
    return {
        "uri": resolve_config("NEO4J_URI", "bolt://localhost:7687"),
        "user": resolve_config("NEO4J_USER", "neo4j"),
        "password": resolve_config("NEO4J_PASSWORD"),
    }


# --- KG access (fail-soft, with static-snapshot fallback) ------------------------
SNAPSHOT_PATH = ROOT / "data" / "kg_snapshots" / "app_snapshot.json"


def snapshot_key(query: str, params: dict) -> str:
    """Cache key for one KG query: whitespace-normalized Cypher + sorted params.

    Whitespace normalization means the *same* query string reused at a
    different indentation level still hits the same snapshot entry — but any
    textually new query misses (by design: the snapshot only ever contains
    results for Cypher the live pages actually issued at generation time)."""
    return " ".join(query.split()) + " | " + json.dumps(params, sort_keys=True,
                                                        default=str)


class SnapshotDriver:
    """Read-only stand-in for a Neo4j driver, backed by app_snapshot.json.

    Serves the recorded result rows for exactly the (query, params) pairs
    captured by scripts/gen_app_snapshot.py against a live KG — nothing else.
    An un-snapshotted query raises instead of returning fabricated data
    (honesty rule)."""

    def __init__(self, path: Path = SNAPSHOT_PATH):
        payload = json.loads(path.read_text())
        self.generated_at: str = payload["generated_at"]
        self.entries: dict[str, list[dict]] = payload["entries"]

    def query(self, query: str, params: dict) -> list[dict]:
        key = snapshot_key(query, params)
        if key not in self.entries:
            raise KeyError(
                "query not in the static KG snapshot (snapshot mode serves only "
                "results captured from a live KG; re-run "
                "scripts/gen_app_snapshot.py after changing queries): " + key)
        return [dict(r) for r in self.entries[key]]

    def close(self) -> None:
        pass


def is_snapshot(driver) -> bool:
    """True when KG 'queries' are served from the static snapshot file."""
    return isinstance(driver, SnapshotDriver)


def make_driver():
    """Return (driver, error_message). driver is None on failure.

    When Neo4j is unreachable and data/kg_snapshots/app_snapshot.json exists,
    returns a SnapshotDriver instead — the pages render identically from the
    recorded query results (Hugging Face Space mode)."""
    s = neo4j_settings()
    err = None
    if not s["password"]:
        err = "NEO4J_PASSWORD is not set (env var or .env)"
    else:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                s["uri"], auth=(s["user"], s["password"]),
                notifications_min_severity="OFF")
            driver.verify_connectivity()
            return driver, None
        except Exception as e:                    # noqa: BLE001 — surface any failure
            err = f"Neo4j unreachable at {s['uri']}: {type(e).__name__}: {e}"
    if SNAPSHOT_PATH.exists():
        try:
            return SnapshotDriver(), None
        except Exception as e:                    # noqa: BLE001
            err += f" (snapshot fallback also failed: {type(e).__name__}: {e})"
    return None, err


def run_query(driver, query: str, **params) -> list[dict]:
    if is_snapshot(driver):
        return driver.query(query, params)
    with driver.session() as s:
        return [r.data() for r in s.run(query, **params)]


# --- artifacts -------------------------------------------------------------------
def load_artifacts():
    """Return (artifacts dict, error_message)."""
    meta_path = ARTIFACT_DIR / "meta.json"
    if not meta_path.exists():
        return None, ("model artifacts missing — run "
                      "`python -m src.models.train_final` first")
    try:
        from xgboost import XGBRegressor
        meta = json.loads(meta_path.read_text())
        models = {}
        for key, fname in [("point", "model.json"), ("q16", "model_q16.json"),
                           ("q84", "model_q84.json")]:
            m = XGBRegressor()
            m.load_model(ARTIFACT_DIR / fname)
            models[key] = m
        bank = pd.read_parquet(ARTIFACT_DIR / "neighbor_bank.parquet")
        return {"meta": meta, "models": models, "bank": bank}, None
    except Exception as e:                        # noqa: BLE001
        return None, f"artifact loading failed: {type(e).__name__}: {e}"


# --- serving math (mirrors src/models/train_final.py exactly) ----------------------
def _zscore_with(values: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    std_arr = np.where(np.asarray(std) == 0, 1.0, np.asarray(std))
    return (values - np.asarray(mean)) / std_arr


def view_neighbors(art: dict, view: str, query_features: dict,
                   exclude_group: str | None = None, k: int | None = None) -> pd.DataFrame:
    """Top-k bank neighbours of a query point in one similarity view.

    Returns a dataframe with cell_id, weight, cycle_life_nominal, policy."""
    meta, bank = art["meta"], art["bank"]
    k = k or meta["k_neighbors"]
    sc = meta["scaler"][view]
    cols = sc["columns"]
    q = _zscore_with(np.array([[query_features[c] for c in cols]], float),
                     sc["mean"], sc["std"])[0]
    B = _zscore_with(bank[cols].to_numpy(float), sc["mean"], sc["std"])
    d = np.linalg.norm(B - q, axis=1)
    w = 1.0 / (1.0 + d)
    out = bank[["cell_id", "policy_group_id", "charge_policy_norm",
                "cycle_life_nominal", "log10_cycle_life"]].copy()
    out["weight"] = w
    if exclude_group is not None:
        out = out[out["policy_group_id"] != exclude_group]
    out = out.sort_values(["weight", "cell_id"], ascending=[False, True])
    # a bank cell identical to the query (self) has weight 1.0 — drop it
    out = out[out["weight"] < 0.999999]
    return out.head(k).reset_index(drop=True)


def graph_features_for_query(art: dict, query_features: dict,
                             exclude_group: str | None = None) -> dict:
    feats = {}
    for view in ("condition", "behavior"):
        nb = view_neighbors(art, view, query_features, exclude_group)
        w = nb["weight"].to_numpy()
        y = nb["log10_cycle_life"].to_numpy()
        wmean = float(np.sum(w * y) / np.sum(w))
        wstd = float(np.sqrt(np.sum(w * (y - wmean) ** 2) / np.sum(w)))
        feats[f"{view}_nbr_wmean_log_life"] = wmean
        feats[f"{view}_nbr_wstd_log_life"] = wstd
        feats[f"{view}_nbr_mean_weight"] = float(w.mean())
        feats[f"{view}_coverage_train"] = float(w.sum())
    return feats


def predict_with_gate(art: dict, query_features: dict,
                      exclude_group: str | None = None) -> dict:
    """Full serving path: graph features -> gate -> prediction + band."""
    meta = art["meta"]
    gf = graph_features_for_query(art, query_features, exclude_group)
    coverage = gf["behavior_coverage_train"]
    threshold = meta["abstention"]["threshold"]
    row = {**{c: query_features[c] for c in meta["base_features"]}, **gf}
    X = pd.DataFrame([row])[meta["features"]]
    pred = {k: float(10 ** m.predict(X)[0]) for k, m in art["models"].items()}
    lo, hi = sorted((pred["q16"], pred["q84"]))
    return {
        "abstain": coverage < threshold,
        "coverage": coverage,
        "threshold": threshold,
        "prediction_cycles": pred["point"],
        "band_lo": lo,
        "band_hi": hi,
        "graph_features": gf,
    }
