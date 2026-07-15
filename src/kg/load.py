"""Idempotent, MERGE-only loader: processed Parquet -> Neo4j knowledge graph.

Running this twice produces identical node/edge counts (every write is a MERGE
keyed on a uniqueness-constrained property). Populates, for the Severson study:

  (Cell)-[:HAS_CHEMISTRY]->(Chemistry)
  (CellInstance)-[:INSTANCE_OF]->(Cell)
  (Measurement)-[:ABOUT]->(CellInstance)
  (Measurement)-[:MEASURED_BY]->(Source)
  (CellInstance)-[:SIMILAR_TO {weight, same_policy_group}]->(CellInstance)

Run:  python -m src.kg.load
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROCESSED, processed_parquet
from src.ingestion import cell_metadata as cm
from src.ingestion.features import FEATURE_COLUMNS
from src.ingestion.qc import cycle_life_table
from src.kg.connection import get_driver
from src.kg.features import parse_policy, similarity, zscore

K_NEIGHBORS = 10          # SIMILAR_TO neighbours kept per instance
# condition view: where the cell sits in charge-protocol space
FEATURE_COLS = ["c_rate_1", "c_rate_2", "soc_transition_pct"]
# behavior view: how the cell degrades in its first 100 cycles
BEHAVIOR_FEATURE_COLS = ["var_dQ_100_10", "min_dQ_100_10",
                         "cap_ratio_100_2", "slope_2_100"]
# (view name, feature columns) pairs loaded by default
EDGE_VIEWS = (("condition", FEATURE_COLS), ("behavior", BEHAVIOR_FEATURE_COLS))


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    type: str
    citation: str = ""


SEVERSON_SOURCE = SourceSpec(
    source_id="severson_mit_2019",
    type="academic_cycling_study",
    citation="Severson et al. 2019, Nature Energy",
)

# --- Cypher ---------------------------------------------------------------
_MEASUREMENT_CONSTRAINT = (
    "CREATE CONSTRAINT measurement_id_unique IF NOT EXISTS "
    "FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE"
)

_CELL_CYPHER = """
MERGE (c:Cell {model: $model})
  SET c.manufacturer = $manufacturer,
      c.form_factor = $form_factor,
      c.nominal_capacity_ah = $nominal
MERGE (ch:Chemistry {name: $chemistry})
MERGE (c)-[:HAS_CHEMISTRY]->(ch)
MERGE (s:Source {source_id: $source_id})
  SET s.type = $source_type, s.citation = $citation
"""

_INSTANCE_CYPHER = """
UNWIND $rows AS row
MATCH (c:Cell {model: $model})
MATCH (s:Source {source_id: $source_id})
MERGE (ci:CellInstance {study_cell_id: row.study_cell_id})
  SET ci.batch = row.batch,
      ci.charge_policy_raw = row.charge_policy_raw,
      ci.charge_policy_norm = row.charge_policy_norm,
      ci.policy_group_id = row.policy_group_id,
      ci.qc_flag = row.qc_flag,
      ci.c_rate_1 = row.c_rate_1,
      ci.c_rate_2 = row.c_rate_2,
      ci.soc_transition_pct = row.soc_transition_pct
  SET ci += row.features
  SET ci += $instance_extra
MERGE (ci)-[:INSTANCE_OF]->(c)
WITH ci, s, row
UNWIND row.measurements AS meas
MERGE (m:Measurement {measurement_id: meas.measurement_id})
  SET m.metric = meas.metric,
      m.value = meas.value,
      m.censored = meas.censored,
      m.unit = meas.unit,
      m.parquet_ref = $parquet_ref
MERGE (m)-[:ABOUT]->(ci)
MERGE (m)-[:MEASURED_BY]->(s)
"""

# SIMILAR_TO is a *derived* edge set (recomputed from features each load), in
# two views distinguished by the `view` edge property:
#   'condition' — similarity in charge-protocol space (c_rate_1/2, soc_transition)
#   'behavior'  — similarity in early-cycle degradation space (var_dQ, min_dQ, ...)
# Unlike the structural graph, a plain MERGE would leave stale edges behind when
# the feature space changes (a shifted top-k). So we clear the loaded instances'
# outgoing edges *of that view* first, then MERGE the fresh set — idempotent AND
# correct under data changes, without clobbering the other view. Legacy edges
# without a view property are treated as 'condition' and migrated on reload.
_EDGE_CLEAR_CYPHER = """
UNWIND $ids AS id
MATCH (a:CellInstance {study_cell_id: id})-[r:SIMILAR_TO]->()
WHERE r.view = $view OR (r.view IS NULL AND $view = 'condition')
DELETE r
"""

_EDGE_CYPHER = """
UNWIND $edges AS e
MATCH (a:CellInstance {study_cell_id: e.src})
MATCH (b:CellInstance {study_cell_id: e.dst})
MERGE (a)-[r:SIMILAR_TO {view: e.view}]->(b)
  SET r.weight = e.weight, r.same_policy_group = e.same_policy_group
"""


# --- Value sanitizing (Neo4j rejects NaN / numpy scalar types) ------------
def _f(x):
    return None if pd.isna(x) else float(x)


def _i(x):
    return None if pd.isna(x) else int(x)


def _s(x):
    if pd.isna(x):
        return None
    return str(x)


def _measurement(cell_id: str, metric: str, value, reached: bool, unit: str) -> dict:
    v = _f(value)
    censored = (not reached) or v is None
    return {
        "measurement_id": f"{cell_id}:{metric}",
        "metric": metric,
        "value": None if censored else v,   # right-censored -> no value, censored flag only
        "censored": bool(censored),
        "unit": unit,
    }


def _instance_rows(inst_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in inst_df.iterrows():
        cid = str(r["study_cell_id"])
        rows.append({
            "study_cell_id": cid,
            "batch": _i(r.get("batch")),
            "charge_policy_raw": _s(r.get("charge_policy_raw")),
            "charge_policy_norm": _s(r.get("charge_policy_norm")),
            "policy_group_id": _s(r.get("policy_group_id")),
            "qc_flag": _s(r.get("qc_flag")) or "",
            "c_rate_1": _f(r.get("c_rate_1")),
            "c_rate_2": _f(r.get("c_rate_2")),
            "soc_transition_pct": _f(r.get("soc_transition_pct")),
            # early-cycle features (empty dict -> no-op if not present)
            "features": {c: _f(r.get(c)) for c in FEATURE_COLUMNS
                         if c in inst_df.columns and not pd.isna(r.get(c))},
            "measurements": [
                _measurement(cid, "cycle_life_nominal",
                             r.get("cycle_life_nominal"), bool(r.get("reached_eol_nominal")), "cycles"),
                _measurement(cid, "cycle_life_initial",
                             r.get("cycle_life_initial"), bool(r.get("reached_eol_initial")), "cycles"),
                _measurement(cid, "initial_capacity_ah",
                             r.get("initial_capacity_ah"), True, "Ah"),
            ],
        })
    return rows


def similarity_edges(inst_df: pd.DataFrame, k: int = K_NEIGHBORS,
                     feature_cols: list[str] = FEATURE_COLS,
                     view: str = "condition") -> list[dict]:
    """Top-k SIMILAR_TO edges per instance on z-scored `feature_cols`.

    Instances with NaN features (e.g. unparseable policy) are excluded from
    that view's similarity graph entirely (no edges in or out). Same-policy-
    group edges are kept but flagged so downstream code can drop them.
    """
    missing = [c for c in feature_cols if c not in inst_df.columns]
    if missing:
        raise ValueError(f"view '{view}': missing feature columns {missing}")
    d = inst_df.dropna(subset=feature_cols).reset_index(drop=True)
    ids = d["study_cell_id"].astype(str).tolist()
    group = dict(zip(ids, d["policy_group_id"].map(_s)))
    Z = zscore(d[feature_cols].to_numpy(dtype=float))

    edges: list[dict] = []
    for i in range(len(ids)):
        sims = [(ids[j], similarity(Z[i], Z[j])) for j in range(len(ids)) if j != i]
        # deterministic ordering for idempotency: weight desc, then dst id asc
        sims.sort(key=lambda t: (-t[1], t[0]))
        for dst, w in sims[:k]:
            edges.append({
                "src": ids[i],
                "dst": dst,
                "view": view,
                "weight": round(float(w), 6),
                "same_policy_group": bool(group[ids[i]] is not None and group[ids[i]] == group[dst]),
            })
    return edges


def load_dataframe(inst_df, *, cell: cm.CommercialCell, source: SourceSpec,
                   parquet_ref: str, driver, k_neighbors: int = K_NEIGHBORS,
                   instance_extra: dict | None = None,
                   edge_views=(("condition", FEATURE_COLS),)) -> dict:
    """Load one study's instance frame into Neo4j (idempotent).

    `instance_extra` sets extra properties on every CellInstance (e.g. a
    `test_run` marker so integration tests can namespace and tear down cleanly).
    `edge_views` lists the (view, feature_cols) similarity graphs to build;
    each view is clean-rebuilt independently.
    """
    rows = _instance_rows(inst_df)
    ids = [r["study_cell_id"] for r in rows]
    edge_counts: dict[str, int] = {}
    with driver.session() as s:
        s.run(_MEASUREMENT_CONSTRAINT)
        s.run(_CELL_CYPHER, model=cell.model, manufacturer=cell.manufacturer,
              form_factor=cell.form_factor, nominal=float(cell.nominal_capacity_ah),
              chemistry=cell.chemistry, source_id=source.source_id,
              source_type=source.type, citation=source.citation)
        s.run(_INSTANCE_CYPHER, rows=rows, model=cell.model,
              source_id=source.source_id, parquet_ref=parquet_ref,
              instance_extra=instance_extra or {})
        # clean-rebuild each view's derived SIMILAR_TO edges independently
        for view, cols in edge_views:
            edges = similarity_edges(inst_df, k_neighbors, feature_cols=cols, view=view)
            s.run(_EDGE_CLEAR_CYPHER, ids=ids, view=view)
            if edges:
                s.run(_EDGE_CYPHER, edges=edges)
            edge_counts[view] = len(edges)
    return {"instances": len(rows),
            "similar_to_edges": sum(edge_counts.values()),
            "edges_by_view": edge_counts}


def build_severson_instances() -> pd.DataFrame:
    """Per-instance frame for Severson: cycle-life summary + policy features."""
    df = pd.read_parquet(processed_parquet("severson_mit"))
    life = cycle_life_table(df)
    meta = (df.groupby("cell_id")
              .agg(batch=("batch", "first"),
                   charge_policy_raw=("charge_policy_raw", "first"),
                   charge_policy_norm=("charge_policy_norm", "first"),
                   policy_group_id=("policy_group_id", "first"))
              .reset_index())
    inst = life.merge(meta, on="cell_id", how="left")
    # early-cycle features (if built) -> extra CellInstance properties
    fpath = PROCESSED / "severson_features.parquet"
    if fpath.exists():
        inst = inst.merge(pd.read_parquet(fpath), on="cell_id", how="left")
    inst = inst.rename(columns={"cell_id": "study_cell_id", "flags": "qc_flag"})
    feats = inst["charge_policy_norm"].map(parse_policy)
    inst["c_rate_1"] = [f.c_rate_1 if f else np.nan for f in feats]
    inst["c_rate_2"] = [f.c_rate_2 if f else np.nan for f in feats]
    inst["soc_transition_pct"] = [f.soc_transition_pct if f else np.nan for f in feats]
    return inst


def load(driver=None) -> dict:
    own = driver is None
    driver = driver or get_driver()
    try:
        inst = build_severson_instances()
        # behavior view needs the early-cycle features parquet; skip if absent
        views = [(v, cols) for v, cols in EDGE_VIEWS
                 if all(c in inst.columns for c in cols)]
        skipped_views = [v for v, cols in EDGE_VIEWS
                         if not all(c in inst.columns for c in cols)]
        result = load_dataframe(
            inst, cell=cm.SEVERSON_CELL, source=SEVERSON_SOURCE,
            parquet_ref=str(processed_parquet("severson_mit")), driver=driver,
            edge_views=views)
        unparseable = inst.loc[inst["c_rate_1"].isna(),
                               ["study_cell_id", "charge_policy_raw"]]
        print(f"[kg.load] {result['instances']} instances, "
              f"SIMILAR_TO by view: {result['edges_by_view']}")
        if skipped_views:
            print(f"[kg.load] views skipped (features missing): {skipped_views} "
                  f"— run: python -m src.ingestion.features")
        if len(unparseable):
            print(f"[kg.load] {len(unparseable)} unparseable policy string(s) "
                  f"(no similarity edges): "
                  f"{list(unparseable.itertuples(index=False, name=None))}")
        return result
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    load()
