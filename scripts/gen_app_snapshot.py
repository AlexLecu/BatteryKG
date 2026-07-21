"""Generate data/kg_snapshots/app_snapshot.json for snapshot-mode deployments.

Runs, against a LIVE Neo4j, exactly the Cypher strings the app pages issue
(copied verbatim from app/pages/*; snapshot_key() normalizes whitespace), over
the full parameter space the page widgets can produce, and records the results.
app.common.make_driver() falls back to serving these recorded results when
Neo4j is unreachable (e.g. the Hugging Face Space) — same rows, no mocking.

Run:  python -m scripts.gen_app_snapshot
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.common import SNAPSHOT_PATH, make_driver, run_query, snapshot_key

# --- Cypher strings, verbatim from the pages -------------------------------------

# app/pages/*Cell_Explorer.py
Q_CELLS = ("MATCH (c:Cell) RETURN c.model AS model, "
           "c.manufacturer AS mfr ORDER BY model")

Q_CLAIMS = """
    MATCH (cl:Claim)-[:ABOUT]->(:Cell {model: $model})
    MATCH (cl)-[:ASSERTED_BY]->(s:Source)
    RETURN cl.property AS property, cl.value AS value, cl.unit AS unit,
           cl.conditions_text AS stated_conditions, cl.page AS page,
           s.document AS document
    ORDER BY document, property
"""

Q_LYGTE = """
    MATCH (m:Measurement)-[:ABOUT]->(:Cell {model: $model})
    MATCH (m)-[:MEASURED_BY]->(s:Source {type: 'independent_test'})
    RETURN m.metric AS property, m.value AS value, m.unit AS unit,
           m.chart_only AS chart_only, m.cond_discharge_current_a AS discharge_a,
           m.cond_duration_min AS duration_min, m.cond_note AS note,
           m.source_fragment AS source_fragment,
           s.url AS url, s.retrieved AS retrieved
    ORDER BY property
"""

Q_SPEC = """
    MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(:Cell {model: $model})
    RETURN d.property AS property, d.value_a AS value_a, d.value_b AS value_b,
           d.source_a AS source_a, d.source_b AS source_b,
           d.verdict AS verdict, d.rationale AS detail,
           d.provenance AS provenance,
           d.fragment_a AS fragment_a, d.fragment_b AS fragment_b
    ORDER BY d.verdict, d.property
"""

Q_CAP_DISC = """
    MATCH (d:Discrepancy)-[:ABOUT]->(:Cell {model: $model})
    WHERE d.property CONTAINS 'capacity' AND d.kind IS NULL
      AND d.measured_value IS NOT NULL
    RETURN d.property AS property, d.claim_value AS claim,
           d.measured_value AS measured, d.claim_current_a AS claim_a,
           d.measured_current_a AS measured_a, d.relative_gap AS gap,
           d.conditions_comparable AS comparable, d.rationale AS rationale
"""

Q_MEAS = """
    MATCH (m:Measurement {metric: 'cycle_life_nominal'})-[:ABOUT]->
          (ci:CellInstance)-[:INSTANCE_OF]->(:Cell {model: $model})
    WHERE m.value IS NOT NULL
    RETURN m.value AS v
"""

Q_DISC = """
    MATCH (d:Discrepancy)-[:ABOUT]->(:Cell {model: $model})
    WHERE d.measured_median IS NOT NULL AND d.claim_value IS NOT NULL
    RETURN d.claim_value AS claim, d.measured_median AS med,
           d.measured_min AS mn, d.measured_max AS mx,
           d.measured_iqr_lo AS q1, d.measured_iqr_hi AS q3,
           d.n_measurements AS n, d.relative_gap AS gap,
           d.conditions_comparable AS comparable, d.rationale AS rationale,
           d.claim_conditions AS claim_conditions,
           d.measured_conditions AS measured_conditions
"""

# app/pages/*The_Graph.py (the graph-neighborhood page)
Q_INSTANCES = """
    MATCH (ci:CellInstance) RETURN ci.study_cell_id AS id ORDER BY id"""

Q_NEIGHBORS = """
    MATCH (a:CellInstance {study_cell_id: $center})-[r:SIMILAR_TO {view: $view}]->(b)
    OPTIONAL MATCH (b)<-[:ABOUT]-(m:Measurement {metric: 'cycle_life_nominal'})
    RETURN b.study_cell_id AS nbr, r.weight AS weight,
           r.same_policy_group AS same_group, m.value AS life,
           b.charge_policy_norm AS policy
    ORDER BY r.weight DESC LIMIT $k
"""

Q_CENTER_LIFE = """
    MATCH (:CellInstance {study_cell_id: $center})<-[:ABOUT]-
          (m:Measurement {metric: 'cycle_life_nominal'})
    RETURN m.value AS life"""

# app/pages/*Pipeline_Status.py
STATUS_LABELS = ["Cell", "CellInstance", "Chemistry", "Source", "Claim",
                 "Measurement", "Discrepancy"]

Q_EDGES = """
        MATCH ()-[r:SIMILAR_TO]->() RETURN r.view AS view, count(r) AS c"""

Q_CLAIMS_PER_CELL = """
        MATCH (cl:Claim)-[:ABOUT]->(c:Cell)
        RETURN c.model AS cell, count(cl) AS claims ORDER BY cell"""

# widget parameter space of the Graph Neighborhood page
VIEWS = ["behavior", "condition"]
K_RANGE = range(3, 11)                      # st.slider("neighbors", 3, 10)


def main() -> None:
    driver, err = make_driver()
    if driver is None or not hasattr(driver, "session"):
        raise SystemExit(f"needs a LIVE Neo4j (not the snapshot): {err}")

    entries: dict[str, list[dict]] = {}

    def record(query: str, **params) -> list[dict]:
        rows = run_query(driver, query, **params)
        entries[snapshot_key(query, params)] = rows
        return rows

    cells = record(Q_CELLS)
    for c in cells:
        model = c["model"]
        for q in (Q_CLAIMS, Q_LYGTE, Q_SPEC, Q_CAP_DISC, Q_MEAS, Q_DISC):
            record(q, model=model)

    instances = record(Q_INSTANCES)
    for inst in instances:
        center = inst["id"]
        record(Q_CENTER_LIFE, center=center)
        for view in VIEWS:
            for k in K_RANGE:
                record(Q_NEIGHBORS, center=center, view=view, k=k)

    for lb in STATUS_LABELS:
        record(f"MATCH (n:`{lb}`) RETURN count(n) AS c")
    record(Q_EDGES)
    record(Q_CLAIMS_PER_CELL)

    driver.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("recorded results of the exact Cypher the app pages issue; "
                 "served by app.common.SnapshotDriver when Neo4j is down"),
        "entries": entries,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(payload, separators=(",", ":"),
                                        default=str))
    size_mb = SNAPSHOT_PATH.stat().st_size / 1e6
    print(f"[snapshot] {len(entries)} query results -> {SNAPSHOT_PATH} "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
