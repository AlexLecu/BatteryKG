"""Load lygte-info.dk independent-test measurements into the knowledge graph.

Per review: (Source {source_id: lygte_<cell>, type: 'independent_test', url,
retrieved}) and Measurement nodes attached at the CELL level (lygte tests one
or two physical specimens, not our study instances):

    (Measurement)-[:ABOUT]->(Cell)
    (Measurement)-[:MEASURED_BY]->(Source)

Chart-only measurement types are stored with value NULL and chart_only=true —
they record that the data exists but is inaccessible to text parsing (no image
extraction attempted). Idempotent: each source's measurements are
clean-rebuilt on load.

Run:  python -m src.kg.independent
"""
from __future__ import annotations

import json

from src.config import PROCESSED
from src.kg.connection import get_driver

MEASUREMENTS_JSON = PROCESSED / "lygte_measurements.json"

_SOURCE_CYPHER = """
MERGE (s:Source {source_id: $source_id})
  SET s.type = 'independent_test', s.url = $url, s.retrieved = $retrieved,
      s.citation = $title, s.n_specimens = $n_specimens
"""

_CLEAR_CYPHER = """
MATCH (m:Measurement)-[:MEASURED_BY]->(:Source {source_id: $source_id})
DETACH DELETE m
"""

_MEASUREMENT_CYPHER = """
UNWIND $rows AS row
MATCH (c:Cell {model: $model})
MATCH (s:Source {source_id: $source_id})
MERGE (m:Measurement {measurement_id: row.measurement_id})
  SET m.metric = row.property,
      m.value = row.value,
      m.unit = row.unit,
      m.chart_only = row.chart_only,
      m.source_fragment = row.source_fragment
  SET m += row.conditions
MERGE (m)-[:ABOUT]->(c)
MERGE (m)-[:MEASURED_BY]->(s)
"""


def load_independent(driver=None, json_path=MEASUREMENTS_JSON) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path.name} missing — run: python -m src.ingestion.lygte")
    data = json.loads(json_path.read_text())
    own = driver is None
    driver = driver or get_driver()
    counts: dict[str, int] = {}
    try:
        with driver.session() as s:
            for review in data["reviews"]:
                sid = review["source_id"]
                s.run(_SOURCE_CYPHER, source_id=sid, url=review["url"],
                      retrieved=review["retrieved"], title=review["title"],
                      n_specimens=review["n_specimens"])
                rows = []
                seen: dict[str, int] = {}
                for meas in review["measurements"]:
                    prop = meas["property"]
                    seen[prop] = seen.get(prop, 0) + 1
                    cond = {f"cond_{k}": v for k, v in (meas.get("conditions") or {}).items()
                            if v is not None}
                    rows.append({
                        "measurement_id": f"{sid}:{prop}:{seen[prop]}",
                        "property": prop,
                        "value": meas["value"],
                        "unit": meas.get("unit"),
                        "chart_only": bool(meas["chart_only"]),
                        "source_fragment": meas.get("source_fragment"),
                        "conditions": cond,
                    })
                s.run(_CLEAR_CYPHER, source_id=sid)        # clean-rebuild per source
                s.run(_MEASUREMENT_CYPHER, rows=rows,
                      model=review["cell_model"], source_id=sid)
                counts[review["cell_model"]] = len(rows)
                print(f"[kg.independent] {review['cell_model']}: {len(rows)} "
                      f"measurements from {sid}")
    finally:
        if own:
            driver.close()
    return counts


if __name__ == "__main__":
    load_independent()
