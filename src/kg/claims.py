"""Load manufacturer claims (data/claims/*.yaml) into the knowledge graph.

Creates, per datasheet:
  (Source {source_id, type:'manufacturer_datasheet', url, retrieved, document})
  (Claim)-[:ABOUT]->(Cell)
  (Claim)-[:ASSERTED_BY]->(Source)

Claims are derived data (from the YAML gold standard), so each source's claims
are clean-rebuilt on load: stale claims deleted, current ones MERGEd. Running
twice yields identical counts.

Run:  python -m src.kg.claims
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.config import DATA
from src.ingestion import cell_metadata as cm
from src.kg.connection import get_driver

CLAIMS_DIR = DATA / "claims"

# Cell.model -> commercial metadata (chemistry, nominal, form factor)
_CELL_META = {cm.SEVERSON_CELL.model: cm.SEVERSON_CELL}
for _c in cm.SANDIA_CELLS.values():
    if _c is not None:
        _CELL_META.setdefault(_c.model, _c)

_CELL_CYPHER = """
MERGE (c:Cell {model: $model})
  SET c.manufacturer = $manufacturer
FOREACH (_ IN CASE WHEN $chemistry IS NULL THEN [] ELSE [1] END |
  MERGE (ch:Chemistry {name: $chemistry})
  MERGE (c)-[:HAS_CHEMISTRY]->(ch)
  SET c.form_factor = $form_factor,
      c.nominal_capacity_ah = $nominal
)
MERGE (s:Source {source_id: $source_id})
  SET s.type = 'manufacturer_datasheet',
      s.url = $url, s.retrieved = $retrieved, s.document = $document,
      s.citation = $citation
"""

_CLEAR_CLAIMS = """
MATCH (cl:Claim)-[:ASSERTED_BY]->(:Source {source_id: $source_id})
DETACH DELETE cl
"""

_CLAIM_CYPHER = """
UNWIND $claims AS cl
MATCH (c:Cell {model: $model})
MATCH (s:Source {source_id: $source_id})
MERGE (n:Claim {claim_id: cl.claim_id})
  SET n.property = cl.property,
      n.value = cl.value,
      n.unit = cl.unit,
      n.page = cl.page,
      n.conditions_text = cl.conditions_text,
      n.notes = cl.notes
  SET n += cl.parsed_conditions
MERGE (n)-[:ABOUT]->(c)
MERGE (n)-[:ASSERTED_BY]->(s)
"""


def _source_id(doc: dict) -> str:
    stem = Path(doc["source_document"]).stem
    return f"datasheet_{stem.replace('-', '_')}"


def _claim_rows(doc: dict, source_id: str) -> list[dict]:
    rows = []
    seen: dict[str, int] = {}
    for cl in doc["claims"]:
        prop = cl["property"]
        seen[prop] = seen.get(prop, 0) + 1
        cond = cl.get("stated_conditions") or {}
        parsed = {f"cond_{k}": v for k, v in cond.items() if k != "text"}
        value = cl["value"]
        rows.append({
            "claim_id": f"{source_id}:{prop}:{seen[prop]}",
            "property": prop,
            "value": value,          # scalar or [lo, hi] list — Neo4j stores both
            "unit": cl.get("unit"),
            "page": cl.get("page"),
            "conditions_text": cond.get("text", "unspecified"),
            "notes": cl.get("notes"),
            "parsed_conditions": parsed,
        })
    return rows


def load_claims(driver=None, claims_dir: Path = CLAIMS_DIR) -> dict:
    files = sorted(claims_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no claim YAML files under {claims_dir}")
    own = driver is None
    driver = driver or get_driver()
    counts: dict[str, int] = {}
    try:
        with driver.session() as s:
            for path in files:
                doc = yaml.safe_load(path.read_text())
                model = doc["cell_model"]
                sid = _source_id(doc)
                meta = _CELL_META.get(model)
                s.run(_CELL_CYPHER, model=model,
                      manufacturer=doc["manufacturer"],
                      chemistry=meta.chemistry if meta else None,
                      form_factor=meta.form_factor if meta else None,
                      nominal=(float(meta.nominal_capacity_ah)
                               if meta and meta.nominal_capacity_ah == meta.nominal_capacity_ah
                               else None),
                      source_id=sid, url=doc.get("source_url"),
                      retrieved=str(doc.get("retrieved")),
                      document=doc.get("source_document"),
                      citation=f"{doc['manufacturer']} datasheet, {doc['cell_model']}")
                rows = _claim_rows(doc, sid)
                s.run(_CLEAR_CLAIMS, source_id=sid)     # clean-rebuild per source
                s.run(_CLAIM_CYPHER, claims=rows, model=model, source_id=sid)
                counts[model] = len(rows)
                print(f"[kg.claims] {model}: {len(rows)} claims from {path.name}")
    finally:
        if own:
            driver.close()
    return counts


if __name__ == "__main__":
    load_claims()
