"""BatteryKG knowledge-graph schema.

Declares the node labels and relationship types of the graph and creates the
uniqueness constraints that pin entity identity. Idempotent: every statement
uses `IF NOT EXISTS`, so this is safe to run repeatedly.

Run:  python -m src.kg.schema
"""
from __future__ import annotations

from neo4j import Driver

from src.kg.connection import get_driver

# --- Ontology -------------------------------------------------------------
# Node labels:
#   Cell         — a commercial cell model (the "claim" anchor), e.g. A123 APR18650M1A
#   CellInstance — one physical cell tested in a study (e.g. Severson b1c0)
#   Chemistry    — LFP / NCA / NMC / LCO ...
#   Source       — a datasheet, dataset, or independent test (provenance)
#   Claim        — a manufacturer-asserted value (e.g. rated cycle life)
#   Measurement  — an observed/measured value (e.g. measured cycle life)
#   Discrepancy  — a reconciled claim-vs-measured conflict
NODE_LABELS = [
    "Cell", "CellInstance", "Chemistry", "Source",
    "Claim", "Measurement", "Discrepancy",
]

# Relationship types:
#   (CellInstance)-[:INSTANCE_OF]->(Cell)
#   (Cell)-[:HAS_CHEMISTRY]->(Chemistry)
#   (Claim|Measurement|Discrepancy)-[:ABOUT]->(Cell|CellInstance)
#   (Claim)-[:ASSERTED_BY]->(Source)
#   (Measurement)-[:MEASURED_BY]->(Source)
#   (Discrepancy)-[:CONTRASTS]->(Claim|Measurement)
#   (Cell)-[:SIMILAR_TO]->(Cell)            # graph-mediated transfer
RELATIONSHIP_TYPES = [
    "INSTANCE_OF", "HAS_CHEMISTRY", "ABOUT",
    "ASSERTED_BY", "MEASURED_BY", "CONTRASTS", "SIMILAR_TO",
]

# (constraint_name, node_label, property) — uniqueness constraints that define
# entity identity for idempotent MERGE-based loading.
UNIQUENESS_CONSTRAINTS = [
    ("cell_model_unique", "Cell", "model"),
    ("cellinstance_study_cell_id_unique", "CellInstance", "study_cell_id"),
    ("chemistry_name_unique", "Chemistry", "name"),
    ("source_source_id_unique", "Source", "source_id"),
    ("claim_id_unique", "Claim", "claim_id"),
    ("discrepancy_id_unique", "Discrepancy", "discrepancy_id"),
    # staging area for the Literature Monitor — deliberately a separate label
    # from Source; promotion to a real Source is a human CLI action
    ("candidate_id_unique", "CandidateSource", "candidate_id"),
]


def apply_schema(driver: Driver | None = None) -> None:
    """Create all uniqueness constraints (idempotent)."""
    own = driver is None
    driver = driver or get_driver()
    try:
        with driver.session() as session:
            for name, label, prop in UNIQUENESS_CONSTRAINTS:
                session.run(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE n.`{prop}` IS UNIQUE"
                )
                print(f"[schema] ensured {name}: ({label}.{prop} IS UNIQUE)")
    finally:
        if own:
            driver.close()


def show_constraints(driver: Driver | None = None) -> list[dict]:
    """Return the current constraints (SHOW CONSTRAINTS)."""
    own = driver is None
    driver = driver or get_driver()
    try:
        with driver.session() as session:
            return [r.data() for r in session.run("SHOW CONSTRAINTS")]
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    apply_schema()
    print("\nSHOW CONSTRAINTS:")
    rows = show_constraints()
    for c in sorted(rows, key=lambda r: r.get("name", "")):
        print(f"  {c.get('name'):40s} {c.get('type'):24s} "
              f"{c.get('labelsOrTypes')} {c.get('properties')}")
    print(f"\n{len(rows)} constraint(s) total.")
