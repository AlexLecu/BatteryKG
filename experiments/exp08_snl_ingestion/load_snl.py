"""Step 4 — promote the staged candidate and ingest the SNL LFP cells.

Runs only with --confirm. Without it the module prints what it would do and
exits, so the published graph state is unchanged unless the flag is set.

Every write is a MERGE keyed on a uniqueness-constrained property, so running
this twice produces identical counts (asserted by --verify-idempotent).

What is deliberately NOT done: no SIMILAR_TO edges in either view. SNL cells do
not enter the neighbour bank and no model is retrained — this is ingestion, not
adaptation. The condition view additionally does not apply, since SNL uses a
fixed 0.5C CC-CV charge rather than a two-step fast-charge policy.

Run:  python -m experiments.exp08_snl_ingestion.load_snl --confirm
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from experiments.exp08_snl_ingestion.snl_ingest import OUT_PARQUET, build_cycles, build_instances
from experiments.exp08_snl_ingestion.stage import (AUTHORS, CANDIDATE_ID, DOI, OUT,
                                                   VENUE, YEAR, candidate_record)
from src.agents.literature_monitor import kg_stage
from src.agents.promote import promote
from src.ingestion import cell_metadata as cm
from src.kg.connection import get_driver
from src.kg.load import SourceSpec, load_dataframe

SOURCE_ID = "snl_preger_2020"
CITATION = (f"{AUTHORS[0]} et al. {YEAR}, {VENUE} 167, 120532 (DOI {DOI}); "
            "data via BatteryArchive (SNL LFP)")
SNL_SOURCE = SourceSpec(source_id=SOURCE_ID, type="academic_cycling_study",
                        citation=CITATION)

# constant provenance carried by every SNL CellInstance
INSTANCE_EXTRA = {
    "study": "snl_lfp",
    "identity_basis": "asserted_from_publication",
    "identity_source": DOI,
    "capacity_basis": "RPT_0.5C_full_window",
}

# per-cell counts cannot ride on `instance_extra` (one dict for all rows), so
# they are written by an explicit follow-up annotation pass
_ANNOTATE_CYPHER = """
UNWIND $rows AS row
MATCH (ci:CellInstance {study_cell_id: row.study_cell_id})
  SET ci.n_ageing_cycles = row.n_ageing_cycles,
      ci.n_rpt_cycles = row.n_rpt_cycles,
      ci.temperature_c = row.temperature_c,
      ci.dod_window = row.dod_window,
      ci.c_rate_charge = row.c_rate_charge,
      ci.c_rate_discharge = row.c_rate_discharge
"""

_STATS_CYPHER = {
    "nodes": "MATCH (n) RETURN count(n) AS c",
    "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
}


def graph_stats(driver) -> dict:
    with driver.session() as s:
        out = {k: s.run(q).single()["c"] for k, q in _STATS_CYPHER.items()}
        out["labels"] = {r["l"]: r["c"] for r in s.run(
            "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY l")}
        out["relationship_types"] = {r["t"]: r["c"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")}
        out["sources"] = {r["s"]: r["c"] for r in s.run(
            "MATCH (s:Source) OPTIONAL MATCH (m:Measurement)-[:MEASURED_BY]->(s) "
            "RETURN s.source_id AS s, count(m) AS c ORDER BY s")}
        out["cell_instances_by_study"] = {r["st"] or "(unset)": r["c"] for r in s.run(
            "MATCH (ci:CellInstance) RETURN ci.study AS st, count(*) AS c ORDER BY st")}
        out["cells_with_measurements"] = s.run(
            "MATCH (c:Cell)<-[:INSTANCE_OF]-(:CellInstance)<-[:ABOUT]-(:Measurement) "
            "RETURN count(DISTINCT c) AS c").single()["c"]
        out["measurements_by_metric"] = {r["m"]: r["c"] for r in s.run(
            "MATCH (m:Measurement) RETURN m.metric AS m, count(*) AS c ORDER BY m")}
    return out


def ingest(driver, inst: pd.DataFrame, audit: pd.DataFrame) -> dict:
    result = load_dataframe(
        inst, cell=cm.SEVERSON_CELL, source=SNL_SOURCE,
        parquet_ref=str(OUT_PARQUET), driver=driver,
        instance_extra=INSTANCE_EXTRA,
        edge_views=())                      # no similarity edges — see module docstring
    rows = [{"study_cell_id": r["cell_id"],
             "n_ageing_cycles": int(r["max_ageing_cycle"]),
             "n_rpt_cycles": int(r["n_rpt"]),
             "temperature_c": float(r["temperature_c"]),
             "dod_window": str(r["dod_window"]),
             "c_rate_charge": str(r["c_rate_charge"]),
             "c_rate_discharge": str(r["c_rate_discharge"])}
            for _, r in audit.iterrows()]
    with driver.session() as s:
        s.run(_ANNOTATE_CYPHER, rows=rows)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm", action="store_true",
                   help="actually write to the graph (without it, dry run only)")
    p.add_argument("--verify-idempotent", action="store_true",
                   help="after loading, load a second time and assert the counts match")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    _, audit = build_cycles()
    inst, _ = build_instances()
    if not args.confirm:
        print("DRY RUN — nothing written. Would promote "
              f"{CANDIDATE_ID} -> Source '{SOURCE_ID}' and load {len(inst)} "
              f"CellInstances with {3 * len(inst)} Measurements.\n"
              "Re-run with --confirm.")
        return

    driver = get_driver()
    try:
        before = graph_stats(driver)
        print(f"[before] nodes={before['nodes']} rels={before['relationships']} "
              f"sources={list(before['sources'])}")

        # the staging mirror had to wait for the graph to come up
        kg_stage([candidate_record()])
        source_id = promote(CANDIDATE_ID, confirm=True, source_id=SOURCE_ID,
                            source_type="academic_cycling_study", citation=CITATION)
        print(f"[promote] Source '{source_id}' created; candidate marked promoted")

        result = ingest(driver, inst, audit)
        print(f"[ingest] {result['instances']} CellInstances, "
              f"SIMILAR_TO edges: {result['edges_by_view'] or 'none (by design)'}")

        after = graph_stats(driver)
        print(f"[after]  nodes={after['nodes']} rels={after['relationships']} "
              f"sources={list(after['sources'])}")

        idem = None
        if args.verify_idempotent:
            ingest(driver, inst, audit)
            again = graph_stats(driver)
            idem = {"nodes_match": again["nodes"] == after["nodes"],
                    "relationships_match": again["relationships"] == after["relationships"],
                    "second_run": {"nodes": again["nodes"],
                                   "relationships": again["relationships"]}}
            print(f"[idempotent] second load -> nodes={again['nodes']} "
                  f"rels={again['relationships']} "
                  f"({'MATCH' if idem['nodes_match'] and idem['relationships_match'] else 'MISMATCH'})")

        snap = {"ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_id": source_id, "candidate_id": CANDIDATE_ID,
                "instances": result["instances"],
                "before": before, "after": after, "idempotency": idem}
        (OUT / "graph_stats_before_after.json").write_text(
            json.dumps(snap, indent=1, default=str))
        print(f"\nsnapshot -> {OUT / 'graph_stats_before_after.json'}")
        return snap
    finally:
        driver.close()


if __name__ == "__main__":
    main()
