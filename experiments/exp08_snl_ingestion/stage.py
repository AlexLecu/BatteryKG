"""Steps 1-3 of the SNL LFP update cycle: discovery/staging, entity matching,
and the human-gate promotion summary.

This module writes NOTHING to the knowledge graph beyond the staging mirror
(:CandidateSource {status:'pending_review'}), which is a deliberately separate
label from real :Source nodes. Promotion and ingestion are separate, explicitly
confirmed steps (load_snl.py).

Run:  python -m experiments.exp08_snl_ingestion.stage
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from src.agents.literature_monitor import (CANDIDATES_JSONL, MIN_YEAR, dedup_key,
                                           kg_stage, stage_candidates)
from src.config import DATA, OUTPUTS, PROCESSED, ROOT
from src.ingestion import cell_metadata as cm

OUT = OUTPUTS / "experiment_08_snl_ingestion"
CLAIMS_YAML = DATA / "claims" / "a123_apr18650m1a.yaml"

# --- the study being registered ----------------------------------------------
DOI = "10.1149/1945-7111/abae37"
TITLE = ("Degradation of Commercial Lithium-Ion Cells as a Function of "
         "Chemistry and Cycling Conditions")
AUTHORS = ["Yuliya Preger", "Heather M. Barkholtz", "Armando Fresquez",
           "Daniel L. Campbell", "Benjamin W. Juba", "Jessica Romàn-Kustas",
           "Summer R. Ferreira", "Babu Chalamala"]
VENUE = "Journal of The Electrochemical Society"
YEAR = 2020
ABSTRACT = (
    "Sandia National Laboratories cycled commercial 18650 cells of three "
    "chemistries — LFP from A123 Systems (Part #APR18650M1A, 1.1 Ah), NCA from "
    "Panasonic (Part #NCR18650B, 3.2 Ah) and NMC from LG Chem (Part #18650HG2, "
    "3 Ah) — across temperature, depth-of-discharge and discharge-rate "
    "conditions, reporting how each factor drives capacity fade. Per-cycle and "
    "time-series data are published through BatteryArchive.")
# The publication sentence that carries the cell identity. Quoted verbatim from
# the article page; this is the provenance for the entity match.
IDENTITY_QUOTE = ("LFP from A123 Systems (Part #APR18650M1A, 1.1 Ah), NCA from "
                  "Panasonic (Part #NCR18650B, 3.2 Ah), and NMC from LG Chem "
                  "(Part #18650HG2, 3 Ah)")

CANDIDATE_ID = dedup_key(DOI, None, TITLE)
DATA_FILES = "BatteryArchive/SNL LFP.zip (30 cells x cycle_data + timeseries CSV)"


def candidate_record() -> dict:
    """A staging record in the Literature Monitor's own schema.

    Registered manually rather than by an API sweep, and the record says so:
    `api` is 'manual_registration' and the triage verdict is attributed to a
    human, not to the LLM triage step. The monitor's automated path could not
    have produced this hit — it filters to year >= {MIN_YEAR} and this study is
    from 2020 — which is precisely the gap manual registration exists to cover.
    """
    return {
        "candidate_id": CANDIDATE_ID,
        "title": TITLE,
        "authors": AUTHORS,
        "venue": VENUE,
        "year": YEAR,
        "doi": DOI,
        "arxiv_id": None,
        "abstract": ABSTRACT,
        "matched_target": cm.SEVERSON_CELL.model,
        "matched_query": "manual registration (BatteryArchive SNL LFP dataset)",
        "api": "manual_registration",
        "triage_verdict": "yes",
        "triage_by": "human (manual registration; LLM triage not invoked)",
        "triage_rationale": (
            "Names the A123 APR18650M1A explicitly and publishes per-cycle "
            "cycling data for 30 LFP cells through BatteryArchive; same "
            "commercial cell as the Severson instances already in the graph, "
            "under an independent laboratory and a different protocol family "
            "(fixed 0.5C charge, varied temperature/DoD/discharge rate)."),
        "data_files": DATA_FILES,
        "min_year_note": (
            f"outside the monitor's automated window (MIN_YEAR={MIN_YEAR})"),
    }


# --- step 2: entity matching ---------------------------------------------------
def matching_evidence(inst: pd.DataFrame) -> dict:
    """Evidence for matching the SNL cells to the existing Cell node.

    Deliberately separates what the DATA shows (an electrical signature
    consistent with the cell) from what the PUBLICATION asserts (the model
    string). The files themselves carry no manufacturer or model string, so the
    identity is asserted-from-publication and is recorded that way — the
    recommendation the feasibility report made.
    """
    claims = yaml.safe_load(CLAIMS_YAML.read_text())
    by_prop = {c["property"]: c for c in claims["claims"]}
    cycles = pd.read_parquet(PROCESSED / "snl_lfp_cycles.parquet")
    return {
        "target_cell_node": {
            "model": cm.SEVERSON_CELL.model,
            "manufacturer": cm.SEVERSON_CELL.manufacturer,
            "chemistry": cm.SEVERSON_CELL.chemistry,
            "nominal_capacity_ah": cm.SEVERSON_CELL.nominal_capacity_ah,
            "form_factor": cm.SEVERSON_CELL.form_factor,
        },
        "asserted_from_publication": {
            "model_string": "A123 APR18650M1A",
            "quote": IDENTITY_QUOTE,
            "citation": f"{AUTHORS[0]} et al. {YEAR}, {VENUE} 167, 120532",
            "doi": DOI,
            "verified": "DOI resolved and the sentence read from the article page",
        },
        "corroborating_signature_from_data": {
            "form_factor": "18650 (encoded in every filename: SNL_18650_LFP_*)",
            "chemistry_tag": "LFP (filename) — consistent with the flat ~3.2 V plateau",
            "nominal_capacity_ah_claimed": by_prop["nominal_capacity_ah"]["value"],
            "measured_initial_capacity_ah": [
                round(float(inst["initial_capacity_ah"].min()), 4),
                round(float(inst["initial_capacity_ah"].max()), 4)],
            "charge_cutoff_v_datasheet": by_prop["charge_voltage_v"]["value"],
            "discharge_cutoff_v_datasheet": by_prop["discharge_cutoff_v"]["value"],
            "note": ("measured initial capacity sits just below the 1.1 Ah "
                     "nameplate for every cell — consistent with this cell, and "
                     "itself the discrepancy this ingestion surfaces"),
        },
        "provenance_decision": {
            "identity_basis": "asserted_from_publication",
            "rationale": ("the CSV files contain no manufacturer or model "
                          "string; grep over the unpacked archive returns no "
                          "match for 'A123', 'APR18650' or 'LFP' outside the "
                          "filenames. The match therefore rests on the "
                          "publication, and the graph must record that rather "
                          "than implying the dataset self-identifies."),
            "recorded_as": ("CellInstance.identity_basis = "
                            "'asserted_from_publication', with "
                            "identity_source = the DOI above"),
        },
        "cells_matched": int(len(inst)),
        "rows_ingested": int(len(cycles)),
    }


# --- step 3: promotion summary --------------------------------------------------
def promotion_summary(inst: pd.DataFrame, audit: pd.DataFrame, ev: dict) -> dict:  # noqa: D401
    n = len(inst)
    reached = inst[inst["reached_eol_nominal"]]
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate": "AWAITING HUMAN APPROVAL — nothing has been written to the graph",
        "source_to_be_promoted": {
            "candidate_id": CANDIDATE_ID,
            "source_id_on_promotion": "snl_preger_2020",
            "citation": f"{AUTHORS[0]} et al. {YEAR}, {VENUE} 167, 120532 "
                        f"(DOI {DOI}); data via BatteryArchive",
            "type": "academic_cycling_study",
            "data_files": DATA_FILES,
        },
        "cells_to_be_added": {
            "count": n,
            "attach_to_existing_cell_node": ev["target_cell_node"]["model"],
            "new_cell_nodes_created": 0,
            "identity_basis": "asserted_from_publication",
            "policy_groups": int(inst["policy_group_id"].nunique()),
            "conditions": {
                "temperature_c": sorted(inst["temperature_c"].unique().tolist()),
                "dod_windows": sorted(inst["dod_window"].unique().tolist()),
                "discharge_rates": sorted(inst["c_rate_discharge"].unique().tolist()),
                "charge": "0.5C CC-CV (fixed for all 30 cells)",
            },
            "ids": inst["study_cell_id"].tolist(),
        },
        "measurements_to_be_created": {
            "per_cell": ["cycle_life_nominal", "cycle_life_initial",
                         "initial_capacity_ah"],
            "total": 3 * n,
            "valued_vs_censored": {
                "cycle_life_nominal": {
                    "valued": int(inst["reached_eol_nominal"].sum()),
                    "right_censored": int((~inst["reached_eol_nominal"]).sum())},
                "cycle_life_initial": {
                    "valued": int(inst["reached_eol_initial"].sum()),
                    "right_censored": int((~inst["reached_eol_initial"]).sum())},
                "initial_capacity_ah": {"valued": n, "right_censored": 0},
            },
            "initial_capacity_range_ah": [
                round(float(inst["initial_capacity_ah"].min()), 4),
                round(float(inst["initial_capacity_ah"].max()), 4)],
            "cycle_life_nominal_of_the_valued": sorted(
                round(float(x)) for x in reached["cycle_life_nominal"]),
        },
        "provenance_annotations": {
            "every_measurement": "(:Measurement)-[:MEASURED_BY]->(:Source snl_preger_2020)",
            "parquet_ref": str(PROCESSED / "snl_lfp_cycles.parquet"),
            "identity": "CellInstance.identity_basis='asserted_from_publication'"
                        f", identity_source='{DOI}'",
            "preprocessing_recorded_on_instances": [
                "capacity_basis='RPT_0.5C_full_window'",
                "n_rpt_cycles", "n_ageing_cycles"],
        },
        "similar_to_edges": {
            "condition_view": "NOT built — SNL has no two-step fast-charge policy "
                              "(fixed 0.5C CC-CV), so the condition features are null",
            "behavior_view": "NOT built — deliberately. SNL cells are not entering "
                             "the neighbour bank and no model is retrained; this is "
                             "ingestion, not adaptation.",
        },
        "preprocessing_decisions": {
            "capacity_series": (
                "RPT cycles only (full 0-100% window at 0.5C): "
                f"{int(audit['n_rpt'].sum())} RPT cycles identified, "
                f"{int(audit['capacity_over_nominal_flagged'].sum())} dropped by the "
                f"over-nominal screen, {ev['rows_ingested']} rows ingested; "
                f"{int(audit['n_cycling'].sum())} partial-DoD/high-rate ageing "
                "cycles excluded from the capacity series"),
            "over_nominal_screen": f"{int(audit['capacity_over_nominal_flagged'].sum())} "
                                   "RPT cycles dropped for capacity >110% of nominal "
                                   "(cycle-level symptom of a merged double discharge)",
            "gap_split_60s": "not applicable — no Q(V) inversion is performed "
                             "(see snl_ingest.py docstring)",
            "cycle_index": "original ageing cycle number preserved",
        },
        "review_flags": review_flags(inst, audit),
    }


def review_flags(inst: pd.DataFrame, audit: pd.DataFrame) -> list[dict]:
    """Things the human gate should see before approving."""
    flags = []
    short = inst[inst["qc_flag"].str.contains("short_series", na=False)]
    if len(short):
        flags.append({
            "flag": "spurious 'short_series' QC flag",
            "detail": f"{len(short)} cells carry it because the QC rule counts "
                      "series rows (<50) and the series is RPTs, not ageing "
                      "cycles; these cells ran 2958-8395 ageing cycles",
            "recommendation": "ingest as-is but also write n_ageing_cycles and "
                              "n_rpt_cycles onto each instance so the record is "
                              "self-explanatory; do not suppress the raw flag",
        })
    anom = inst[inst["is_anomalous"]]
    if len(anom):
        flags.append({
            "flag": "QC-anomalous cell",
            "detail": f"{anom['study_cell_id'].tolist()} flagged "
                      f"{anom['qc_flag'].tolist()}",
            "recommendation": "load it (the graph records QC state rather than "
                              "hiding it); is_anomalous excludes it from modelling "
                              "downstream exactly as for Severson",
        })
    flags.append({
        "flag": "EOL count differs from the feasibility report",
        "detail": "6 of 30 cells reach 80% of nominal under the project's "
                  "standard cycle_life_table(); the feasibility report said 5, "
                  "using a stricter bespoke rule (3 consecutive RPTs below "
                  "threshold). The extra cell is SNL_18650_LFP_35C_0-100_0.5-1C_b "
                  "(final SoH 80.8%, dips transiently below). Cycle-life VALUES "
                  "also differ by 0-280 cycles for the same reason.",
        "recommendation": "use the standard definition, for consistency with the "
                          "Severson and HUST instances already in the graph; note "
                          "the difference in the paper rather than re-deriving",
    })
    return flags


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from experiments.exp08_snl_ingestion.snl_ingest import build_cycles, build_instances

    print("=" * 78)
    print("STEP 1 — DISCOVERY / STAGING")
    print("=" * 78)
    rec = candidate_record()
    new = stage_candidates([rec])
    if new:
        print(f"staged NEW candidate: {rec['candidate_id']}")
    else:
        print(f"candidate already staged (idempotent): {rec['candidate_id']}")
    print(f"  canonical store : {CANDIDATES_JSONL}")
    print(f"  title           : {rec['title']}")
    print(f"  venue/year      : {rec['venue']} ({rec['year']})   DOI {rec['doi']}")
    print(f"  registered via  : {rec['api']} — {rec['min_year_note']}")
    print(f"  triage          : [{rec['triage_verdict']}] by {rec['triage_by']}")
    print(f"                    {rec['triage_rationale']}")
    kg_ok = kg_stage([rec])
    print(f"  KG staging mirror (:CandidateSource pending_review): "
          f"{'ok' if kg_ok else 'UNAVAILABLE — JSONL remains canonical'}")

    print()
    print("=" * 78)
    print("STEP 2 — ENTITY MATCHING")
    print("=" * 78)
    _, audit = build_cycles()
    inst, _ = build_instances()
    ev = matching_evidence(inst)
    print(f"matching {ev['cells_matched']} SNL cells -> existing Cell node "
          f"'{ev['target_cell_node']['model']}'")
    a = ev["asserted_from_publication"]
    print(f"  publication states : \"{a['quote'][:66]}...\"")
    print(f"  citation           : {a['citation']}")
    print(f"  verification       : {a['verified']}")
    c = ev["corroborating_signature_from_data"]
    print(f"  corroboration      : {c['form_factor']}")
    print(f"                       measured initial capacity "
          f"{c['measured_initial_capacity_ah'][0]}-{c['measured_initial_capacity_ah'][1]} Ah "
          f"vs {c['nominal_capacity_ah_claimed']} Ah nameplate")
    p = ev["provenance_decision"]
    print(f"  identity basis     : {p['identity_basis']}")
    print(f"                       {p['rationale'][:70]}...")

    print()
    print("=" * 78)
    print("STEP 3 — HUMAN VALIDATION GATE")
    print("=" * 78)
    summary = promotion_summary(inst, audit, ev)
    _print_summary(summary)

    (OUT / "promotion_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT / "matching_evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    inst.to_csv(OUT / "snl_instances_pending.csv", index=False)
    audit.to_csv(OUT / "snl_preprocessing_audit.csv", index=False)
    print(f"\nsnapshot -> {OUT / 'promotion_summary.json'}")
    return summary, inst, audit, ev


def _print_summary(s: dict) -> None:
    print(f"\n*** {s['gate']} ***\n")
    src = s["source_to_be_promoted"]
    print("SOURCE")
    print(f"  {src['candidate_id']}  ->  Source '{src['source_id_on_promotion']}'")
    print(f"  {src['citation']}")
    cells = s["cells_to_be_added"]
    print(f"\nCELLS  ({cells['count']} CellInstances, "
          f"{cells['new_cell_nodes_created']} new Cell nodes)")
    print(f"  attach to        : {cells['attach_to_existing_cell_node']}")
    print(f"  identity basis   : {cells['identity_basis']}")
    print(f"  conditions       : T {cells['conditions']['temperature_c']} degC, "
          f"DoD {cells['conditions']['dod_windows']}, "
          f"discharge {cells['conditions']['discharge_rates']}")
    print(f"                     charge {cells['conditions']['charge']}")
    print(f"  policy groups    : {cells['policy_groups']}")
    m = s["measurements_to_be_created"]
    print(f"\nMEASUREMENTS  ({m['total']} nodes = {len(m['per_cell'])} x {cells['count']})")
    for metric, vc in m["valued_vs_censored"].items():
        print(f"  {metric:22s} {vc['valued']:3d} valued, "
              f"{vc['right_censored']:3d} right-censored")
    print(f"  initial capacity range : {m['initial_capacity_range_ah']} Ah")
    print(f"  cycle lives (valued)   : {m['cycle_life_nominal_of_the_valued']}")
    print("\nPROVENANCE")
    for k, v in s["provenance_annotations"].items():
        print(f"  {k}: {v}")
    print("\nPREPROCESSING")
    for k, v in s["preprocessing_decisions"].items():
        print(f"  {k}: {v}")
    print("\nSIMILAR_TO EDGES")
    for k, v in s["similar_to_edges"].items():
        print(f"  {k}: {v}")
    print("\nREVIEW FLAGS")
    for f in s["review_flags"]:
        print(f"  [!] {f['flag']}")
        print(f"      {f['detail']}")
        print(f"      -> {f['recommendation']}")


if __name__ == "__main__":
    main()
