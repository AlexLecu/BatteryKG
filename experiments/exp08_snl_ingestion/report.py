"""Step 6 — the walkthrough the paper section can be written from.

Reads the artifacts the earlier steps wrote plus the live graph, and formats.
Computes no new science.

Run:  python -m experiments.exp08_snl_ingestion.report
"""
from __future__ import annotations

import json

import pandas as pd

from experiments.exp08_snl_ingestion.load_snl import graph_stats
from experiments.exp08_snl_ingestion.stage import CANDIDATE_ID, DOI, OUT
from src.kg.connection import get_driver


def main() -> None:
    stats = json.loads((OUT / "graph_stats_before_after.json").read_text())
    disc = json.loads((OUT / "discrepancies.json").read_text())
    cov = json.loads((OUT / "snl_coverage_summary.json").read_text())
    promo = json.loads((OUT / "promotion_summary.json").read_text())
    ev = json.loads((OUT / "matching_evidence.json").read_text())
    audit = pd.read_csv(OUT / "snl_preprocessing_audit.csv")
    inst = pd.read_csv(OUT / "snl_instances_pending.csv")

    driver = get_driver()
    try:
        final = graph_stats(driver)
    finally:
        driver.close()
    before, after = stats["before"], stats["after"]

    L, a = [], None
    a = L.append
    a("# End-to-end update cycle: adding SNL LFP to the knowledge graph")
    a("")
    a("A full pass of the staged, human-gated ingestion pipeline on a source the "
      "graph had never seen: discovery → entity matching → human validation → "
      "graph update → changed outputs. The new source is the Sandia National "
      "Laboratories LFP cycling study (Preger et al. 2020), 30 cells of the A123 "
      "APR18650M1A — the same commercial cell the graph already holds Severson and "
      "HUST instances of, under a third laboratory and a third protocol family.")
    a("")
    a("Every step below was executed; nothing is illustrative. The human gate was "
      "real: the pipeline stopped, printed a promotion summary, and did not proceed "
      "until approved.")
    a("")

    # ---------------------------------------------------------------- 1
    a("## 1. Discovery and staging")
    a("")
    a(f"The study was registered through the Literature Monitor's staging path as "
      f"candidate `{CANDIDATE_ID}`, in the monitor's own record schema, with "
      f"`status='pending_review'`.")
    a("")
    a("| field | value |")
    a("|---|---|")
    a(f"| title | {promo['source_to_be_promoted']['citation'].split('(DOI')[0].strip()} |")
    a(f"| DOI | `{DOI}` (resolved and verified) |")
    a("| registered via | `manual_registration` |")
    a("| triage | `yes`, attributed to a human — the LLM triage step was not invoked |")
    a("| data | `BatteryArchive/SNL LFP.zip` |")
    a("")
    a("Two honest notes about this step. First, the monitor's **automated sweep could "
      "not have found this study**: it filters to publication year ≥ 2024 and this is "
      "2020. Manual registration exists for exactly that gap, and the record says so "
      "rather than implying an API hit. Second, the record is stored in "
      "`data/monitor/candidates.jsonl`, which is canonical, and mirrored into the "
      "graph as a `:CandidateSource` node — a label deliberately distinct from a real "
      "`:Source`. At first attempt the graph was unreachable and the mirror was "
      "skipped with a warning while the JSONL still recorded the candidate; that "
      "fail-soft path is by design and it worked.")
    a("")

    # ---------------------------------------------------------------- 2
    a("## 2. Entity matching")
    a("")
    a("All 30 cells were matched to the **existing** `A123 APR18650M1A` Cell node. "
      "No new Cell node was created. The evidence is deliberately split into what the "
      "publication *asserts* and what the data merely *corroborates*.")
    a("")
    a("**Asserted (the basis of the match).** The article states verbatim:")
    a("")
    a(f"> {ev['asserted_from_publication']['quote']}")
    a("")
    a(f"— {ev['asserted_from_publication']['citation']}")
    a("")
    a("**Corroborating signature from the data.** 18650 form factor encoded in every "
      "filename; LFP tag consistent with the flat ~3.2 V plateau; measured initial "
      f"capacity {ev['corroborating_signature_from_data']['measured_initial_capacity_ah'][0]}–"
      f"{ev['corroborating_signature_from_data']['measured_initial_capacity_ah'][1]} Ah "
      f"against a {ev['corroborating_signature_from_data']['nominal_capacity_ah_claimed']} Ah "
      "nameplate; 3.6 V / 2.0 V window matching the datasheet's stated cut-offs.")
    a("")
    a("**Provenance decision.** The CSV files carry no manufacturer or model string "
      "anywhere. The identity therefore rests on the publication, and the graph "
      "records that rather than implying the dataset self-identifies. Every SNL "
      "instance carries:")
    a("")
    a("```")
    a("identity_basis   = 'asserted_from_publication'")
    a(f"identity_source  = '{DOI}'")
    a("```")
    a("")

    # ---------------------------------------------------------------- 3
    a("## 3. Human validation gate")
    a("")
    a("The pipeline halted and printed a promotion summary — source, cells, "
      "measurements, provenance annotations, preprocessing decisions and review "
      "flags — then exited without writing. Ingestion runs only behind `--confirm`. "
      "Full snapshot: `promotion_summary.json`.")
    a("")
    a("Three items were escalated for a decision rather than resolved silently:")
    a("")
    a("| flag | resolution |")
    a("|---|---|")
    a("| **6 of 30 cells reach 80 % of nominal**, not the 5 in the earlier "
      "feasibility report — that report used a stricter bespoke rule (3 consecutive "
      "RPTs below threshold); the standard `cycle_life_table()` interpolates the "
      "first crossing. The extra cell is `35C_0-100_0.5-1C_b` (final SoH 80.8 %). "
      "Cycle-life *values* also differ by 0–280 cycles. | Approved: use the standard "
      "definition, for consistency with the Severson and HUST instances already in "
      "the graph. |")
    a("| **10 cells carry a spurious `short_series` QC flag** — the rule fires below "
      "50 series rows and the series is RPTs, not ageing cycles. | Approved: ingest "
      "as-is, retain the raw flag, and additionally write `n_ageing_cycles` and "
      "`n_rpt_cycles` onto every instance so the record explains itself. |")
    a("| **One QC-anomalous cell** (`25C_20-80_0.5-0.5C_b`, `capacity_spike_down`). | "
      "Approved: load it with its flag — the graph records QC state rather than "
      "hiding it. |")
    a("")

    # ---------------------------------------------------------------- 4
    a("## 4. Ingestion")
    a("")
    a("Loaded through the standard `src.kg.load.load_dataframe` path (MERGE-only, "
      "uniqueness-constrained keys). Re-running produced identical node and "
      "relationship counts — asserted, not assumed, by `--verify-idempotent`.")
    a("")
    a("### Preprocessing decisions")
    a("")
    pp = promo["preprocessing_decisions"]
    a("| decision | detail |")
    a("|---|---|")
    a(f"| capacity series | {pp['capacity_series']} |")
    a(f"| over-nominal screen | {pp['over_nominal_screen']} |")
    a(f"| gap-split (>60 s) | {pp['gap_split_60s']} |")
    a(f"| cycle index | {pp['cycle_index']} |")
    a("")
    a("The **RPT restriction is the load-bearing decision**. SNL interleaves "
      "full-window 0.5C Reference Performance Tests into an ageing schedule that may "
      "run at partial depth of discharge (20–80 %, 40–60 %) and up to 3C. A "
      "partial-DoD cell never discharges fully, so its cycling-cycle capacity is not "
      "comparable to nameplate — fed to the standard cycle-life routine it would "
      "\"reach\" 80 % of nominal at cycle 4. Capacity is therefore read from RPT "
      "cycles only, which is also the basis a datasheet capacity number would be "
      "measured on.")
    a("")
    a("The >60 s gap-split the feasibility report requires is **not** applied here, "
      "and the module says so rather than claiming a step it did not take: that "
      "defect corrupts Q(V) inversion in the *timeseries* files, and this ingestion "
      "reads per-cycle tables and inverts no discharge curve. Its cycle-level "
      "symptom was screened for explicitly — 8 RPT cycles exceeding 110 % of nominal "
      "(merged double discharges) were dropped. The gap-split does become necessary "
      "in §5c, where it is applied.")
    a("")
    a("### What was deliberately not built")
    a("")
    a("**No `SIMILAR_TO` edges, in either view.** The condition view does not apply "
      "(SNL uses a fixed 0.5C CC-CV charge, not a two-step fast-charge policy, so "
      "those features are null). The behaviour view was skipped by choice: SNL cells "
      "do not enter the neighbour bank and no model was retrained. This is ingestion, "
      "not adaptation. `SIMILAR_TO` stands at 4 020 edges before and after.")
    a("")

    # ---------------------------------------------------------------- 5
    a("## 5. Changed outputs")
    a("")
    a("### 5a. New claim-vs-measured discrepancies")
    a("")
    a("Both records read their measured side **out of the graph** — queries the graph "
      "could not answer before this ingestion.")
    a("")
    a("| property | claim | measured (median) | n | gap | conditions comparable |")
    a("|---|---|---|---|---|---|")
    cap, life = disc["capacity"], disc["cycle_life"]
    a(f"| `nominal_capacity_ah` | {cap['claim_value']} Ah | "
      f"{cap['measured']['median']:.4f} Ah | {cap['measured']['n']} | "
      f"{100 * cap['relative_gap']:+.1f} % | **False** |")
    a(f"| `cycle_life_cycles` | {life['claim_value']:.0f} cycles | "
      f"{life['measured']['median']:.0f} cycles | {life['measured']['n']} | "
      f"{100 * life['relative_gap']:+.0f} % | **False** |")
    a("")
    a(f"**Capacity.** All {cap['cells_below_claim']} of "
      f"{cap['measured']['n']} cells sit below the claimed 1.1 Ah, by "
      f"{cap['shortfall_pct_range'][0]:.1f} % to {cap['shortfall_pct_range'][1]:.1f} %. "
      "The `conditions_comparable=False` verdict here is **not** driven by the "
      "0.5C-versus-4C rate gap that separates SNL from Severson. It is driven by the "
      "datasheet itself. The capacity entry reads, in full:")
    a("")
    a(f"> {cap['claim_conditions']}")
    a("")
    a("There is no stated discharge current, no temperature and no cut-off, so there "
      "is no condition set to compare against, and the project's own "
      "`currents_comparable` rule cannot return true against a null. **A capacity "
      "number published with no measurement conditions attached cannot be falsified, "
      "only contextualised** — and that omission is itself the finding. The "
      "consistent one-directional shortfall across 30 cells from an independent "
      "laboratory is the context.")
    a("")
    a("**Cycle life.** The six EOL-reaching cells exceed the 1 000-cycle claim by "
      f"{100 * life['relative_gap']:+.0f} %. The verdict is again False: the claim "
      "specifies 5C discharge while these cells ran at 1C–3C; the claim states no "
      "charge regime, no temperature and no end-of-life threshold; and the measured "
      f"median is censored-biased, since only {life['measured']['n']} of 30 cells have "
      "reached EOL at all — the other "
      f"{life['n_censored']} were still above 80 % of nominal when testing stopped.")
    a("")
    a("**The sign is the result worth reporting.** The graph now holds two "
      "claim-vs-measured cycle-life discrepancies for the *same cell* against the "
      "*same datasheet number*:")
    a("")
    a("| measurement source | n | measured median | gap vs 1 000-cycle claim |")
    a("|---|---|---|---|")
    a("| `severson_mit_2019` | 124 | 736 cycles | **−26.5 %** |")
    a(f"| `snl_preger_2020` | {life['measured']['n']} | "
      f"{life['measured']['median']:.0f} cycles | **+244.6 %** |")
    a("")
    a("Same cell, same claim, opposite direction — because Severson fast-charges at "
      "3.6C–8C by design while SNL charges at 0.5C. Neither is a false claim and "
      "neither is a verdict; both are `conditions_comparable=False`. This contrast is "
      "the clearest available argument for why the system reports comparability at "
      "all rather than a bare percentage gap, and it exists only because a second "
      "measurement source entered the graph.")
    a("")

    a("### 5b. Graph before and after")
    a("")
    a("| | before | after ingestion | after discrepancies |")
    a("|---|---|---|---|")
    a(f"| nodes | {before['nodes']} | {after['nodes']} | {final['nodes']} |")
    a(f"| relationships | {before['relationships']} | {after['relationships']} | "
      f"{final['relationships']} |")
    a("")
    a("| label | before | after | Δ |")
    a("|---|---|---|---|")
    for k in sorted(set(before["labels"]) | set(final["labels"])):
        x, y = before["labels"].get(k, 0), final["labels"].get(k, 0)
        a(f"| `{k}` | {x} | {y} | {'+' + str(y - x) if y != x else '—'} |")
    a("")
    a("| relationship | before | after | Δ |")
    a("|---|---|---|---|")
    for k in sorted(set(before["relationship_types"]) | set(final["relationship_types"])):
        x, y = before["relationship_types"].get(k, 0), final["relationship_types"].get(k, 0)
        a(f"| `{k}` | {x} | {y} | {'+' + str(y - x) if y != x else '**unchanged**'} |")
    a("")
    a(f"Sources: {len(before['sources'])} → {len(final['sources'])} "
      f"(`snl_preger_2020` added). CellInstances by study: "
      f"{after['cell_instances_by_study']}. Cells with measurements stays at "
      f"{final['cells_with_measurements']} — all 30 new instances attach to the "
      "A123 node that already had them, which is the point of the entity match.")
    a("")
    a("`SIMILAR_TO` is unchanged, so the neighbour graph the published experiments "
      "read is untouched. The experiments themselves read a pinned edge snapshot "
      "(`data/kg_snapshots/severson_edges_publication.json`), so they are unaffected "
      "regardless.")
    a("")

    a("### 5c. Coverage of the SNL cells against the deployed bank")
    a("")
    a("| population | min | median | max | ≥ threshold 4.08 |")
    a("|---|---|---|---|---|")
    sc, sv = cov["snl_coverage"], cov["severson_in_study_coverage"]
    a(f"| SNL LFP ({cov['n_cells']}) | {sc['min']:.2f} | **{sc['median']:.2f}** | "
      f"{sc['max']:.2f} | **{cov['snl_cells_above_threshold']} / {cov['n_cells']}** |")
    a(f"| Severson in-study (120) | {sv['min']:.2f} | {sv['median']:.2f} | "
      f"{sv['max']:.2f} | 69 / 120 |")
    a("")
    a("**Nothing was added to the bank and nothing was retrained** "
      f"({cov['bank_unchanged']['cells']} bank cells, "
      f"{cov['bank_unchanged']['snl_added_to_bank']} SNL added, "
      f"{cov['bank_unchanged']['models_retrained']} models retrained). This is a "
      "read-only placement.")
    a("")
    a("The qualification matters more than the number. **SNL cells cannot be placed "
      "in the deployed behaviour space on the paper's own feature definition at "
      "all.** That view is dQ(V) between cycles 10 and 100 of a 4C discharge; SNL's "
      "ordinary discharges are sampled at 120 s (9–58 points, too sparse to invert) "
      "and its dense discharges are the 0.5C RPTs at cycles ~1–3 and then every "
      "200–500. The figures above therefore use an explicitly labelled **surrogate**: "
      "dQ(V) between the first RPT and the first RPT at least 150 ageing cycles "
      "later, both at 0.5C, with the >60 s gap-split applied (without it, correlation "
      "with the true curve falls to 0.32 on the 35 °C cells). Three distortions "
      "separate this from a like-for-like number — discharge rate, cycle window, and "
      "a frozen Severson z-score scaler — all pushing toward understating similarity. "
      "Read it as \"far outside the bank\", not as a calibrated distance.")
    a("")
    a("That is the expected and correct outcome: a 0.5C-charged, RPT-characterised "
      "study does not resemble a 4C-discharge fast-charge study in behaviour space, "
      "and the gate would refuse all 30 cells. Ingesting the source improves the "
      "graph's *reconciliation* capability immediately while leaving its *prediction* "
      "capability untouched — the two are separable, and this cycle exercises only "
      "the first.")
    a("")

    # ---------------------------------------------------------------- files
    a("## Artifacts")
    a("")
    a("| file | contents |")
    a("|---|---|")
    a("| `promotion_summary.json` | the human-gate snapshot, as printed |")
    a("| `matching_evidence.json` | entity-match evidence and provenance decision |")
    a("| `snl_instances_pending.csv` | the 30 instances as offered for approval |")
    a("| `snl_preprocessing_audit.csv` | per-cell RPT/cycling counts, screen results |")
    a("| `graph_stats_before_after.json` | node/edge counts + idempotency check |")
    a("| `discrepancies.json` | both Discrepancy records with full rationales |")
    a("| `discrepancy_capacity_per_cell.csv` | per-cell measured initial capacity |")
    a("| `snl_behavior_coverage.csv` | per-cell surrogate features and coverage |")
    a("| `snl_coverage_summary.json` | coverage summary and stated distortions |")
    a("")
    a("Reproduce: `python -m experiments.exp08_snl_ingestion.{snl_ingest,stage}`, then "
      "`load_snl --confirm --verify-idempotent`, `discrepancies --confirm`, "
      "`coverage`, `report`.")
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print(f"[report] -> {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
