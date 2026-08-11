# End-to-end update cycle: adding SNL LFP to the knowledge graph

A full pass of the staged, human-gated ingestion pipeline on a source the graph had never seen: discovery → entity matching → human validation → graph update → changed outputs. The new source is the Sandia National Laboratories LFP cycling study (Preger et al. 2020), 30 cells of the A123 APR18650M1A — the same commercial cell the graph already holds Severson and HUST instances of, under a third laboratory and a third protocol family.

Every step below was executed; nothing is illustrative. The human gate was real: the pipeline stopped, printed a promotion summary, and did not proceed until approved.

## 1. Discovery and staging

The study was registered through the Literature Monitor's staging path as candidate `doi:10.1149/1945-7111/abae37`, in the monitor's own record schema, with `status='pending_review'`.

| field | value |
|---|---|
| title | Yuliya Preger et al. 2020, Journal of The Electrochemical Society 167, 120532 |
| DOI | `10.1149/1945-7111/abae37` (resolved and verified) |
| registered via | `manual_registration` |
| triage | `yes`, attributed to a human — the LLM triage step was not invoked |
| data | `BatteryArchive/SNL LFP.zip` |

Two honest notes about this step. First, the monitor's **automated sweep could not have found this study**: it filters to publication year ≥ 2024 and this is 2020. Manual registration exists for exactly that gap, and the record says so rather than implying an API hit. Second, the record is stored in `data/monitor/candidates.jsonl`, which is canonical, and mirrored into the graph as a `:CandidateSource` node — a label deliberately distinct from a real `:Source`. At first attempt the graph was unreachable and the mirror was skipped with a warning while the JSONL still recorded the candidate; that fail-soft path is by design and it worked.

## 2. Entity matching

All 30 cells were matched to the **existing** `A123 APR18650M1A` Cell node. No new Cell node was created. The evidence is deliberately split into what the publication *asserts* and what the data merely *corroborates*.

**Asserted (the basis of the match).** The article states verbatim:

> LFP from A123 Systems (Part #APR18650M1A, 1.1 Ah), NCA from Panasonic (Part #NCR18650B, 3.2 Ah), and NMC from LG Chem (Part #18650HG2, 3 Ah)

— Yuliya Preger et al. 2020, Journal of The Electrochemical Society 167, 120532

**Corroborating signature from the data.** 18650 form factor encoded in every filename; LFP tag consistent with the flat ~3.2 V plateau; measured initial capacity 1.029–1.076 Ah against a 1.1 Ah nameplate; 3.6 V / 2.0 V window matching the datasheet's stated cut-offs.

**Provenance decision.** The CSV files carry no manufacturer or model string anywhere. The identity therefore rests on the publication, and the graph records that rather than implying the dataset self-identifies. Every SNL instance carries:

```
identity_basis   = 'asserted_from_publication'
identity_source  = '10.1149/1945-7111/abae37'
```

## 3. Human validation gate

The pipeline halted and printed a promotion summary — source, cells, measurements, provenance annotations, preprocessing decisions and review flags — then exited without writing. Ingestion runs only behind `--confirm`. Full snapshot: `promotion_summary.json`.

Three items were escalated for a decision rather than resolved silently:

| flag | resolution |
|---|---|
| **6 of 30 cells reach 80 % of nominal**, not the 5 in the earlier feasibility report — that report used a stricter bespoke rule (3 consecutive RPTs below threshold); the standard `cycle_life_table()` interpolates the first crossing. The extra cell is `35C_0-100_0.5-1C_b` (final SoH 80.8 %). Cycle-life *values* also differ by 0–280 cycles. | Approved: use the standard definition, for consistency with the Severson and HUST instances already in the graph. |
| **10 cells carry a spurious `short_series` QC flag** — the rule fires below 50 series rows and the series is RPTs, not ageing cycles. | Approved: ingest as-is, retain the raw flag, and additionally write `n_ageing_cycles` and `n_rpt_cycles` onto every instance so the record explains itself. |
| **One QC-anomalous cell** (`25C_20-80_0.5-0.5C_b`, `capacity_spike_down`). | Approved: load it with its flag — the graph records QC state rather than hiding it. |

## 4. Ingestion

Loaded through the standard `src.kg.load.load_dataframe` path (MERGE-only, uniqueness-constrained keys). Re-running produced identical node and relationship counts — asserted, not assumed, by `--verify-idempotent`.

### Preprocessing decisions

| decision | detail |
|---|---|
| capacity series | RPT cycles only (full 0-100% window at 0.5C): 4895 RPT cycles identified, 8 dropped by the over-nominal screen, 4887 rows ingested; 163395 partial-DoD/high-rate ageing cycles excluded from the capacity series |
| over-nominal screen | 8 RPT cycles dropped for capacity >110% of nominal (cycle-level symptom of a merged double discharge) |
| gap-split (>60 s) | not applicable — no Q(V) inversion is performed (see snl_ingest.py docstring) |
| cycle index | original ageing cycle number preserved |

The **RPT restriction is the load-bearing decision**. SNL interleaves full-window 0.5C Reference Performance Tests into an ageing schedule that may run at partial depth of discharge (20–80 %, 40–60 %) and up to 3C. A partial-DoD cell never discharges fully, so its cycling-cycle capacity is not comparable to nameplate — fed to the standard cycle-life routine it would "reach" 80 % of nominal at cycle 4. Capacity is therefore read from RPT cycles only, which is also the basis a datasheet capacity number would be measured on.

The >60 s gap-split the feasibility report requires is **not** applied here, and the module says so rather than claiming a step it did not take: that defect corrupts Q(V) inversion in the *timeseries* files, and this ingestion reads per-cycle tables and inverts no discharge curve. Its cycle-level symptom was screened for explicitly — 8 RPT cycles exceeding 110 % of nominal (merged double discharges) were dropped. The gap-split does become necessary in §5c, where it is applied.

### What was deliberately not built

**No `SIMILAR_TO` edges, in either view.** The condition view does not apply (SNL uses a fixed 0.5C CC-CV charge, not a two-step fast-charge policy, so those features are null). The behaviour view was skipped by choice: SNL cells do not enter the neighbour bank and no model was retrained. This is ingestion, not adaptation. `SIMILAR_TO` stands at 4 020 edges before and after.

## 5. Changed outputs

### 5a. New claim-vs-measured discrepancies

Both records read their measured side **out of the graph** — queries the graph could not answer before this ingestion.

| property | claim | measured (median) | n | gap | conditions comparable |
|---|---|---|---|---|---|
| `nominal_capacity_ah` | 1.1 Ah | 1.0485 Ah | 30 | -4.7 % | **False** |
| `cycle_life_cycles` | 1000 cycles | 3446 cycles | 6 | +245 % | **False** |

**Capacity.** All 30 of 30 cells sit below the claimed 1.1 Ah, by 2.2 % to 6.5 %. The `conditions_comparable=False` verdict here is **not** driven by the 0.5C-versus-4C rate gap that separates SNL from Severson. It is driven by the datasheet itself. The capacity entry reads, in full:

> 'Nominal capacity and voltage: 1.1Ah, 3.3 V' — no measurement conditions stated

There is no stated discharge current, no temperature and no cut-off, so there is no condition set to compare against, and the project's own `currents_comparable` rule cannot return true against a null. **A capacity number published with no measurement conditions attached cannot be falsified, only contextualised** — and that omission is itself the finding. The consistent one-directional shortfall across 30 cells from an independent laboratory is the context.

**Cycle life.** The six EOL-reaching cells exceed the 1 000-cycle claim by +245 %. The verdict is again False: the claim specifies 5C discharge while these cells ran at 1C–3C; the claim states no charge regime, no temperature and no end-of-life threshold; and the measured median is censored-biased, since only 6 of 30 cells have reached EOL at all — the other 24 were still above 80 % of nominal when testing stopped.

**The sign is the result worth reporting.** The graph now holds two claim-vs-measured cycle-life discrepancies for the *same cell* against the *same datasheet number*:

| measurement source | n | measured median | gap vs 1 000-cycle claim |
|---|---|---|---|
| `severson_mit_2019` | 124 | 736 cycles | **−26.5 %** |
| `snl_preger_2020` | 6 | 3446 cycles | **+244.6 %** |

Same cell, same claim, opposite direction — because Severson fast-charges at 3.6C–8C by design while SNL charges at 0.5C. Neither is a false claim and neither is a verdict; both are `conditions_comparable=False`. This contrast is the clearest available argument for why the system reports comparability at all rather than a bare percentage gap, and it exists only because a second measurement source entered the graph.

### 5b. Graph before and after

| | before | after ingestion | after discrepancies |
|---|---|---|---|
| nodes | 973 | 1095 | 1097 |
| relationships | 5656 | 5866 | 5870 |

| label | before | after | Δ |
|---|---|---|---|
| `CandidateSource` | 1 | 2 | +1 |
| `Cell` | 3 | 3 | — |
| `CellInstance` | 201 | 231 | +30 |
| `Chemistry` | 3 | 3 | — |
| `Claim` | 62 | 62 | — |
| `Discrepancy` | 81 | 83 | +2 |
| `Measurement` | 613 | 703 | +90 |
| `Source` | 9 | 10 | +1 |

| relationship | before | after | Δ |
|---|---|---|---|
| `ABOUT` | 756 | 848 | +92 |
| `ASSERTED_BY` | 62 | 62 | **unchanged** |
| `CONTRASTS` | 1 | 3 | +2 |
| `HAS_CHEMISTRY` | 3 | 3 | **unchanged** |
| `INSTANCE_OF` | 201 | 231 | +30 |
| `MEASURED_BY` | 613 | 703 | +90 |
| `SIMILAR_TO` | 4020 | 4020 | **unchanged** |

Sources: 9 → 10 (`snl_preger_2020` added). CellInstances by study: {'snl_lfp': 30, '(unset)': 201}. Cells with measurements stays at 1 — all 30 new instances attach to the A123 node that already had them, which is the point of the entity match.

`SIMILAR_TO` is unchanged, so the neighbour graph the published experiments read is untouched. The experiments themselves read a pinned edge snapshot (`data/kg_snapshots/severson_edges_publication.json`), so they are unaffected regardless.

### 5c. Coverage of the SNL cells against the deployed bank

| population | min | median | max | ≥ threshold 4.08 |
|---|---|---|---|---|
| SNL LFP (30) | 0.82 | **1.65** | 2.13 | **0 / 30** |
| Severson in-study (120) | 0.43 | 4.17 | 4.55 | 69 / 120 |

**Nothing was added to the bank and nothing was retrained** (120 bank cells, 0 SNL added, 0 models retrained). This is a read-only placement.

The qualification matters more than the number. **SNL cells cannot be placed in the deployed behaviour space on the paper's own feature definition at all.** That view is dQ(V) between cycles 10 and 100 of a 4C discharge; SNL's ordinary discharges are sampled at 120 s (9–58 points, too sparse to invert) and its dense discharges are the 0.5C RPTs at cycles ~1–3 and then every 200–500. The figures above therefore use an explicitly labelled **surrogate**: dQ(V) between the first RPT and the first RPT at least 150 ageing cycles later, both at 0.5C, with the >60 s gap-split applied (without it, correlation with the true curve falls to 0.32 on the 35 °C cells). Three distortions separate this from a like-for-like number — discharge rate, cycle window, and a frozen Severson z-score scaler — all pushing toward understating similarity. Read it as "far outside the bank", not as a calibrated distance.

That is the expected and correct outcome: a 0.5C-charged, RPT-characterised study does not resemble a 4C-discharge fast-charge study in behaviour space, and the gate would refuse all 30 cells. Ingesting the source improves the graph's *reconciliation* capability immediately while leaving its *prediction* capability untouched — the two are separable, and this cycle exercises only the first.

## Artifacts

| file | contents |
|---|---|
| `promotion_summary.json` | the human-gate snapshot, as printed |
| `matching_evidence.json` | entity-match evidence and provenance decision |
| `snl_instances_pending.csv` | the 30 instances as offered for approval |
| `snl_preprocessing_audit.csv` | per-cell RPT/cycling counts, screen results |
| `graph_stats_before_after.json` | node/edge counts + idempotency check |
| `discrepancies.json` | both Discrepancy records with full rationales |
| `discrepancy_capacity_per_cell.csv` | per-cell measured initial capacity |
| `snl_behavior_coverage.csv` | per-cell surrogate features and coverage |
| `snl_coverage_summary.json` | coverage summary and stated distortions |

Reproduce: `python -m experiments.exp08_snl_ingestion.{snl_ingest,stage}`, then `load_snl --confirm --verify-idempotent`, `discrepancies --confirm`, `coverage`, `report`.
