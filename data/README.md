# Data sources

`data/raw/` is **gitignored** (large, reproducible). Severson and Sandia are
re-downloadable with `python -m src.ingestion.download all`; lygte-info.dk
pages are re-fetched by `python -m src.ingestion.lygte`; **HUST is a manual
download** (see §5). Everything is rebuilt into tidy Parquet with
`python -m src.ingestion.build_all` (HUST separately via
`python -m src.ingestion.hust`).

Processed artifacts land in `data/processed/<study>_cycles.parquet`, all
conforming to the common tidy schema in `src/ingestion/schema.py`:

| column | type | notes |
|---|---|---|
| `cell_id` | string | original study id, preserved verbatim |
| `study` | string | `severson_mit` \| `sandia` \| `hust` |
| `cycle_index` | Int64 | 1-based cycle number |
| `discharge_capacity_ah` | float | per-cycle discharge capacity [Ah] |
| `chemistry` | string | LFP / NCA / NMC / LCO |
| `nominal_capacity_ah` | Float64 | datasheet nominal [Ah] (nullable) |
| `temperature_c` | Float64 | test temperature [°C] (nullable) |
| `dod_window` | string | depth-of-discharge window (nullable) |
| `c_rate_charge` | string | charge policy / C-rate (nullable) |
| `c_rate_discharge` | string | discharge C-rate (nullable) |

One row = one (cell, cycle). Per-cell metadata is denormalised onto every row.

---

## 1. MIT / Stanford / Toyota — Severson 2019  (`study = severson_mit`)

- **Paper:** Severson et al., *Data-driven prediction of battery cycle life
  before capacity degradation*, Nature Energy 4, 383–391 (2019).
- **Host:** https://data.matr.io/1/  (project `5c48dd2bc625d700019f3204`)
- **Downloaded:** 2026-07-09
- **Files (batches 1–3, MATLAB v7.3 / HDF5):**
  - `2017-05-12_batchdata_updated_struct_errorcorrect.mat`  (~2.82 GiB)
  - `2017-06-30_batchdata_updated_struct_errorcorrect.mat`  (~1.87 GiB)
  - `2018-04-12_batchdata_updated_struct_errorcorrect.mat`  (~3.01 GiB)
- **Cells:** all 124 are **A123 APR18650M1A (LFP), 1.1 Ah nominal**, cycled at
  **30 °C**, discharged at **4C** to 2.0 V; per-cell fast-charge policy captured
  in `c_rate_charge` (from the file's `policy_readable`).
- **Cleaning reproduced from the canonical repo** (rdbraatz
  `data-driven-prediction-of-battery-cycle-life`): batch1 excludes
  `b1c8,b1c10,b1c12,b1c13,b1c22`; batch3 excludes
  `b3c37,b3c2,b3c23,b3c32,b3c42,b3c43`; five batch1 cells continued into batch2
  are merged (`b1c0↔b2c7, b1c1↔b2c8, b1c2↔b2c9, b1c3↔b2c15, b1c4↔b2c16`) and the
  batch2 duplicates dropped → **124 cells**.
- **Note:** cycle 1 is absent for every cell (excluded by the original authors).
- Batch 4 (`2019-01-24…`, Attia 2020 closed-loop optimisation) is **not** part
  of Severson 2019 and is never included in the modelling set; it is downloaded
  on request for experiment 09 (§7).

## 2. Sandia National Laboratories — Cell Cycle Testing Data  (`study = sandia`)

- **Study:** Preger et al., *Degradation of Commercial Lithium-Ion Cells as a
  Function of Chemistry and Cycling Conditions*, J. Electrochem. Soc. 167,
  120532 (2020); OSTI 1650174.
- **Host:** https://www.sandia.gov/energystoragesafety/rd-data-repository/
- **File:** `Sandia_Cell_Cycle_Testing_Data.zip` (~298 MiB), **downloaded 2026-07-09**
- **Contents:** 24 cells = 4 chemistries × 6 (LCO/LFP/NCA/NMC), short-term
  cycled across a **5/15/25/35/45 °C** temperature sweep. Arbin Excel exports
  (`.xls`/`.xlsx`); we read the per-cycle `Statistics_*` sheet.
- **Series vs. cell:** each `(cell, temperature, discharge-mode)` workbook is one
  fade series → `cell_id = "<chem>_<n>_<temp>C_<Reg|Mod>"` (physical cell =
  leading `<chem>_<n>`). Reg = non-segmented discharge, Mod = segmented (self-heating).
- **Nominal capacities** (from Preger 2020; see `cell_metadata.py`): LFP = A123
  APR18650M1A 1.1 Ah, NCA = Panasonic NCR18650B 3.2 Ah, NMC = LG 18650HG2 3.0 Ah.
  **LCO model/capacity is UNCONFIRMED** — nominal left null and flagged until
  verified against `README_Cycle_Data.docx`.
- **Finding (important):** this short-term set is **characterisation data, not
  degradation-to-EOL data.** The per-cycle `Statistics` sheet contains
  formation / rate-characterisation rows whose `Discharge_Capacity(Ah)` reaches
  physically impossible values for an 18650 (up to ~8 Ah — i.e. multi-step
  aggregated rows), producing a sawtooth rather than a monotone fade. Series are
  short (median ~12 cycles). The QC step therefore flags **all 192 series**
  (`short_series`, `capacity_gt_nominal`, `capacity_spike_down`, `no_eol_reached`),
  and the computed "cycle life" for Sandia is **not trustworthy** — use it only
  as a schema/ingestion demonstrator. For long-term Sandia degradation, use the
  Battery Archive **SNL 86-cell** set (below), not this file. The clean
  degradation curves in this project come from the Severson set above.

---

## 3. Manufacturer datasheets  (claims side of the KG)

PDFs in `claims/datasheets/` (gitignored, re-downloadable from the URLs below).
Hand-extracted claims live in `claims/*.yaml` (tracked — the gold standard for
later LLM extraction; format in `claims/README.md`). All retrieved **2026-07-11**.

| cell | file | source URL | notes |
|---|---|---|---|
| A123 APR18650M1A | `a123_apr18650m1a.pdf` | https://www.batteryspace.com/prod-specs/6612.pdf | original A123 sheet MD100009 (© 2009), 1 page. Warning: `6610.pdf` on the same host is the ANR26650M1-B, not this cell. |
| Panasonic NCR18650B | `panasonic_ncr18650b.pdf` | https://www.orbtronic.com/content/NCR18650B-Datasheet-Panasonic-Specifications.pdf | 4-page marketing sheet 2G23X0KYKU; **contains no cycle-life claim** (recorded as an omission). |
| Panasonic NCR18650B | `panasonic_ncr18650b_full_spec_sanyo.pdf` | https://www.dnkpower.com/wp-content/uploads/2022/08/NCR18650B-datasheet.pdf | official SANYO/Panasonic full spec, File No. NCR18650-068, type NCR18650B-H00BA, issued 2012-05-29 (Tentative); §6.3 holds the formal cycle-life claim. |
| LG Chem 18650HG2 | `lg_inr18650hg2.pdf` | https://www.batteryspace.com/prod-specs/9989.specs.pdf | LG Chem PRODUCT SPECIFICATION PS-HG2-Rev0 (2015-01-28), 11 pages; §4.2.3 cycle life. |

Load into the KG: `python -m src.kg.claims` then `python -m src.kg.discrepancy`.

## 4. lygte-info.dk independent tests  (third source)

HKJ's independent battery reviews. Each page fetched **once** (2026-07-10 UTC)
with an academic-research User-Agent, cached in `raw/lygte/` (gitignored,
re-fetchable via `python -m src.ingestion.lygte`); all parsing works from the
cache. No crawling beyond these three review pages.

| cell | review URL |
|---|---|
| A123 APR18650M1A | https://lygte-info.dk/review/batteries2012/A123%2018650%201100mAh%20(Yellow)%20UK.html |
| Panasonic NCR18650B | https://lygte-info.dk/review/batteries2012/Panasonic%20NCR18650B%203400mAh%20(Green)%20UK.html |
| LG Chem 18650HG2 | https://lygte-info.dk/review/batteries2012/LG%2018650%20HG2%203000mAh%20(Brown)%20UK.html |

**Parsing reality:** the measured discharge-capacity/energy data on these pages
exists ONLY as chart images — recorded as `chart_only: true` (no image
extraction, by policy). Text-explicit measurements extracted with source
fragments: A123 sustained 30 A for ~2 min; Panasonic test cutoff 2.8 V (full
capacity not measured); LG cell temperature 81 °C @ 20 A and 90 °C @ 30 A
(30 A run terminated early on temperature). The "Official specifications"
echoes are also kept (`spec_echo`) because they sometimes disagree with the
manufacturer datasheets. Processed output:
`processed/lygte_measurements.json`; KG loading:
`python -m src.ingestion.lygte && python -m src.kg.independent`.

## 5. HUST 77-cell dataset  (`study = hust`)

- **Paper:** Ma et al., *Real-time personalized health status prediction of
  lithium-ion batteries using deep transfer learning*, Energy Environ. Sci. 15
  (2022); DOI 10.1039/d2ee01676a.
- **Dataset:** Mendeley Data, DOI 10.17632/nsc7hnsg4s.2 — **CC BY 4.0**.
- **Download (manual — not covered by `src.ingestion.download`):** fetch
  `our_data.zip` (~1.2 GB) from https://data.mendeley.com/datasets/nsc7hnsg4s/2
  and save it as `data/raw/hust/hust_data.zip`. The loader streams the per-cell
  pickles directly from the zip; no extraction needed.
- **Cells:** 77 × A123 APR18650M1A (LFP, 1.1 Ah — the same commercial cell as
  Severson), fixed charge 5C(80%)-1C, per-cell varied multi-stage discharge
  (rates in `data/hust_discharge_rates.json`, vendored from BatteryML, MIT).
- **Build:** `python -m src.ingestion.hust` →
  `processed/hust_cycles.parquet` (tidy schema above, `study = hust`) and
  `processed/hust_features.parquet` (early-cycle features).
- Used as the **cross-study transfer target** (experiment 04): train on
  Severson, predict HUST.

## 6. Battery Archive — Sandia SNL LFP study  (`study = snl`, experiment 08)

- **Paper:** Preger et al., *Degradation of Commercial Lithium-Ion Cells as a
  Function of Chemistry and Cycling Conditions*, J. Electrochem. Soc. 167,
  120532 (2020); DOI 10.1149/1945-7111/abae37. Same study as §2, published
  through Battery Archive at per-cycle resolution.
- **Host:** https://www.batteryarchive.org/snl_study.html
- **Download (manual — not covered by `src.ingestion.download`):** from the page
  above take `SNL LFP.zip` (~252 MiB, 60 files = 30 cells × `cycle_data` +
  `timeseries` CSV) and save it under `BatteryArchive/` in the repo root.
  **Downloaded 2026-07-15.** `BatteryArchive/` is gitignored.
- **Cells:** 30 × A123 APR18650M1A (LFP, 1.1 Ah) — the same commercial cell as
  Severson and HUST, from an independent laboratory: 0.5C CC-CV charge, ageing
  at 0.5–3C discharge over 0–100/20–80/40–60 % DoD at 15/25/35 °C, EOL at 80 %
  of the 1.1 Ah nominal.
- **Build:** the experiment-08 pipeline, which runs the study through the
  system's own update path rather than a bare loader —
  `python -m experiments.exp08_snl_ingestion.stage`, then `.snl_ingest`,
  `.load_snl`, `.coverage`, `.discrepancies`, `.report`. Needs a live Neo4j.
- Used as a **second, independent measurement source** for a cell whose only
  prior evidence was Severson.
- The other archives under `BatteryArchive/` (NCA, NMC, CALCE, HNEI, Oxford,
  UL-Purdue, Michigan) were downloaded on the same date for feasibility scans
  (`scripts/snl_feasibility/`) and are not ingested.

## 7. Severson batch 4 — Attia 2020 closed-loop optimisation  (experiment 09)

- **Paper:** Attia et al., *Closed-loop optimization of fast-charging protocols
  for batteries with machine learning*, Nature 578, 397–402 (2020).
- **Download:** `python -m src.ingestion.download attia` → `data/raw/severson_mit/
  2019-01-24_batchdata_updated_struct_errorcorrect.mat` (~2.4 GiB). Deliberately
  excluded from the `severson` and `all` targets so both stay reproducible as
  published — batch 4 is **not** part of the Severson 124-cell modelling set.
- **Cells:** the same A123 APR18650M1A, same laboratory and equipment as
  Severson, under multi-step charge protocols the graph has never seen.
- Used as the hardest honest test of the abstention gate (experiment 09):
  abstention on genuinely novel protocols is the correct outcome, not a failure.
