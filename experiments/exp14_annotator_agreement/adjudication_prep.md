# Adjudication prep — the claims only the second annotator recorded

62 claims that have no counterpart in the reference gold standard, located, grouped, and checked against the source text. Nothing here is adjudicated: every row is a question for the session.

Read §1 to decide whole categories, then use §4 only for the rows that need a document check.

## 1. Categories and counts

| category | A123 APR18650M1A | LG Chem 18650HG2 | Panasonic NCR18650B (marketing sheet, image-only) | Panasonic NCR18650B (SANYO full specification) | Samsung INR18650-25R (held out) | total |
|---|---|---|---|---|---|---|
| **cell dimensions** | 0 | 2 | 5 | 0 | 0 | **7** |
| **charge-termination (cut-off) currents** | 0 | 2 | 1 | 2 | 2 | **7** |
| **limit specs (max current / cut-off voltage)** | 0 | 1 | 0 | 3 | 0 | **4** |
| **other** | 2 | 4 | 0 | 8 | 2 | **16** |
| **plot-derived values** | 4 | 0 | 12 | 0 | 0 | **16** |
| **storage-duration rows** | 0 | 2 | 0 | 4 | 1 | **7** |
| **test-criterion rows** | 0 | 2 | 0 | 3 | 0 | **5** |
| _total_ | 6 | 13 | 18 | 20 | 5 | _62_ |

## 2. Does the gold standard already do this?

| category | new claims | gold precedent | where | leaning |
|---|---:|---:|---|---|
| cell dimensions | 7 | 2 | Samsung INR18650-25R (held out) (2) | **gold is inconsistent** — present in 1 of 5 gold documents; decide once, then backfill the rest |
| charge-termination (cut-off) currents | 7 | 0 | — | **scope decision** — no precedent anywhere in gold |
| limit specs (max current / cut-off voltage) | 4 | 11 | A123 APR18650M1A (2), LG Chem 18650HG2 (3), Panasonic NCR18650B (SANYO full specification) (2), Panasonic NCR18650B (marketing sheet, image-only) (1), Samsung INR18650-25R (held out) (3) | **include** — gold does this in every document; excluding them would make the corpus internally inconsistent |
| other | 16 | 77 | A123 APR18650M1A (9), LG Chem 18650HG2 (14), Panasonic NCR18650B (SANYO full specification) (11), Panasonic NCR18650B (marketing sheet, image-only) (13), Samsung INR18650-25R (held out) (30) | **row by row** — a catch-all, not a category; no single decision covers it |
| plot-derived values | 16 | 1 | A123 APR18650M1A (1) | **gold is inconsistent** — present in 1 of 5 gold documents; decide once, then backfill the rest |
| storage-duration rows | 7 | 7 | LG Chem 18650HG2 (3), Samsung INR18650-25R (held out) (4) | **gold is inconsistent** — present in 2 of 5 gold documents; decide once, then backfill the rest |
| test-criterion rows | 5 | 5 | LG Chem 18650HG2 (2), Panasonic NCR18650B (SANYO full specification) (1), Samsung INR18650-25R (held out) (2) | **gold is inconsistent** — present in 3 of 5 gold documents; decide once, then backfill the rest |

### Per-category detail

**cell dimensions** — 7 new, 2 in gold (1/5 documents)

- precedent: `cell_height_mm` = 64.85 mm (Samsung INR18650-25R (held out))
- precedent: `cell_diameter_mm` = 18.33 mm (Samsung INR18650-25R (held out))

- guideline: §2 final bullet — a dimension is excluded when it appears only as a tolerance on the mechanical drawing, and is a claim when the document states it as a specification value. The distinction is per-document.

**charge-termination (cut-off) currents** — 7 new, 0 in gold (0/5 documents)
- **the information is already in gold, but as a condition**: 4 gold claims carry an `end_current_ma`/cut-off field inside `stated_conditions` (LG Chem 18650HG2). So the decision is not whether to record it but at what level — claim or condition.

- guideline: §7.4 lists `end_current_ma` as a *condition* field, while §8.1 says a row stating several quantities yields several claims. The guideline genuinely does not settle this one — it is the clearest scope decision for the session.

**limit specs (max current / cut-off voltage)** — 4 new, 11 in gold (5/5 documents)

- precedent: `max_cont_discharge_a` = 30 A (A123 APR18650M1A)
- precedent: `discharge_cutoff_v` = 2 V (A123 APR18650M1A)
- precedent: `max_charge_voltage_v` = 4.2 V (LG Chem 18650HG2)

- guideline: §4.1 gives `max_charge_current_a`, `max_cont_discharge_a` and `discharge_cutoff_v` as vocabulary entries, so these are claims by construction; the question is only whether the specific row was missed.

**other** — 16 new, 77 in gold (5/5 documents)

- precedent: `nominal_capacity_ah` = 1.1 Ah (A123 APR18650M1A)
- precedent: `nominal_voltage_v` = 3.3 V (A123 APR18650M1A)
- precedent: `std_charge_current_a` = 1.5 A (A123 APR18650M1A)

- guideline: §2 — general scope of what counts as a quantitative claim.

**plot-derived values** — 16 new, 1 in gold (1/5 documents)

- precedent: `cycle_life_retention_pct` = 93 pct (A123 APR18650M1A)
- the image-only Panasonic marketing sheet has no text layer, so **every** value in it — in gold as much as in the second set — was read by eye; gold simply did not label them `read from graph`. Treat the 12 rows there as a labelling question first, a scope question second.

- guideline: §7.8 — plotted values are claims, read conservatively and flagged `read from graph`; §2 lists them under what to annotate.

**storage-duration rows** — 7 new, 7 in gold (2/5 documents)

- precedent: `storage_temp_range_c` = [-20, 20] degC (LG Chem 18650HG2)
- precedent: `storage_capacity_remaining_pct` = 90 pct (LG Chem 18650HG2)
- precedent: `storage_capacity_recovery_pct` = 80 pct (LG Chem 18650HG2)

- guideline: §8.5 — one property may appear several times with different conditions (the duration belongs in `stated_conditions`, not in the property name).

**test-criterion rows** — 5 new, 5 in gold (3/5 documents)

- precedent: `cycle_life_cycles` = 300 cycles (LG Chem 18650HG2)
- precedent: `cycle_life_cycles` = 200 cycles (LG Chem 18650HG2)
- precedent: `cycle_life_cycles` = 300 cycles (Panasonic NCR18650B (SANYO full specification))

- guideline: §7.9 — numerically stated pass criteria are claims, flagged in `notes`. §2 excludes mechanical/safety/abuse criteria, so the line runs between electrical performance tests and safety tests.

## 3. The claims, by category

| # | category | document | page | section | property | value | verification | condition text |
|---:|---|---|---:|---|---|---|---|---|
| 1 | cell dimensions | lg | 5 | §3.2 Dimension — Diameter | `diameter_mm` | 18.3 mm | found on cited page | 3.2 Dimension — Diameter: 18.3 + 0.2/-0.3 mm ( Max. 18.5 mm ). 'Diameter is defi |
| 2 | cell dimensions | lg | 5 | §3.2 Dimension — Height | `height_mm` | 65 mm | found on cited page | 3.2 Dimension — Height: 65.0 ± 0.2 mm ( Max. 65.2 mm ) |
| 3 | cell dimensions | panasonic_mkt | 1 | — | `diameter_mm` | 18.25 mm | no text layer ⚑ | Specifications: Dimensions (Max.) — 'Maximum size without tube' | (D) | 18.25mm |
| 4 | cell dimensions | panasonic_mkt | 1 | — | `diameter_mm` | 18.2 mm | no text layer ⚑ | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assembling |
| 5 | cell dimensions | panasonic_mkt | 1 | — | `height_mm` | 65.1 mm | no text layer ⚑ | Specifications: Dimensions (Max.) — 'Maximum size without tube' | (H) | 65.10mm |
| 6 | cell dimensions | panasonic_mkt | 1 | — | `height_mm` | 64.93 mm | no text layer ⚑ | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assembling |
| 7 | cell dimensions | panasonic_mkt | 1 | — | `terminal_diameter_mm` | 7.9 mm | no text layer ⚑ | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assembling |
| 8 | charge-termination (cut-off) currents | lg | 4 | §2.3.2 Fast charge (Refer to 4.1.3) | `fast_charge_end_current_a` | 0.1 A | found on cited page | 2.3.2 Fast charge (Refer to 4.1.3) | End condition(Cut off) | 100mA |
| 9 | charge-termination (cut-off) currents | lg | 4 | §2.3.1 Standard Charge (Refer to 4.1.1) | `std_charge_end_current_a` | 0.05 A | found on cited page | 2.3.1 Standard Charge (Refer to 4.1.1) | End condition(Cut off) | 50mA |
| 10 | charge-termination (cut-off) currents | panasonic_full | 11 | §10 | `precharge_current_a` | 0.32 A | found on cited page | 10-1(1) Charge: 'the battery should be charged by pre-charge current of maximum  |
| 11 | charge-termination (cut-off) currents | panasonic_full | 11 | §10 | `precharge_threshold_voltage_v` | 3 V | found on cited page | 10-1(1) Charge: 'If battery voltage goes down to less than 3.0V/cell, the batter |
| 12 | charge-termination (cut-off) currents | panasonic_mkt | 2 | — | `std_charge_end_current_a` | 0.065 A | no text layer ⚑ | Plot caption 'Discharge Rate Characteristics for NCR18650B': Charge:CC-CV:1.625A |
| 13 | charge-termination (cut-off) currents | samsung | 3 | §3.4 Rapid charge | `fast_charge_end_current_a` | 0.1 A | found on cited page | 3.4 Rapid charge | CCCV, 4A, 4.20 ± 0.05 V, 100mA cut-off |
| 14 | charge-termination (cut-off) currents | samsung | 3 | §3.3 Standard charge | `std_charge_end_current_a` | 0.125 A | found on cited page | 3.3 Standard charge | CCCV, 1.25A, 4.20 ± 0.05 V, 125mA cut-off |
| 15 | limit specs (max current / cut-off voltage) | lg | 4 | §2.6.1 Standard Discharge (Refer to 4.1.2) | `discharge_cutoff_v` | 2 V | found on cited page | 2.6.1 Standard Discharge (Refer to 4.1.2) | End voltage(Cut off) | 2.0V |
| 16 | limit specs (max current / cut-off voltage) | panasonic_full | 11 | §10 | `max_charge_current_a` | 1.625 A | found on cited page | 10-1(1) Charge: 'Regarding NCR18650B, the charging current should not exceed 1.6 |
| 17 | limit specs (max current / cut-off voltage) | panasonic_full | 11 | §10 | `max_charge_voltage_v` | 4.2 V | found on cited page | 10-1(1) Charge: 'The charging voltage should not exceed 4.20V /cell.' |
| 18 | limit specs (max current / cut-off voltage) | panasonic_full | 13 | §12 | `series_discharge_cutoff_v` | 2.75 V | found on cited page | 12-1 Series Connections Precautions: 'If cells are connected in series, the disc |
| 19 | other | a123 | 1 | — | `fast_charge_time_min` | 15 min | found on cited page | Recommended fast charge current: 4A to 3.6V CCCV, 15 min |
| 20 | other | a123 | 1 | — | `std_charge_time_min` | 45 min | found on cited page | Recommended standard charge method: 1.5A to 3.6V CCCV, 45 min |
| 21 | other | lg | 4 | §2.3.1 Standard Charge (Refer to 4.1.1) | `charge_voltage_v` | 4.2 V | found on cited page | 2.3.1 Standard Charge (Refer to 4.1.1) | Constant voltage | 4.2V |
| 22 | other | lg | 4 | §2.6.2 Fast Discharge (Refer to 4.1.3) | `fast_discharge_current_a` | 20 A | found on cited page | 2.6.2 Fast Discharge (Refer to 4.1.3) | Constant current | 10000mA , 20000mA | E |
| 23 | other | lg | 4 | — | `shipping_state_soc_pct` | 40 pct | found on cited page | footnote to section 2: '* Shipping state : About 40% capacity of fully charged s |
| 24 | other | lg | 6 | §4.2.3 Cycle Life | `cycle_life_retention_pct` | 60 pct | found on cited page | 4.2.3 Cycle Life | end-of-life threshold for both rates: ≥ 60 % (of Cnom in 2.1) |
| 25 | other | panasonic_full | 8 | §5.13 Storing Conditions | `storage_capacity_recovery_pct` | 80 pct | found on cited page | 5.13 Storing Conditions, Notes column (spanning all three duration rows): 'Perce |
| 26 | other | panasonic_full | 9 | §6.2 Capacity ① 'Within 1 hour | `discharge_time_min` | 300 min | found on cited page | 6.2 Capacity ① 'Within 1 hour, after fully charged at 25℃, the battery is discha |
| 27 | other | panasonic_full | 9 | §6.3 Cycle Life | `discharge_time_min` | 38 min | found on cited page | 6.3 Cycle Life, criteria: 'More than 38min.' — the discharge time measured per I |
| 28 | other | panasonic_full | 9 | §6.4 Temperature Characteristics ① 'Within 1 hour | `discharge_time_min` | 30 min | found on cited page | 6.4 Temperature Characteristics ① 'Within 1 hour, after fully charged at 25℃, th |
| 29 | other | panasonic_full | 9 | §6.4 Temperature Characteristics ② 'Within 1 hour | `discharge_time_min` | 50 min | found on cited page | 6.4 Temperature Characteristics ② 'Within 1 hour, after fully charged at 25℃, th |
| 30 | other | panasonic_full | 9 | §6.5 Storage at Fully Charged State 'After fully c | `discharge_time_min` | 30 min | found on cited page | 6.5 Storage at Fully Charged State 'After fully charged at 25℃, the battery is s |
| 31 | other | panasonic_full | 11 | §10 | `over_discharge_limit_v` | 2 V | found on cited page | 10-1(3) Over discharge: 'Do not discharge the battery less than 2.0V/cell.' |
| 32 | other | panasonic_full | 11 | §9 | `shipping_state_soc_pct` | 40 pct | found on cited page | 9. Shipping Charge: 'The battery is shipped out with the approximately 40%＊ char |
| 33 | other | samsung | 3 | §3.11 Operating temperature (surface temperature) | `recommended_recharge_release_temp_c` | 45 degC | found on cited page | 3.11 Operating temperature (surface temperature) | Charge : 0 to 50℃ (recommende |
| 34 | other | samsung | 3 | §3.11 Operating temperature (surface temperature) | `recommended_redischarge_release_temp_c` | 60 degC | found on cited page | 3.11 Operating temperature (surface temperature) | Discharge: -20 to 75℃ (recomm |
| 35 | plot-derived values | a123 | 1 | — | `capacity_retention_at_cycle_pct` | 95 pct | found on cited page | Plot 'Projected Cycle Life, 100% DOD, 1C/1C, Room Temperature': the curve is at  |
| 36 | plot-derived values | a123 | 1 | — | `discharge_capacity_ah` | 1.03 Ah | plot read ⚑ | Plot 'Discharge Characteristics, Room Temperature': the 5A curve reaches the 2.0 |
| 37 | plot-derived values | a123 | 1 | — | `discharge_capacity_ah` | 1.03 Ah | plot read ⚑ | Plot 'Discharge Characteristics, Room Temperature': the 10A curve reaches the 2. |
| 38 | plot-derived values | a123 | 1 | — | `discharge_capacity_ah` | 1.01 Ah | plot read ⚑ | Plot 'Discharge Characteristics, Room Temperature': the 20A curve reaches the 2. |
| 39 | plot-derived values | panasonic_mkt | 2 | — | `discharge_capacity_ah` | 3.31 Ah | plot read ⚑ | Plot 'Discharge Rate Characteristics for NCR18650B', Temp:25°C, Charge:CC-CV:1.6 |
| 40 | plot-derived values | panasonic_mkt | 2 | — | `discharge_capacity_ah` | 3.24 Ah | plot read ⚑ | Plot 'Discharge Rate Characteristics for NCR18650B': the 0.5CA curve reaches 2.5 |
| 41 | plot-derived values | panasonic_mkt | 2 | — | `discharge_capacity_ah` | 3.28 Ah | plot read ⚑ | Plot 'Discharge Rate Characteristics for NCR18650B': the 1.0CA curve reaches 2.5 |
| 42 | plot-derived values | panasonic_mkt | 2 | — | `discharge_capacity_ah` | 3.29 Ah | plot read ⚑ | Plot 'Discharge Rate Characteristics for NCR18650B': the 2.0CA curve reaches 2.5 |
| 43 | plot-derived values | panasonic_mkt | 3 | — | `discharge_capacity_ah` | 3.34 Ah | plot read ⚑ | Plot 'Discharge Temperature Characteristics for NCR18650B', Charge:CC-CV:1.625A- |
| 44 | plot-derived values | panasonic_mkt | 3 | — | `discharge_capacity_ah` | 3.28 Ah | plot read ⚑ | Plot 'Discharge Temperature Characteristics for NCR18650B': the 25°C curve reach |
| 45 | plot-derived values | panasonic_mkt | 3 | — | `discharge_capacity_ah` | 3.01 Ah | plot read ⚑ | Plot 'Discharge Temperature Characteristics for NCR18650B': the 0°C curve reache |
| 46 | plot-derived values | panasonic_mkt | 3 | — | `discharge_capacity_ah` | 2.89 Ah | plot read ⚑ | Plot 'Discharge Temperature Characteristics for NCR18650B': the -10°C curve reac |
| 47 | plot-derived values | panasonic_mkt | 3 | — | `discharge_capacity_ah` | 2.6 Ah | plot read ⚑ | Plot 'Discharge Temperature Characteristics for NCR18650B': the -20°C curve reac |
| 48 | plot-derived values | panasonic_mkt | 4 | — | `charge_capacity_ah` | 3.27 Ah | plot read ⚑ | Plot 'Charge Characteristics for NCR18650B', Charge:CC-CV:1.625A-4.20V(65.0mA cu |
| 49 | plot-derived values | panasonic_mkt | 4 | — | `charge_capacity_ah` | 3.38 Ah | plot read ⚑ | Plot 'Charge Characteristics for NCR18650B': the 25°C capacity trace plateaus at |
| 50 | plot-derived values | panasonic_mkt | 4 | — | `charge_capacity_ah` | 3.06 Ah | plot read ⚑ | Plot 'Charge Characteristics for NCR18650B': the 0°C capacity trace plateaus at  |
| 51 | storage-duration rows | lg | 4 | §2.10 Storage Temperature (for shipping state) | `storage_temp_range_c` | [-20, 60] degC | found on cited page | 2.10 Storage Temperature (for shipping state) | 1 month | -20 ~ 60℃ |
| 52 | storage-duration rows | lg | 4 | §2.10 Storage Temperature (for shipping state) | `storage_temp_range_c` | [-20, 45] degC | found on cited page | 2.10 Storage Temperature (for shipping state) | 3 month | -20 ~ 45℃ |
| 53 | storage-duration rows | panasonic_full | 8 | §5.13 Storing Conditions | `storage_temp_range_c` | [-20, 50] degC | found on cited page | 5.13 Storing Conditions | less than 1 month | -20 ~ +50℃ |
| 54 | storage-duration rows | panasonic_full | 8 | §5.13 Storing Conditions | `storage_temp_range_c` | [-20, 40] degC | found on cited page | 5.13 Storing Conditions | less than 3 months | -20 ~ +40℃ |
| 55 | storage-duration rows | panasonic_full | 8 | §5.13 Storing Conditions | `storage_temp_range_c` | [-20, 20] degC | found on cited page | 5.13 Storing Conditions | less than 1 year | -20 ~ + 20℃ |
| 56 | storage-duration rows | panasonic_full | 13 | §11 | `storage_humidity_max_pct` | 70 pct | found on cited page | 11-1 Storage Temperature and Humidity (Within 3 months): 'Cells should be stored |
| 57 | storage-duration rows | samsung | 5 | §7.11 Storage characteristics | `storage_capacity_remaining_pct` | 90 pct | found on cited page | 7.11 Storage characteristics: 'Standard rated discharge capacity after storage f |
| 58 | test-criterion rows | lg | 6 | §4.2.2 Initial Capacity | `initial_capacity_ah` | 3 Ah | found on cited page | 4.2.2 Initial Capacity | Cell shall be charged per 4.1.1 and discharged per 4.1. |
| 59 | test-criterion rows | lg | 7 | §4.3.3 Thermal Shock Test | `thermal_shock_capacity_recovery_pct` | 80 pct | found on cited page | 4.3.3 Thermal Shock Test | 72ºC (8h) ← 3hrs → -20ºC (8h) for 8 cycles with cells |
| 60 | test-criterion rows | panasonic_full | 9 | §6.2 Capacity ② 'Within 1 hour | `discharge_time_min` | 54 min | found on cited page | 6.2 Capacity ② 'Within 1 hour, after fully charged at 25℃, the battery is discha |
| 61 | test-criterion rows | panasonic_full | 9 | §6.5 Storage at Fully Charged State | `discharge_time_min` | 40 min | found on cited page | 6.5 Storage at Fully Charged State, second criterion: 'Then, the same battery is |
| 62 | test-criterion rows | panasonic_full | 10 | §6.6 Storage at Full Discharged State 'After fully | `discharge_time_min` | 50 min | found on cited page | 6.6 Storage at Full Discharged State 'After fully charged at 25℃, the battery is |

## 4. Verify these 21 against the document

Of the 62 claims, 41 were found as printed on the page the annotator cited. In every document that has a text layer, **no value was missing and none sat on a different page than the one cited** — so there are no location or transcription anomalies. The rows below need eyes for a different reason:

- **6** are in the image-only marketing sheet, which has no text to check against;
- **15** are values read off a curve, which by definition are not printed anywhere;
- **0** are genuine anomalies (value absent from the text, or printed on a different page than cited).

| # | document | page | property | value | why |
|---:|---|---:|---|---|---|
| 1 | panasonic_mkt | 1 | `diameter_mm` | 18.25 mm | image-only document — verify on the PDF by eye |
| 2 | panasonic_mkt | 1 | `diameter_mm` | 18.2 mm | image-only document — verify on the PDF by eye |
| 3 | panasonic_mkt | 1 | `height_mm` | 65.1 mm | image-only document — verify on the PDF by eye |
| 4 | panasonic_mkt | 1 | `height_mm` | 64.93 mm | image-only document — verify on the PDF by eye |
| 5 | panasonic_mkt | 1 | `terminal_diameter_mm` | 7.9 mm | image-only document — verify on the PDF by eye |
| 6 | panasonic_mkt | 2 | `std_charge_end_current_a` | 0.065 A | image-only document — verify on the PDF by eye |
| 7 | a123 | 1 | `discharge_capacity_ah` | 1.03 Ah | read off a curve — check the reading against the plot |
| 8 | a123 | 1 | `discharge_capacity_ah` | 1.03 Ah | read off a curve — check the reading against the plot |
| 9 | a123 | 1 | `discharge_capacity_ah` | 1.01 Ah | read off a curve — check the reading against the plot |
| 10 | panasonic_mkt | 2 | `discharge_capacity_ah` | 3.31 Ah | read off a curve — check the reading against the plot |
| 11 | panasonic_mkt | 2 | `discharge_capacity_ah` | 3.24 Ah | read off a curve — check the reading against the plot |
| 12 | panasonic_mkt | 2 | `discharge_capacity_ah` | 3.28 Ah | read off a curve — check the reading against the plot |
| 13 | panasonic_mkt | 2 | `discharge_capacity_ah` | 3.29 Ah | read off a curve — check the reading against the plot |
| 14 | panasonic_mkt | 3 | `discharge_capacity_ah` | 3.34 Ah | read off a curve — check the reading against the plot |
| 15 | panasonic_mkt | 3 | `discharge_capacity_ah` | 3.28 Ah | read off a curve — check the reading against the plot |
| 16 | panasonic_mkt | 3 | `discharge_capacity_ah` | 3.01 Ah | read off a curve — check the reading against the plot |
| 17 | panasonic_mkt | 3 | `discharge_capacity_ah` | 2.89 Ah | read off a curve — check the reading against the plot |
| 18 | panasonic_mkt | 3 | `discharge_capacity_ah` | 2.6 Ah | read off a curve — check the reading against the plot |
| 19 | panasonic_mkt | 4 | `charge_capacity_ah` | 3.27 Ah | read off a curve — check the reading against the plot |
| 20 | panasonic_mkt | 4 | `charge_capacity_ah` | 3.38 Ah | read off a curve — check the reading against the plot |
| 21 | panasonic_mkt | 4 | `charge_capacity_ah` | 3.06 Ah | read off a curve — check the reading against the plot |

Verification compares the recorded value against the document's text snapshot, trying milli-unit forms too (0.05 A also as `50mA`). It confirms the number is printed where the annotator says it is; it cannot confirm the number means what the annotator took it to mean.
