> **Annotators:** A1 = first author (reference gold standard),
> A2 = second author. Both annotation sets were produced independently by
> manual annotation against `ANNOTATION_GUIDELINE.md`; A2 was blind to the
> reference annotations. Metrics reported on the independent sets
> (pre-adjudication).

# Annotation agreement — reference gold vs. second annotation set

## 1. Overall

Reference claims: **103** · second-set claims: **164** across 5 documents.

| view | matched | agreement (F1) | Jaccard |
|---|---:|---:|---:|
| exact property + value | 76 | **0.569** | 0.398 |
| value-level (unit-class guarded) | 101 | **0.757** | 0.608 |

Agreement is positive specific agreement (F1 over the matched set), symmetric in the two annotators — neither set is truth. Value tests use `src.agents.evaluation.values_match` (2 % relative tolerance, unit-scaled), matching greedy 1:1 in document order.

F1 is depressed by set-size asymmetry rather than by contradiction. The one-sided coverages separate the two effects:

| direction | covered | of | share |
|---|---:|---:|---:|
| reference claims with a value-level counterpart in the second set | 101 | 103 | **0.981** |
| second-set claims with a value-level counterpart in the reference | 101 | 164 | **0.616** |

## 2. Per document

| document | n(ref) | n(2nd) | exact F1 | value-level F1 | ref covered | naming-only | value disagr. | ref-only | 2nd-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A123 APR18650M1A | 12 | 18 | 0.733 | 0.800 | 1.000 | 1 | 0 | 0 | 6 |
| LG Chem 18650HG2 | 22 | 35 | 0.737 | 0.737 | 0.955 | 0 | 1 | 0 | 13 |
| Panasonic NCR18650B (marketing sheet, image-only) | 14 | 31 | 0.533 | 0.578 | 0.929 | 1 | 0 | 1 | 18 |
| Panasonic NCR18650B (SANYO full specification) | 14 | 34 | 0.542 | 0.583 | 1.000 | 1 | 0 | 0 | 20 |
| Samsung INR18650-25R (held out) | 41 | 46 | 0.437 | 0.943 | 1.000 | 22 | 0 | 0 | 5 |

## 3. Where the disagreements sit

Reference-side claims by stratum, and how many the value-level view matched:

| document | stratum | reference claims | matched | unmatched |
|---|---|---:|---:|---:|
| A123 APR18650M1A | read from graph | 1 | 1 | 0 |
| A123 APR18650M1A | specification row | 11 | 11 | 0 |
| LG Chem 18650HG2 | characteristic grid | 4 | 4 | 0 |
| LG Chem 18650HG2 | specification row | 18 | 17 | 1 |
| Panasonic NCR18650B (marketing sheet, image-only) | read from graph | 1 | 0 | 1 |
| Panasonic NCR18650B (marketing sheet, image-only) | specification row | 13 | 13 | 0 |
| Panasonic NCR18650B (SANYO full specification) | specification row | 14 | 14 | 0 |
| Samsung INR18650-25R (held out) | characteristic grid | 17 | 17 | 0 |
| Samsung INR18650-25R (held out) | specification row | 24 | 24 | 0 |

## 4. Conditions on matched pairs

| document | pairs | identical | both `unspecified` | ref richer | 2nd richer | only ref states | only 2nd states | conflicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A123 APR18650M1A | 12 | 10 | 0 | 0 | 0 | 0 | 1 | 2 |
| LG Chem 18650HG2 | 21 | 7 | 0 | 0 | 12 | 0 | 0 | 2 |
| Panasonic NCR18650B (marketing sheet, image-only) | 13 | 13 | 0 | 0 | 0 | 0 | 3 | 0 |
| Panasonic NCR18650B (SANYO full specification) | 14 | 8 | 0 | 1 | 4 | 0 | 0 | 1 |
| Samsung INR18650-25R (held out) | 41 | 9 | 0 | 0 | 20 | 0 | 1 | 2 |

## 5. Omissions

| document | ref | 2nd | agreed | ref only | 2nd only |
|---|---:|---:|---|---|---|
| A123 APR18650M1A | 0 | 7 | — | — | charge_temp_range_c, cycle_life_retention_pct, discharge_temp_range_c, gravimetric_energy_density_wh_kg, internal_impedance_mohm, minimum_capacity_ah, std_discharge_current_a |
| LG Chem 18650HG2 | 0 | 5 | — | — | capacity_retention_at_temp_pct, gravimetric_energy_density_wh_kg, minimum_capacity_ah, std_charge_time_h, volumetric_energy_density_wh_l |
| Panasonic NCR18650B (marketing sheet, image-only) | 1 | 7 | cycle_life_cycles | — | cycle_life_retention_pct, discharge_cutoff_v, internal_impedance_mohm, max_charge_current_a, max_cont_discharge_a, storage_capacity_remaining_pct |
| Panasonic NCR18650B (SANYO full specification) | 0 | 7 | — | — | charge_temp_range_c, cycle_life_retention_pct, diameter_mm, gravimetric_energy_density_wh_kg, max_charge_voltage_v, minimum_capacity_ah, volumetric_energy_density_wh_l |
| Samsung INR18650-25R (held out) | 5 | 6 | — | dc_internal_resistance_mohm, max_pulse_discharge_a, nominal_energy_wh, self_discharge_pct_per_month, specific_energy_wh_kg | cycle_life_cycles, gravimetric_energy_density_wh_kg, max_charge_current_a, max_charge_voltage_v, storage_capacity_remaining_pct, volumetric_energy_density_wh_l |

## 5b. Chance-corrected agreement (Cohen's κ)

κ needs a common set of items and a fixed category set. The extraction step has neither — each annotator decides for themselves what the items are, and there is no negative class ('all the numbers we both agreed were not claims'), so no chance-agreement term is definable. κ is therefore reported only for the two decisions taken **on an already matched pair**, where both annotators did label the same item:

| decision | items | observed agr. | expected by chance | κ |
|---|---:|---:|---:|---:|
| which property name | 101 | 0.752 | 0.034 | **0.744** |
| property name, grids excluded | 80 | 0.9 | 0.045 | **0.895** |
| conditions stated vs `unspecified` | 101 | 0.95 | 0.95 | **0.0** |

Per document (property-name κ over that document's matched pairs):

| document | pairs | observed | κ (property) | κ (conditions stated) |
|---|---:|---:|---:|---:|
| A123 APR18650M1A | 12 | 0.917 | 0.91 | 0.0 |
| LG Chem 18650HG2 | 21 | 1.0 | 1.0 | n/a |
| Panasonic NCR18650B (marketing sheet, image-only) | 13 | 0.923 | 0.917 | 0.0 |
| Panasonic NCR18650B (SANYO full specification) | 14 | 0.929 | 0.923 | n/a |
| Samsung INR18650-25R (held out) | 41 | 0.463 | 0.455 | 0.0 |

The conditions row is the κ prevalence paradox, not a real disagreement: the two annotators agree on 95% of pairs, but almost every pair is 'conditions stated' on both sides, so chance expectation (0.95) equals observed agreement and κ collapses. Quote the raw 95% for that decision.

κ = n/a means the decision was constant for at least one annotator (no variance to chance-correct: e.g. every matched pair on a document states conditions). For the extraction step itself, quote the F1/coverage figures above, or commission Krippendorff's α for unitizing, which is the measure built for annotators who choose their own units.


## 5c. Krippendorff's α

Krippendorff's α for **unitizing** (α_U) is defined over a continuum and needs each unit's offsets in it. These annotations carry no offsets — a claim is located by page, property, value and a compressed quotation — and one document is image-only, so it has no text continuum at all. α_U is therefore not computable from these files and is not reported. What follows is α for **nominal data over the aligned item set**: items are the union of both annotations aligned 1:1 by the value-level matcher, each carrying two codings, with `∅` (not annotated) for the annotator who did not record it.

| α | items | observed | D_o | D_e | α |
|---|---:|---:|---:|---:|---:|
| property name, incl. `∅` — *what to extract and what to call it* | 166 | 0.458 | 0.5422 | 0.9457 | **0.427** |
| naming only, matched items — *given both found it* | 101 | 0.752 | 0.2475 | 0.9666 | **0.744** |
| presence only — **degenerate, do not quote** | 166 | 0.608 | 0.3916 | 0.3159 | **-0.240** |

Per document:

| document | items | α (property) | α (naming) |
|---|---:|---:|---:|
| A123 APR18650M1A | 18 | 0.592 | 0.913 |
| LG Chem 18650HG2 | 36 | 0.555 | 1.000 |
| Panasonic NCR18650B (marketing sheet, image-only) | 32 | 0.296 | 0.920 |
| Panasonic NCR18650B (SANYO full specification) | 34 | 0.311 | 0.926 |
| Samsung INR18650-25R (held out) | 46 | 0.398 | 0.447 |

**Two caveats that must travel with these numbers.**

1. The item universe is generated by the annotators themselves, so `∅` can only appear where at least one of them proposed a claim; there is no inventory of numbers in the document that both correctly ignored. These are reliability figures *conditional on the union of proposals*, and are conservative relative to a task with a fixed item inventory.
2. That is also why the presence-only α is negative and meaningless: with the universe built from the union, the (`∅`, `∅`) cell cannot occur, so expected disagreement is computed from a distribution the data cannot realise. It is listed for completeness and must not be quoted. The comparable honest figure for that question is the coverage pair in §1.

α (naming) and Cohen's κ (§5b) agree to three decimals, which is the expected consistency check: for two coders on complete nominal data the two chance models coincide when the marginals are close.


## 6. Adjudication list

Claim by claim, nothing resolved. Columns are reference / second set.

### A123 APR18650M1A

**Naming-only — same value, different property name (1)**

| # | reference property | 2nd property | value | unit | p(ref) | p(2nd) |
|---:|---|---|---|---|---:|---:|
| 1 | `cycle_life_retention_pct` | `capacity_retention_at_cycle_pct` | 93 | pct | 1 | 1 |

**In the second set only (6)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `std_charge_time_min` | 45 | min | 1 | specification row | Recommended standard charge method: 1.5A to 3.6V CCCV, 45 min |
| 2 | `fast_charge_time_min` | 15 | min | 1 | specification row | Recommended fast charge current: 4A to 3.6V CCCV, 15 min |
| 3 | `discharge_capacity_ah` | 1.03 | Ah | 1 | read from graph | Plot 'Discharge Characteristics, Room Temperature': the 5A curve reaches the 2.… |
| 4 | `discharge_capacity_ah` | 1.03 | Ah | 1 | read from graph | Plot 'Discharge Characteristics, Room Temperature': the 10A curve reaches the 2… |
| 5 | `discharge_capacity_ah` | 1.01 | Ah | 1 | read from graph | Plot 'Discharge Characteristics, Room Temperature': the 20A curve reaches the 2… |
| 6 | `capacity_retention_at_cycle_pct` | 95 | pct | 1 | read from graph | Plot 'Projected Cycle Life, 100% DOD, 1C/1C, Room Temperature': the curve is at… |

**Conflicting condition fields on matched pairs (2)**

| # | property | field | reference | 2nd |
|---:|---|---|---|---|
| 1 | `cycle_life_cycles` | `discharge_c_rate` | 5.0 | 5 |
| 2 | `cycle_life_retention_pct` | `charge_c_rate` | 1.0 | 1 |
| 3 | `cycle_life_retention_pct` | `discharge_c_rate` | 1.0 | 1 |

### LG Chem 18650HG2

**Value disagreements — same property, different value (1)**

| # | property | reference | 2nd | p(ref) | p(2nd) | reference conditions | 2nd conditions |
|---:|---|---|---|---:|---:|---|---|
| 1 | `fast_discharge_current_a` | [10, 20] A | 10 A | 4 | 4 | 2.6.2 Fast Discharge: CC 10000mA, 20000mA; end voltage 2.0V | 2.6.2 Fast Discharge (Refer to 4.1.3) | Constant current | 10000mA , … |

**In the second set only (13)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `charge_voltage_v` | 4.2 | V | 4 | specification row | 2.3.1 Standard Charge (Refer to 4.1.1) | Constant voltage | 4.2V |
| 2 | `std_charge_end_current_a` | 0.05 | A | 4 | specification row | 2.3.1 Standard Charge (Refer to 4.1.1) | End condition(Cut off) | 50mA |
| 3 | `fast_charge_end_current_a` | 0.1 | A | 4 | specification row | 2.3.2 Fast charge (Refer to 4.1.3) | End condition(Cut off) | 100mA |
| 4 | `discharge_cutoff_v` | 2 | V | 4 | specification row | 2.6.1 Standard Discharge (Refer to 4.1.2) | End voltage(Cut off) | 2.0V |
| 5 | `fast_discharge_current_a` | 20 | A | 4 | specification row | 2.6.2 Fast Discharge (Refer to 4.1.3) | Constant current | 10000mA , 20000mA | … |
| 6 | `storage_temp_range_c` | [-20, 60] | degC | 4 | specification row | 2.10 Storage Temperature (for shipping state) | 1 month | -20 ~ 60℃ |
| 7 | `storage_temp_range_c` | [-20, 45] | degC | 4 | specification row | 2.10 Storage Temperature (for shipping state) | 3 month | -20 ~ 45℃ |
| 8 | `shipping_state_soc_pct` | 40 | pct | 4 | specification row | footnote to section 2: '* Shipping state : About 40% capacity of fully charged … |
| 9 | `diameter_mm` | 18.3 | mm | 5 | specification row | 3.2 Dimension — Diameter: 18.3 + 0.2/-0.3 mm ( Max. 18.5 mm ). 'Diameter is def… |
| 10 | `height_mm` | 65 | mm | 5 | specification row | 3.2 Dimension — Height: 65.0 ± 0.2 mm ( Max. 65.2 mm ) |
| 11 | `initial_capacity_ah` | 3 | Ah | 6 | specification row | 4.2.2 Initial Capacity | Cell shall be charged per 4.1.1 and discharged per 4.1… |
| 12 | `cycle_life_retention_pct` | 60 | pct | 6 | specification row | 4.2.3 Cycle Life | end-of-life threshold for both rates: ≥ 60 % (of Cnom in 2.1) |
| 13 | `thermal_shock_capacity_recovery_pct` | 80 | pct | 7 | specification row | 4.3.3 Thermal Shock Test | 72ºC (8h) ← 3hrs → -20ºC (8h) for 8 cycles with cell… |

**Conflicting condition fields on matched pairs (2)**

| # | property | field | reference | 2nd |
|---:|---|---|---|---|
| 1 | `cycle_life_cycles` | `end_condition` | capacity >= 60% of nominal at cycle 301 | 301st cycle discharge capacity >= 60% of Cnom |
| 2 | `cycle_life_cycles` | `end_condition` | capacity >= 60% of nominal at cycle 201 | 201st cycle discharge capacity >= 60% of Cnom |

### Panasonic NCR18650B (marketing sheet, image-only)

**Naming-only — same value, different property name (1)**

| # | reference property | 2nd property | value | unit | p(ref) | p(2nd) |
|---:|---|---|---|---|---:|---:|
| 1 | `minimum_capacity_ah` | `nominal_capacity_ah` | 3.25 | Ah | 1 | 1 |

**In the reference set only (1)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `discharge_cutoff_v` | 2.5 | V | 2 | read from graph | graph annotation: 'Discharge: CC: Variable Current (E.V.:2.50V)' |

**In the second set only (18)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `diameter_mm` | 18.25 | mm | 1 | specification row | Specifications: Dimensions (Max.) — 'Maximum size without tube' | (D) | 18.25mm |
| 2 | `height_mm` | 65.1 | mm | 1 | specification row | Specifications: Dimensions (Max.) — 'Maximum size without tube' | (H) | 65.10mm |
| 3 | `height_mm` | 64.93 | mm | 1 | specification row | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assemblin… |
| 4 | `diameter_mm` | 18.2 | mm | 1 | specification row | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assemblin… |
| 5 | `terminal_diameter_mm` | 7.9 | mm | 1 | specification row | Drawing area: 'Dimensions(Typ.) of Bare Cell — Discharged State after Assemblin… |
| 6 | `std_charge_end_current_a` | 0.065 | A | 2 | specification row | Plot caption 'Discharge Rate Characteristics for NCR18650B': Charge:CC-CV:1.625… |
| 7 | `discharge_capacity_ah` | 3.31 | Ah | 2 | read from graph | Plot 'Discharge Rate Characteristics for NCR18650B', Temp:25°C, Charge:CC-CV:1.… |
| 8 | `discharge_capacity_ah` | 3.24 | Ah | 2 | read from graph | Plot 'Discharge Rate Characteristics for NCR18650B': the 0.5CA curve reaches 2.… |
| 9 | `discharge_capacity_ah` | 3.28 | Ah | 2 | read from graph | Plot 'Discharge Rate Characteristics for NCR18650B': the 1.0CA curve reaches 2.… |
| 10 | `discharge_capacity_ah` | 3.29 | Ah | 2 | read from graph | Plot 'Discharge Rate Characteristics for NCR18650B': the 2.0CA curve reaches 2.… |
| 11 | `discharge_capacity_ah` | 3.34 | Ah | 3 | read from graph | Plot 'Discharge Temperature Characteristics for NCR18650B', Charge:CC-CV:1.625A… |
| 12 | `discharge_capacity_ah` | 3.28 | Ah | 3 | read from graph | Plot 'Discharge Temperature Characteristics for NCR18650B': the 25°C curve reac… |
| 13 | `discharge_capacity_ah` | 3.01 | Ah | 3 | read from graph | Plot 'Discharge Temperature Characteristics for NCR18650B': the 0°C curve reach… |
| 14 | `discharge_capacity_ah` | 2.89 | Ah | 3 | read from graph | Plot 'Discharge Temperature Characteristics for NCR18650B': the -10°C curve rea… |
| 15 | `discharge_capacity_ah` | 2.6 | Ah | 3 | read from graph | Plot 'Discharge Temperature Characteristics for NCR18650B': the -20°C curve rea… |
| 16 | `charge_capacity_ah` | 3.27 | Ah | 4 | read from graph | Plot 'Charge Characteristics for NCR18650B', Charge:CC-CV:1.625A-4.20V(65.0mA c… |
| 17 | `charge_capacity_ah` | 3.38 | Ah | 4 | read from graph | Plot 'Charge Characteristics for NCR18650B': the 25°C capacity trace plateaus a… |
| 18 | `charge_capacity_ah` | 3.06 | Ah | 4 | read from graph | Plot 'Charge Characteristics for NCR18650B': the 0°C capacity trace plateaus at… |

### Panasonic NCR18650B (SANYO full specification)

**Naming-only — same value, different property name (1)**

| # | reference property | 2nd property | value | unit | p(ref) | p(2nd) |
|---:|---|---|---|---|---:|---:|
| 1 | `minimum_capacity_ah` | `nominal_capacity_ah` | 3.25 | Ah | 8 | 8 |

**In the second set only (20)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `storage_temp_range_c` | [-20, 50] | degC | 8 | specification row | 5.13 Storing Conditions | less than 1 month | -20 ~ +50℃ |
| 2 | `storage_temp_range_c` | [-20, 40] | degC | 8 | specification row | 5.13 Storing Conditions | less than 3 months | -20 ~ +40℃ |
| 3 | `storage_temp_range_c` | [-20, 20] | degC | 8 | specification row | 5.13 Storing Conditions | less than 1 year | -20 ~ + 20℃ |
| 4 | `storage_capacity_recovery_pct` | 80 | pct | 8 | specification row | 5.13 Storing Conditions, Notes column (spanning all three duration rows): 'Perc… |
| 5 | `discharge_time_min` | 300 | min | 9 | specification row | 6.2 Capacity ① 'Within 1 hour, after fully charged at 25℃, the battery is disch… |
| 6 | `discharge_time_min` | 54 | min | 9 | specification row | 6.2 Capacity ② 'Within 1 hour, after fully charged at 25℃, the battery is disch… |
| 7 | `discharge_time_min` | 38 | min | 9 | specification row | 6.3 Cycle Life, criteria: 'More than 38min.' — the discharge time measured per … |
| 8 | `discharge_time_min` | 30 | min | 9 | specification row | 6.4 Temperature Characteristics ① 'Within 1 hour, after fully charged at 25℃, t… |
| 9 | `discharge_time_min` | 50 | min | 9 | specification row | 6.4 Temperature Characteristics ② 'Within 1 hour, after fully charged at 25℃, t… |
| 10 | `discharge_time_min` | 30 | min | 9 | specification row | 6.5 Storage at Fully Charged State 'After fully charged at 25℃, the battery is … |
| 11 | `discharge_time_min` | 40 | min | 9 | specification row | 6.5 Storage at Fully Charged State, second criterion: 'Then, the same battery i… |
| 12 | `discharge_time_min` | 50 | min | 10 | specification row | 6.6 Storage at Full Discharged State 'After fully charged at 25℃, the battery i… |
| 13 | `shipping_state_soc_pct` | 40 | pct | 11 | specification row | 9. Shipping Charge: 'The battery is shipped out with the approximately 40%＊ cha… |
| 14 | `max_charge_current_a` | 1.625 | A | 11 | specification row | 10-1(1) Charge: 'Regarding NCR18650B, the charging current should not exceed 1.… |
| 15 | `max_charge_voltage_v` | 4.2 | V | 11 | specification row | 10-1(1) Charge: 'The charging voltage should not exceed 4.20V /cell.' |
| 16 | `precharge_threshold_voltage_v` | 3 | V | 11 | specification row | 10-1(1) Charge: 'If battery voltage goes down to less than 3.0V/cell, the batte… |
| 17 | `precharge_current_a` | 0.32 | A | 11 | specification row | 10-1(1) Charge: 'the battery should be charged by pre-charge current of maximum… |
| 18 | `over_discharge_limit_v` | 2 | V | 11 | specification row | 10-1(3) Over discharge: 'Do not discharge the battery less than 2.0V/cell.' |
| 19 | `storage_humidity_max_pct` | 70 | pct | 13 | specification row | 11-1 Storage Temperature and Humidity (Within 3 months): 'Cells should be store… |
| 20 | `series_discharge_cutoff_v` | 2.75 | V | 13 | specification row | 12-1 Series Connections Precautions: 'If cells are connected in series, the dis… |

**Conflicting condition fields on matched pairs (1)**

| # | property | field | reference | 2nd |
|---:|---|---|---|---|
| 1 | `cycle_life_cycles` | `end_condition` | discharge time > 38 min at 3.25A after 300 cycles (vs 54-min initial-capacity criterion, i.e. ~70% retention) | discharge time > 38 min measured per 6.2 (2) after 300 cycles |

### Samsung INR18650-25R (held out)

**Naming-only — same value, different property name (22)**

| # | reference property | 2nd property | value | unit | p(ref) | p(2nd) |
|---:|---|---|---|---|---:|---:|
| 1 | `max_charge_voltage_v` | `charge_voltage_v` | 4.2 | V | 3 | 3 |
| 2 | `rapid_charge_time_min` | `fast_charge_time_min` | 60 | min | 3 | 3 |
| 3 | `cell_height_mm` | `height_mm` | 64.85 | mm | 3 | 3 |
| 4 | `cell_diameter_mm` | `diameter_mm` | 18.33 | mm | 3 | 3 |
| 5 | `rel_discharge_capacity_m20c_pct` | `capacity_retention_at_temp_pct` | 60 | pct | 5 | 5 |
| 6 | `rel_discharge_capacity_m10c_pct` | `capacity_retention_at_temp_pct` | 75 | pct | 5 | 5 |
| 7 | `rel_discharge_capacity_0c_pct` | `capacity_retention_at_temp_pct` | 80 | pct | 5 | 5 |
| 8 | `rel_discharge_capacity_25c_pct` | `capacity_retention_at_temp_pct` | 100 | pct | 5 | 5 |
| 9 | `rel_discharge_capacity_60c_pct` | `capacity_retention_at_temp_pct` | 100 | pct | 5 | 5 |
| 10 | `rel_charge_capacity_0c_pct` | `capacity_retention_at_temp_pct` | 80 | pct | 5 | 5 |
| 11 | `rel_charge_capacity_5c_pct` | `capacity_retention_at_temp_pct` | 90 | pct | 5 | 5 |
| 12 | `rel_charge_capacity_25c_pct` | `capacity_retention_at_temp_pct` | 100 | pct | 5 | 5 |
| 13 | `rel_charge_capacity_45c_pct` | `capacity_retention_at_temp_pct` | 95 | pct | 5 | 5 |
| 14 | `rel_charge_capacity_50c_pct` | `capacity_retention_at_temp_pct` | 95 | pct | 5 | 5 |
| 15 | `rel_capacity_std_charge_pct` | `relative_discharge_capacity_pct` | 100 | pct | 5 | 5 |
| 16 | `rel_capacity_rapid_charge_pct` | `relative_discharge_capacity_pct` | 98 | pct | 5 | 5 |
| 17 | `rel_discharge_capacity_0_5a_pct` | `relative_discharge_capacity_pct` | 100 | pct | 5 | 5 |
| 18 | `rel_discharge_capacity_5a_pct` | `relative_discharge_capacity_pct` | 97 | pct | 5 | 5 |
| 19 | `rel_discharge_capacity_10a_pct` | `relative_discharge_capacity_pct` | 100 | pct | 5 | 5 |
| 20 | `rel_discharge_capacity_15a_pct` | `relative_discharge_capacity_pct` | 97 | pct | 5 | 5 |
| 21 | `rel_discharge_capacity_20a_pct` | `relative_discharge_capacity_pct` | 95 | pct | 5 | 5 |
| 22 | `ex_factory_soc_pct` | `shipping_state_soc_pct` | 50 | pct | 5 | 5 |

**In the second set only (5)**

| # | property | value | unit | page | stratum | conditions |
|---:|---|---|---|---:|---|---|
| 1 | `std_charge_end_current_a` | 0.125 | A | 3 | specification row | 3.3 Standard charge | CCCV, 1.25A, 4.20 ± 0.05 V, 125mA cut-off |
| 2 | `fast_charge_end_current_a` | 0.1 | A | 3 | specification row | 3.4 Rapid charge | CCCV, 4A, 4.20 ± 0.05 V, 100mA cut-off |
| 3 | `recommended_recharge_release_temp_c` | 45 | degC | 3 | specification row | 3.11 Operating temperature (surface temperature) | Charge : 0 to 50℃ (recommend… |
| 4 | `recommended_redischarge_release_temp_c` | 60 | degC | 3 | specification row | 3.11 Operating temperature (surface temperature) | Discharge: -20 to 75℃ (recom… |
| 5 | `storage_capacity_remaining_pct` | 90 | pct | 5 | specification row | 7.11 Storage characteristics: 'Standard rated discharge capacity after storage … |

**Conflicting condition fields on matched pairs (2)**

| # | property | field | reference | 2nd |
|---:|---|---|---|---|
| 1 | `cycle_life_cycles` | `end_condition` | capacity >= 1,500mAh (60% of nominal) after 250 cycles | capacity >= 1500 mAh, i.e. 60% of nominal capacity, at cycle 250 |
| 2 | `storage_capacity_recovery_pct` | `temperature_c` | 60 | 25 |
