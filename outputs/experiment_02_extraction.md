# Experiment 02 — LLM claim extraction vs manual gold standard

**Model:** `llama-3.3-70b-versatile` via groq, temperature 0, seed 42, 3 runs. **Gold:** 62 hand-labeled claims; **48 evaluable** (text-bearing PDFs: a123_apr18650m1a, lg_inr18650hg2, panasonic_ncr18650b_full_spec_sanyo).
**Excluded:** ['panasonic_ncr18650b'] — image-only PDF (no text layer), its 14 gold claims need OCR/vision extraction; recorded as a finding, not counted against the model.

Matching: document + property (exact) + value (2% tol, unit-scaled); conditions scored separately on matches (gold-'unspecified' requires the prediction not to invent conditions). Hallucination = unmatched prediction whose value appears nowhere in the source text. Note: hallucination-flagged items include silent unit conversions (value absent in the document's units) and schema violations (nested values) — per-item diagnosis in the catalog.

## Results (run 1)

| document | gold | pred | TP | FP | FN | P | R | F1 | cond acc | halluc |
|---|---|---|---|---|---|---|---|---|---|---|
| a123_apr18650m1a | 12 | 12 | 10 | 2 | 2 | 0.833 | 0.833 | 0.833 | 0.700 | 1 |
| lg_inr18650hg2 | 22 | 19 | 13 | 6 | 9 | 0.684 | 0.591 | 0.634 | 0.615 | 1 |
| panasonic_ncr18650b_full_spec_sanyo | 14 | 21 | 12 | 9 | 2 | 0.571 | 0.857 | 0.686 | 1.000 | 1 |
| **OVERALL** | 48 | 52 | 35 | 17 | 13 | **0.673** | **0.729** | **0.700** | **0.771** | **3** |

## Run-to-run variance (temperature 0, fixed seed)

| run | n_pred | P | R | F1 | cond acc | halluc |
|---|---|---|---|---|---|---|
| 1 | 52 | 0.673 | 0.729 | 0.700 | 0.771 | 3 |
| 2 | 53 | 0.698 | 0.771 | 0.733 | 0.730 | 3 |
| 3 | 48 | 0.729 | 0.729 | 0.729 | 0.771 | 3 |

F1 spread across runs: 0.033 (std 0.0146). Pairwise Jaccard of predicted claim sets: 0.793, 0.768, 0.836 (1.0 = byte-identical runs).

## Error catalog (run 1)

`FN` missed gold claim · `FP` spurious extraction (value exists in text) · `HALL` hallucination (value not in text) · `COND` matched claim, wrong conditions

```
COND [a123_apr18650m1a] charge_voltage_v=3.6: conditions wrong — gold 'Recommended charge and cut-off V at 25°C: 3.6V to 2V...' vs pred 'unspecified'
COND [a123_apr18650m1a] fast_charge_current_a=4.0: conditions wrong — gold 'Recommended fast charge current: 4A to 3.6V CCCV, 15 min...' vs pred 'unspecified'
COND [a123_apr18650m1a] std_charge_current_a=1.5: conditions wrong — gold 'Recommended standard charge method: 1.5A to 3.6V CCCV, 45 min...' vs pred 'unspecified'
COND [lg_inr18650hg2] fast_discharge_current_a=[10.0, 20.0]: conditions wrong — gold '2.6.2 Fast Discharge: CC 10000mA, 20000mA; end voltage 2.0V...' vs pred 'Constant current'
COND [lg_inr18650hg2] nominal_capacity_ah=3.0: conditions wrong — gold '2.1 Capacity: Nominal 3000 mAh (Cnom), Std. charge/discharge (1500mA C...' vs pred 'Std. charge / discharge'
COND [lg_inr18650hg2] std_charge_current_a=1.5: conditions wrong — gold '2.3.1 Standard Charge: CC 1500mA, CV 4.2V, end (cut-off) 50mA...' vs pred 'Constant current'
COND [lg_inr18650hg2] std_discharge_current_a=0.6: conditions wrong — gold '2.6.1 Standard Discharge: CC 600mA, end voltage (cut off) 2.0V...' vs pred 'Constant current'
COND [lg_inr18650hg2] storage_capacity_recovery_pct=80: conditions wrong — gold '4.3.2 High Temperature Storage Test: capacity recovery rate >= 80% of ...' vs pred 'after 1 week at 60°C'
FN   [a123_apr18650m1a] cycle_life_retention_pct=93 pct: not extracted at all
FN   [a123_apr18650m1a] discharge_cutoff_v=2.0 V: value found under different property (['charge_cutoff_v'])
FN   [lg_inr18650hg2] capacity_retention_at_temp_pct=100 pct: not extracted at all
FN   [lg_inr18650hg2] capacity_retention_at_temp_pct=60 pct: value found under different property (['cycle_life_retention_pct'])
FN   [lg_inr18650hg2] capacity_retention_at_temp_pct=80 pct: not extracted at all
FN   [lg_inr18650hg2] capacity_retention_at_temp_pct=95 pct: not extracted at all
FN   [lg_inr18650hg2] cycle_life_cycles=200 cycles: property predicted but value differs (gold 200 cycles; pred [[300, 200]])
FN   [lg_inr18650hg2] cycle_life_cycles=300 cycles: property predicted but value differs (gold 300 cycles; pred [[300, 200]])
FN   [lg_inr18650hg2] fast_charge_current_a=4.0 A: not extracted at all
FN   [lg_inr18650hg2] max_charge_voltage_v=4.2 V: property predicted but value differs (gold 4.2 V; pred [[4.15, 4.25]])
FN   [lg_inr18650hg2] storage_temp_range_c=[-20, 20] degC: property predicted but value differs (gold [-20, 20] degC; pred [[[-20, 60], [-20, 45], [-20, 20]]])
FN   [panasonic_ncr18650b_full_spec_sanyo] cycle_life_cycles=300 cycles: not extracted at all
FN   [panasonic_ncr18650b_full_spec_sanyo] minimum_capacity_ah=3.25 Ah: value found under different property (['nominal_capacity_ah'])
FP   [a123_apr18650m1a] charge_cutoff_v=2 V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [lg_inr18650hg2] charge_voltage_v=4.2 V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [lg_inr18650hg2] cycle_life_cycles=[300, 200] cycles: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [lg_inr18650hg2] cycle_life_retention_pct=60 pct: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [lg_inr18650hg2] discharge_cutoff_v=2 V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [lg_inr18650hg2] max_charge_voltage_v=[4.15, 4.25] V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] charge_current_a=0.32 A: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] discharge_cutoff_v=2.0 V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] discharge_temp_range_c=[-20, 60] ℃: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] max_charge_voltage_v=4.23 V: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] nominal_capacity_ah=3250 mAh: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] storage_capacity_remaining_pct=80 pct: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] storage_temp_range_c=[-20, 20] ℃: value exists in document but is not a gold claim (over-extraction or wrong property name)
FP   [panasonic_ncr18650b_full_spec_sanyo] storage_temp_range_c=[-20, 40] ℃: value exists in document but is not a gold claim (over-extraction or wrong property name)
HALL [a123_apr18650m1a] std_charge_time_h=0.75 h: value NOT FOUND anywhere in the document text
HALL [lg_inr18650hg2] storage_temp_range_c=[[-20, 60], [-20, 45], [-20, 20]] degC: value NOT FOUND anywhere in the document text
HALL [panasonic_ncr18650b_full_spec_sanyo] storage_temp_range_c=[[-20, 50], [-20, 40], [-20, 20]] ℃: value NOT FOUND anywhere in the document text
```

*LLM-extracted claims are NOT loaded into the KG; the KG carries the manual gold standard only, pending a decision based on these numbers.*

## 02b: Validated pipeline

Deterministic Validator (schema gate with mechanical unnesting; value-in-source with number normalization; per-property plausibility bounds; 'X ± Y' tolerance normalization) applied per run, then >= 2-of-3 consensus. Same cached extractions and metrics as experiment 02.

| pipeline | n_pred | TP | FP | FN | P | R | F1 | cond acc | halluc |
|---|---|---|---|---|---|---|---|---|---|
| raw single-run | 52 | 35 | 17 | 13 | 0.673 | 0.729 | 0.700 | 0.771 | 3 |
| consensus-only (>=2/3) | 47 | 36 | 11 | 12 | 0.766 | 0.750 | 0.758 | 0.778 | 1 |
| consensus + Validator | 49 | 38 | 11 | 10 | 0.776 | 0.792 | 0.784 | 0.763 | 0 |

### Rejected-claim log (Validator, all runs)

5 rejections across 3 runs x 3 docs; 4 range claims tolerance-normalized (not rejections). **False-rejection rate: 0/5 (0.0%)** — rejected claims that would have matched gold.

```
value_not_in_source  [a123_apr18650m1a run1] std_charge_time_h=0.75 
malformed_value      [a123_apr18650m1a run2] charge_temp_range_c=unspecified 
value_not_in_source  [a123_apr18650m1a run2] std_charge_time_h=0.75 
malformed_value      [a123_apr18650m1a run3] charge_temp_range_c=unspecified 
value_not_in_source  [a123_apr18650m1a run3] std_charge_time_h=0.75 
```

### Conditions accuracy before/after

raw 0.771 -> validated 0.763. The Validator checks values, not conditions — dropped charge-protocol conditions (the main LLM weakness from 02) remain uncorrected by design. Fixing that requires an LLM-side change (conditions-focused prompting or a second extraction pass), not more validation.

## 02c: Conditions-focused extraction

Second, conditions-focused LLM pass for accepted claims only (value pipeline frozen): one call per claim with just the claim's source page, 3 runs, field-level majority consensus (ties -> unspecified), and condition-side value-in-source validation.

| configuration | n | claim-level cond acc | over-extraction rate |
|---|---|---|---|
| (a) single-pass extractor | 38 | 0.763 | 0.000 |
| (b) focused second pass (1 run) | 38 | 0.816 | 0.211 |
| (c) focused + consensus + validation | 38 | 0.789 | 0.211 |

### Field-level accuracy (gold claims stating that field group)

| group | (a) single-pass | (b) focused | (c) focused+val+cons |
|---|---|---|---|
| temperature | 4/6 (0.67) | 4/6 (0.67) | 4/6 (0.67) |
| charge | 0/4 (0.00) | 3/4 (0.75) | 4/4 (1.00) |
| discharge | 3/6 (0.50) | 5/6 (0.83) | 5/6 (0.83) |
| dod | 1/1 (1.00) | 1/1 (1.00) | 1/1 (1.00) |
| eol | 0/0 (–) | 0/0 (–) | 0/0 (–) |

### Per-document claim-level conditions accuracy

| document | (a) | (b) | (c) |
|---|---|---|---|
| a123_apr18650m1a | 8/11 (0.73) | 10/11 (0.91) | 10/11 (0.91) |
| lg_inr18650hg2 | 9/15 (0.60) | 11/15 (0.73) | 11/15 (0.73) |
| panasonic_ncr18650b_full_spec_sanyo | 12/12 (1.00) | 10/12 (0.83) | 9/12 (0.75) |

### Cost (3-document pipeline)

| stage | LLM calls | tokens |
|---|---|---|
| values only (3 docs x 3 runs) | 9 | ~107,921 (est., chars/4 from cached artifacts) |
| + conditions pass | 9 + 147 = 156 | values est. + 159,130 measured |

The conditions stage multiplies call count by ~17x — the price of per-claim focus.

### Reading the over-extraction rate

The detector is unit-blind (a prediction of 0.6 A is grounded by a quote saying '600mA'). Manual review of the flagged cases shows two distinct phenomena rather than free invention: (i) **cross-reference resolution** — the model attaches protocols from sections the claim's row explicitly references (e.g. LG 4.3.x tests 'charged per 4.1.1, discharged per 4.1.2'), which is arguably *richer* than the gold quote and correct in substance; (ii) **neighbour-row misattribution** — conditions of an adjacent spec row attached to the wrong claim (e.g. the standard-charge protocol attached to 'Max. Charge Voltage'). Only (ii) is a genuine error; condition-side value-in-source validation cannot catch it by construction, because the numbers ARE on the page — they just belong to a different claim. Distinguishing (i) from (ii) needs row/cell-level grounding, not more validation.

Also honestly noted: the EOL-threshold field group scored 0/0 — the cycle-life claims carrying EOL conditions were largely missed by the frozen VALUE pipeline (e.g. LG's two cycle-life claims merged into a range), so their conditions never reached this stage. Improving cycle-life claim extraction remains the prerequisite for condition-complete cycle-life claims.
