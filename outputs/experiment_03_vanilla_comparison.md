# Experiment 03 — vanilla LLM vs BatteryKG on specification questions

62 questions (one per gold claim). Vanilla = llama-3.3-70b-versatile (temp 0, 3 runs, no documents/retrieval). BatteryKG = deterministic Claim-node lookup (no LLM); conflicting variants listed with sources (correct-with-provenance).

## Headline table

| metric | vanilla LLM | BatteryKG |
|---|---|---|
| value accuracy | 0.306 | 1.000 |
| conditions stated & correct (n=27) | 0.111 | 0.852 |
| specific source attributed | 0.000 | 1.000 |
| uncertainty expressed when wrong | 0.302 | n/a (abstains on missing data) |
| run-to-run consistency | 0.887 | 1.000 (deterministic) |

KG answers listing multiple conflicting variants with sources: 4 questions (scored correct-with-provenance when the gold value is among them).

Scoring note: conditions accuracy requires every numeric value of the gold's parsed condition fields to appear in the answer. The KG's misses on this metric are artifacts of that strictness — the gold annotations normalize wording the source states non-numerically ('1 week' -> 7 days, a 4.2-2.0 V window -> 100% DOD) while the KG answer quotes the source verbatim. The same rule applies to both conditions, so the comparison is conservative for the KG.

## Vanilla error taxonomy (run 1)

| class | count | meaning |
|---|---|---|
| fabricated_or_stale | 39 | matches nothing we hold (manual review needed to split fabricated vs stale) |
| variant_value | 3 | matches a DIFFERENT document's value that we hold |
| refused | 1 | declined to state a value (with uncertainty) |

## Example pairs (for the paper figure)

### What is the maximum charge current of the LG Chem 18650HG2?
*Gold: 6.0 A (lg_inr18650hg2:max_charge_current_a:5); vanilla class: variant_value*

**Vanilla:** 4A (constant current), test conditions: 25°C, 3.0V cut-off.

**BatteryKG:** 6.0 A (conditions: 2.5 Max. Charge Current: 6000mA; source: lg_inr18650hg2.pdf p.4)

### What is the nominal voltage of the A123 APR18650M1A?
*Gold: 3.3 V (a123_apr18650m1a:nominal_voltage_v:1); vanilla class: fabricated_or_stale*

**Vanilla:** 3.7 V (at 50% SOC, 20°C, according to the A123 Systems datasheet)

**BatteryKG:** 3.3 V (conditions: unspecified; source: a123_apr18650m1a.pdf p.1)

### What is the nominal capacity of the A123 APR18650M1A?
*Gold: 1.1 Ah (a123_apr18650m1a:nominal_capacity_ah:0); vanilla class: correct*

**Vanilla:** 1100 mAh.

**BatteryKG:** 1.1 Ah (conditions: 'Nominal capacity and voltage: 1.1Ah, 3.3 V' — no measurement conditions stated; source: a123_apr18650m1a.pdf p.1)

