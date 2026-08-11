# Experiment 11 — held-out evaluation on Samsung SDI INR18650-25R

> **Gold revised twice after blind extraction; predictions unchanged.**
> v2: annotation-convention alignment. v3: gold property names aligned to the
> canonical vocabulary — names only, values/pages/conditions untouched.
> The extraction was and remains blind: the model never saw the gold, and no
> prediction was re-derived. All numbers below score the *frozen* predictions of
> the blind run. §6 carries both diffs and the replay-integrity proof.

**Result in one line:** on a genuinely held-out datasheet the pipeline is
**precise (0.875) and hallucinates nothing (0/16), but recovers only 34 % of the
gold claims** — and consensus and the Validator change *nothing*, because all
three extraction runs were already near-identical.

## Setup

| | |
|---|---|
| Document | Samsung SDI INR18650-25R, *Specification of product*, Rev 1.0 (Mar 2014), 16 pages |
| Text snapshot | `data/claims/extracted_text/samsung_inr18650_25r.txt` (86,750 chars, 17 page markers) |
| Gold | `data/gold/samsung_inr18650_25r_gold.yaml` **v3**, 41 claims, hand-written, read **only** at scoring time |
| Model | `llama-3.3-70b-versatile` via Groq, `temperature=0`, `seed=42`, frozen schema-locked prompt (`src/agents/extractor.py`) |
| Pipeline | 3 extraction runs → deterministic Validator → ≥2/3 consensus → conditions-focused second pass (3 runs, field majority, value-in-source) |
| Cost | blind run: 51 LLM calls, 74,658 prompt + 5,492 completion tokens. Re-scoring: **0 calls** |

Held-out in the strict sense: this document contributed nothing to the prompt,
the property vocabulary, the Validator's plausibility bounds, or the conditions
pass. All of those were frozen before it was annotated. The v3 rename went the
other way — the gold was aligned to the vocabulary, and
`src/agents/extractor.py`'s `PROPERTY_VOCAB` was **not** touched, so the
extraction prompt stays frozen and comparable.

Reproduce:

```
python -m experiments.exp11_heldout_samsung.run_heldout   # blind extraction + scoring (LLM)
python -m experiments.exp11_heldout_samsung.rescore       # re-score only, no LLM path
python -m experiments.exp11_heldout_samsung.analyze       # error buckets + diagnostics
```

`rescore.py` cannot reach an LLM by construction: raw runs come from
`outputs/extraction_raw/`, the final claim set from `predictions_final.json`,
and a missing cache file is a hard error rather than a new call. It also asserts
that the re-derived consensus+Validator set equals the frozen one. Re-running it
is a no-op: the archived scores of earlier gold versions are written once and
never recomputed from the current file.

### Without the document text

The text snapshot is derived verbatim from a copyrighted datasheet and does not
ship in the public release, and neither do the cached raw runs, which quote it
back. To reproduce the full re-score, download the datasheet from the URL and
retrieval date recorded in `data/README.md` and rebuild the snapshot:

```
python -m src.agents.pdf_text          # writes data/claims/extracted_text/
```

Without it, `rescore.py` runs in a limited mode instead of failing: it scores
the frozen predictions against the shipped gold, so TP/FP/FN, precision, recall,
F1 and condition accuracy are exact — none of them reads the document. What it
cannot do is answer "does this value appear in the source", so the hallucination
count and the Validator/consensus replay are reported as **unavailable** rather
than as a number, and `results.json` is left untouched so the full run's record
survives. The numbers of the full run are the tables below.

## 1. Value-stage metrics (paper metric: exact property + value within 2 %)

Gold v3, n = 41.

| Config | Gold (n) | Pred. | TP | FP | FN | P | R | F1 | Cond. acc. | Halluc. |
|---|---|---|---|---|---|---|---|---|---|---|
| Raw single run (run 1) | 41 | 16 | 14 | 2 | 27 | 0.875 | 0.342 | 0.491 | 0.786 | 0 |
| Consensus only (≥2/3, no Validator) | 41 | 16 | 14 | 2 | 27 | 0.875 | 0.342 | 0.491 | 0.786 | 0 |
| Consensus + Validator | 41 | 16 | 14 | 2 | 27 | 0.875 | 0.342 | 0.491 | 0.786 | 0 |

The three rows are identical, and that is the honest finding, not a bug:

- Runs 1 and 3 returned the **same 16 claims**; run 2 returned those 16 plus
  `std_charge_time_h = 180 min` and `fast_charge_time_h = 60 min`. At
  `temperature=0` with a fixed seed there is almost no disagreement for
  consensus to exploit.
- Consensus therefore only drops run 2's two singletons (1/3 support).
- The Validator rejected exactly one claim across all three runs —
  `std_charge_time_h = 180` with `unit: min`, flagged `implausible` against the
  bounds `[0.05, 50] h`. That claim was already going to lose the 2/3 vote, so
  the rejection has no effect on the final set.
- **0 hallucinations** in every configuration. Every predicted number is present
  in the source text.

`Cond. acc.` in this table is the single-pass condition score on matched pairs
(§2 breaks it out).

## 2. Conditions-focused second pass

Scored on the 14 matched (gold, prediction) pairs:

| Pass | Cond. accuracy | Over-extraction rate |
|---|---|---|
| Single-pass (conditions from the value extraction) | 11/14 = **0.786** | 0/14 = **0.000** |
| Focused second pass (3 runs + field majority + value-in-source) | 10/14 = **0.714** | 4/14 = **0.286** |

The second pass is a **net negative**: one pair worse, and it invents conditions
on four pairs where the single pass stayed silent. It is far more eager — it
fills in `temperature_c: 25`, `charge_voltage_v: 4.2`, `discharge_cutoff_v: 2.5`
for claims whose gold conditions are narrower or empty. Its value-in-source gate
did fire correctly twice, dropping an invented `dod_pct: 40` from both
`cycle_life_cycles` and `cycle_life_retention_pct` — so the gate works; the
problem is upstream eagerness, not ungrounded numbers.

## 3. Remaining mismatches (gold v3): 27 FN, 2 FP, 0 hallucinations

### 3.1 (a) Table cells §7.6–7.9 never extracted — 17 FN

All of the temperature- and rate-dependence grids on page 5:
`rel_discharge_capacity_{m20c,m10c,0c,25c,60c}_pct`,
`rel_charge_capacity_{0c,5c,25c,45c,50c}_pct`,
`rel_capacity_{std,rapid}_charge_pct`,
`rel_discharge_capacity_{0_5a,5a,10a,15a,20a}_pct`.

The model extracted nothing from these grids in any of the three runs. After the
v3 renames this is **63 % of all remaining FNs** and by far the dominant error
source: layout-mode text turns a 2-D table into whitespace-aligned rows, and the
extractor never reconstructs the row × column claim structure.

### 3.2 (b) Ordinary spec rows missed — 9 FN

| Gold property | Value | Note |
|---|---|---|
| `rated_capacity_ah` | 2.45 Ah | §7.4 "≥ 2,450mAh" never extracted |
| `std_charge_time_min` | 180 min | run 2 only; Validator-rejected (`180` with `unit: min` vs `std_charge_time_h` bounds), then lost consensus |
| `rapid_charge_time_min` | 60 min | run 2 only, lost consensus |
| `cell_height_mm` | 64.85 mm | dimensions not in the extractor's `PROPERTY_VOCAB` |
| `cell_diameter_mm` | 18.33 mm | dimensions not in the extractor's `PROPERTY_VOCAB` |
| `storage_temp_range_c` | [-30, 45] °C | 3-month row; model collapsed three storage rows into one |
| `storage_temp_range_c` | [-30, 60] °C | 1-month row; same |
| `ex_factory_soc_pct` | 50 % | §7.12 shipping state not extracted |
| `ex_factory_ocv_range_v` | [3.600, 3.690] V | §7.12 not extracted |

The two `storage_temp_range_c` misses now read as "property predicted but value
differs" rather than "not extracted": since v3 the gold and the model agree on
the property name, and the model simply produced one of the three duration rows.
That is the correct diagnosis, and it is a genuine recall miss, not a naming one.

### 3.3 (c) Canonical name existed but the model chose a different one — 1 FN + 1 FP

| Gold (canonical) | Model predicted | Why it is wrong |
|---|---|---|
| `storage_capacity_recovery_pct = 90 %` | `storage_capacity_remaining_pct = 90 %` | §7.11 is a high-temperature storage **recovery** test (1 month at 60 °C, capacity re-measured afterwards) — the direct analogue of `lg_18650hg2` §4.3.2. `storage_capacity_remaining_pct` is the *other* canonical property (`lg_18650hg2` §4.3.1: capacity still present after 30 days at 23 °C). Both names were available to the model; it picked the wrong one. |

This is the only naming error left, and it is **model-side, not gold-side**: the
gold now uses the canonical name and the model did not.

### 3.4 (d) Real over-extraction — 1 FP

- `charge_voltage_v = 4.2 V` — the same 4.20 V spec the model *also* emitted
  correctly as `max_charge_voltage_v`. One physical claim, two property names.

Every predicted value appears in the source text: **0 hallucinations**.

### 3.5 (e) Wrong conditions — 4 of 14 pairs (second pass)

| Claim | Gold parsed conditions | Second-pass prediction | Why wrong |
|---|---|---|---|
| `cycle_life_cycles = 250` | `charge_current_a: 1.25, discharge_current_a: 20, end_condition: capacity ≥ 1,500mAh (60 % of nominal)` | `charge_current_a: 1.25, charge_voltage_v: 4.2, discharge_current_a: 10, discharge_cutoff_v: 2.5, temperature_c: 25, end_condition: Capacity ≥ 60 % …` | discharge current wrong: §7.10 says "maximum continuous discharge" (= 20 A per §3.7); the model substituted the 10 A used elsewhere on the page |
| `nominal_capacity_ah = 2.5` | `temperature_c: 25, discharge_current_a: 0.5` | `charge_current_a: 1.25, charge_voltage_v: 4.2, discharge_c_rate: 0.2, discharge_cutoff_v: 2.5, temperature_c: 25, charge_mode: CCCV` | the gold's 0.5 A (the explicit "500mA" of §7.3) is absent; the model kept the C-rate form (0.2C) and added the charge protocol |
| `std_charge_current_a = 1.25` | *(none — §3.1 states no test conditions)* | `charge_current_a: 1.25, charge_time_min: 180, charge_voltage_v: 4.2, discharge_c_rate: 0.2, discharge_cutoff_v: 2.5, temperature_c: 25, charge_mode: CCCV` | over-extraction: pulled the whole page-3 spec block into one claim's conditions |
| `discharge_cutoff_v = 2.5` | *(none — §3.9 states none)* | `discharge_current_a: 0.2, discharge_cutoff_v: 2.5` | over-extraction: invented a 0.2 A discharge current (the 0.2**C** rate misread as amps) |

The first is the substantive error. Resolving "maximum continuous discharge"
requires a cross-reference to §3.7, which a page-local conditions pass cannot do
by construction. The other three are all over-extraction — the same eagerness
§2 quantifies.

## 4. Diagnostics (NOT the paper metric)

| Variant | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Paper metric (exact property name) | 14 | 2 | 27 | 0.875 | 0.342 | 0.491 |
| Alias-normalised (grants the recovery/remaining confusion) | 15 | 1 | 26 | 0.938 | 0.366 | 0.526 |
| Property-agnostic (value only) — recall upper bound | 15 | — | — | 0.938 | 0.366 | — |

The alias-normalised and property-agnostic rows coincide exactly, which says the
alias map is complete: after v3 there is exactly **one** prediction that is right
on value but unaccounted for on name (§3.3). Naming is no longer a meaningful
lever — the gap between 0.938 and the paper metric is a single claim, and recall
still stalls at **0.37** because of §3.1 and §3.2.

## 5. What this means for the paper

1. **Report the drop honestly.** Held-out F1 is 0.491 with precision 0.875 and
   recall 0.342. Precision transfers; recall does not.
2. **The Validator's safety claim survives; its accuracy claim does not.**
   0 hallucinations out of 16 predictions, and the condition-side value-in-source
   gate correctly killed an invented `dod_pct: 40` twice. But on this document
   the Validator improved no metric, because there was nothing to reject.
3. **Consensus is near-free but also near-useless at temperature 0.** Three runs
   cost 3× and changed the final set by two singletons. If the paper keeps 3
   runs, it should justify them on variance across *documents*, not on this one.
4. **The conditions second pass is a net negative here** (0.786 → 0.714, with
   over-extraction 0.000 → 0.286). Report it as such, or gate it to fire only
   where the single-pass conditions are `unspecified`.
5. **Table extraction is now the whole story**: 17 of 27 remaining misses. The
   property-name problem is closed — worth +0.313 precision and +0.122 recall
   once the gold used canonical names, with a single residual model-side error.
   Second priority is the extractor's `PROPERTY_VOCAB`, which has no entry for
   cell dimensions or minute-valued charge times.

## 6. Gold revision history

Neither revision touched a prediction. `rescore.py` asserts that the frozen
prediction set equals the cached pipeline output, and replays the archived score
of **every** earlier gold version against the frozen predictions
(`replay_integrity_ok: {"v1": true, "v2": true}` in `results.json`). A replay
mismatch would mean a snapshot no longer represents what was scored.

Snapshots: `gold_v1_snapshot.json` (reconstructed, replay-verified — `data/gold/`
is untracked so v1 has no git object), `gold_v2_snapshot.json` (taken verbatim
before the v3 renames).

### v1 → v2 — annotation-convention alignment

| Change | Count |
|---|---|
| Added | 1 · `cycle_life_retention_pct = 60 pct` (v1 folded the threshold into `cycle_life_cycles`' `end_condition`) |
| Removed | 1 · duplicate `nominal_capacity_ah = 2.5 Ah` on page 3 |
| Renamed | 0 |
| Conditions changed | 0 |
| Unchanged | 40 |

### v2 → v3 — canonical property names

Names only. Values, units, pages and conditions are byte-identical (verified
field-by-field before the edit).

| Change | Count |
|---|---|
| Added / removed | 0 / 0 |
| **Renamed** | **8** |
| Conditions changed | 0 |
| Unchanged | 33 |

| v2 name | → v3 canonical name | Value |
|---|---|---|
| `rapid_charge_current_a` | `fast_charge_current_a` | 4 A |
| `discharge_cutoff_voltage_v` | `discharge_cutoff_v` | 2.5 V |
| `cell_weight_g` | `mass_g` | 45.0 g |
| `initial_impedance_mohm` | `internal_impedance_mohm` | 18 mΩ |
| `storage_recovery_pct` | `storage_capacity_recovery_pct` | 90 % |
| `storage_temp_range_1_5yr_c` | `storage_temp_range_c` | [-30, 25] °C |
| `storage_temp_range_3mo_c` | `storage_temp_range_c` | [-30, 45] °C |
| `storage_temp_range_1mo_c` | `storage_temp_range_c` | [-30, 60] °C |

The last two go beyond the seven names originally flagged: leaving sibling rows
under bespoke names while the 1.5-year row became canonical would have made the
file internally inconsistent. `lg_18650hg2` §2.10 sets the precedent — the
storage duration belongs in `stated_conditions.text`, not in the property name.

`std_charge_time_min` was **not** renamed. The canonical `std_charge_time_h`
hard-codes hours in its name, and the datasheet states 180 min; renaming without
converting would produce an hours-named claim holding minutes, and converting
would break both the "record what the datasheet SAYS" rule and the
value-in-source check. It was registered in the vocabulary instead, together with
`rapid_charge_time_min`, the two cell dimensions, the two ex-factory properties
and the 17 relative-capacity table cells — 23 new entries, via
`scripts/gen_claims_vocab.py` (now sourcing `data/gold/*.yaml` as well as
`data/claims/*.yaml`, excluding annotation TEMPLATEs). The vocabulary is
documentation for annotators and is deliberately **not** wired into
`src/agents/extractor.py`.

### Score across gold versions (same frozen predictions throughout)

| | v1 | v2 | v3 |
|---|---|---|---|
| TP | 8 | 9 | **14** |
| FP | 8 | 7 | **2** |
| FN | 33 | 32 | **27** |
| P | 0.500 | 0.562 | **0.875** |
| R | 0.195 | 0.220 | **0.342** |
| F1 | 0.281 | 0.316 | **0.491** |
| Cond. acc. (single-pass) | 6/8 = 0.750 | 6/9 = 0.667 | 11/14 = **0.786** |
| Cond. acc. (second pass) | 5/8 = 0.625 | 6/9 = 0.667 | 10/14 = **0.714** |
| Hallucinations | 0 | 0 | 0 |

v2 → v3 moved 5 claims from FN+FP to TP. Eight renames, five payoffs: the
`storage_recovery_pct` rename did not pay off (the model had picked the other
canonical name, §3.3), and `std_charge_time_min` plus the two extra
`storage_temp_range_c` rows were never in the prediction set to begin with.

## Files

| File | Contents |
|---|---|
| `run_heldout.py` | blind pipeline + paper-metric scoring (makes LLM calls) |
| `rescore.py` | re-scoring against the current gold + version diffs + replay integrity (no LLM path) |
| `analyze.py` | post-hoc error buckets and diagnostics |
| `gold_v1_snapshot.json`, `gold_v2_snapshot.json` | earlier gold versions, replay-verified |
| `results.json` | v3 metrics + full error catalog, archived v1/v2 metrics, both gold diffs, Validator rejections, conditions, usage |
| `predictions_final.json` | the 16 frozen consensus + Validator claims — **unchanged since the blind run** |
| `analysis.json` | error buckets, diagnostics, and what the latest gold revision resolved |

Raw LLM responses (3 extraction runs + 48 conditions calls) are cached under
`outputs/extraction_raw/`. The knowledge graph and all other experiments are
untouched by this experiment's code.
