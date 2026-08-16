# Experiment 12 — row-level table grounding pilot (Samsung SDI INR18650-25R)

> Answers Reviewer 2, round 2, comment 2.5: row-level table grounding was
> identified as the fix for experiment 11's recall gap but never implemented.
> This is the minimal pilot that measures it, on the same held-out document,
> against the same 41-claim gold, with the same matching rules.

**Result in one line:** isolating each table region into its own extraction
context **recovers every one of the 17 table-grid cells the frozen pipeline
missed** — 17/17 at value level, 10/17 under the paper's exact-name metric —
lifting recall from **0.342 to 0.634** at a precision cost of 0.875 → 0.591.
The fix is real; it is not free.

## Setup

| | |
|---|---|
| Document | Samsung SDI INR18650-25R, *Specification of product*, Rev 1.0 (Mar 2014), 17 pages — the same held-out document as experiment 11 |
| Gold | `data/gold/samsung_inr18650_25r_gold.yaml` **v3**, 41 claims, unchanged, read only at scoring time |
| Text snapshot | `data/claims/extracted_text/samsung_inr18650_25r.txt` — the **frozen** snapshot; every value is still gated against it, not against the PDF this experiment re-reads |
| Model | `llama-3.3-70b-versatile` via Groq, `temperature=0`, `seed=42` |
| Pipeline | 15 row-level extraction units × 3 runs → deterministic Validator (unchanged) → ≥2/3 consensus (unchanged) |
| Cost | **45 LLM calls**, 68,070 prompt + 10,286 completion tokens |

Reproduce:

```
python -m experiments.exp12_table_grounding.table_grid       # deterministic, no LLM
python -m experiments.exp12_table_grounding.run_grounding    # 3 cached runs (LLM)
python -m experiments.exp12_table_grounding.score            # re-score only, no LLM path
pytest tests/test_exp12_table_grid.py                        # 15 tests, no PDF needed
```

Like `exp11/rescore.py`, `score.py` cannot reach an LLM by construction: it reads
cached predictions and a missing artifact is a hard error, never a new call.
Re-running `run_grounding.py` with the cache present makes zero calls.

### In the public release

Two inputs are withheld because they reproduce a copyrighted document, so on the
public tree `score.py` raises `FileNotFoundError` until you rebuild them:

| withheld | rebuild |
|---|---|
| `data/claims/extracted_text/samsung_inr18650_25r.txt` — the frozen text snapshot `score.py` reads | download the datasheet listed in `data/README.md` §3, then `python -m src.agents.pdf_text` |
| the `rows` / rendered `units` fields of `tables_extracted.json` — the table cells verbatim | `python -m experiments.exp12_table_grounding.table_grid` (needs the PDF above) |

`tables_extracted.json` ships with its geometry and classification intact
(`region_id`, `pages`, `n_cols`, `bboxes`, `kind`, `stitched`, `stitch_note`,
`qty_rows`, `counts`), so the page-boundary stitch and the 4-grid/11-spec-row
split remain inspectable without republishing the tables themselves. Every
metric in §4–§6 is already computed in `results.json`.

## 1. What changed, and what deliberately did not

The **only** thing that changes relative to the frozen pipeline is the extraction
context. Experiment 11 hands the model 86,750 characters of layout-mode text in
one call. Here `table_grid.py` deterministically isolates each table region and
the model sees **one** of them per call, with the frozen page text alongside it
for conditions.

Unchanged and imported, not copied:

- `extractor.SYSTEM_PROMPT` — verbatim, with a per-kind addendum appended. The
  addendum is the entire prompt diff.
- `extractor._parse_json_claims` — still the final schema gate for both unit kinds.
- `validator.validate_claims` / `validator.consensus` — untouched. Every value
  still passes value-in-source against the frozen text snapshot.
- `src/agents/evaluation.py` — the same `values_match`, the same greedy 1:1
  matching, the same 2 % tolerance.

Nothing under `experiments/exp11_heldout_samsung/` is written. Experiment 11's
frozen predictions are read as the baseline and the merge partner.

## 2. The deterministic stage

`pdfplumber` ruling-line detection over all 17 pages → 24 tables → **7 content
tables** after dropping the repeated `Spec. No.` page header → **6 regions**
after one page-boundary stitch → **15 extraction units** (4 grids + 11
specification rows). The page-17 revision-history table is classified `other`
and yields no units.

### 2.1 The §7.6 page-boundary split — why flat text never had a chance

Section 7.6's grid is **torn in half by a page break**. The last table on page 4
holds the grid's caption and its axis row — five temperature headers, no data —
and the table ends there. The first table on page 5 is the matching value row:
five percentages across the same five columns, with no caption, no axis and no
free text of its own. Neither half is a grid; each is a single quantity row.

In layout-mode text the value row therefore appears on page 5 as a bare line of
five percentages, 60 lines below its own caption and on the other side of a page
marker, with the page header and confidentiality footer interleaved between
them. No amount of prompting recovers the row × column structure from that,
because the structure is not present in the text: **the axis and the values
never appear on the same page.** This is the mechanical explanation for the
single largest error bucket in experiment 11, and it belongs in the paper text.

`stitch_regions` rejoins the halves under a conservative, stated rule: a grid
needs two quantity rows (an axis row and a value row); when the last table on
page *n* has exactly one, the first table on page *n+1* has exactly one and no
free text at all, and both have the same column count, they are one grid. It
cannot fire when either half is already a complete grid. On this document it
fires **exactly once**, and the firing is logged to `tables_extracted.json`.
`tests/test_exp12_table_grid.py` asserts the load-bearing fact: neither half
classifies as a grid alone, the stitched region does.

This is a minimal heuristic validated on one document, not a general
table-stitching solution.

### 2.2 Preserving intra-cell line breaks

pdfplumber returns a multi-line table cell newline-joined. §3.12 is three
storage-temperature rows inside **one** cell, and flattening it to a single line
is what makes them look like a single claim — which is exactly how experiment 11
read them (§3.2 of that README: "model collapsed three storage rows into one").
`clean_cell` preserves the cell's own line breaks and `render_spec_row` keeps one
source line per line. Both recovered `storage_temp_range_c` claims below come
from that change.

## 3. Convention leakage — what was disclosed to the model

**Grid cells** are named by a deterministic namer. The model emits
`(family, axis_kind, axis_value)`; `compose_property_name` composes the
snake_case property name:

```
temperature_c → f"{family}_{'m' if v<0 else ''}{fmt(|v|)}c_pct"
current_a     → f"{family}_{fmt(v)}a_pct"
charge_mode   → f"{family}_{std|rapid}_charge_pct"
families        rel_discharge_capacity | rel_charge_capacity | rel_capacity
```

**The naming rule was authored knowing the gold's naming scheme.** That is the
disclosed convention leakage of this pilot, and it is why the exact-name column
is reported alongside a value-level column that does not depend on it. What was
**not** disclosed: the gold's values, the grid structure, the axis values, which
grid maps to which family, and how many claims any grid contains. The model was
given the three family names and a one-line meaning for each — the same level of
description `data/claims/README.md` gives human annotators.

**Specification rows carry no naming rule at all** — an asymmetry worth stating,
because it makes their recoveries leakage-free. The model names them from the
frozen `extractor.PROPERTY_VOCAB` exactly as in experiment 11, and gold names
outside that vocabulary (`cell_height_mm`, `cell_diameter_mm`,
`std_charge_time_min`, `rapid_charge_time_min`) are therefore unreachable on the
exact-name metric by construction. They surface only in the value-level column —
which is precisely what happened (§5.2).

## 4. Value-stage metrics (paper metric: exact property + value within 2 %)

Gold v3, n = 41.

| Config | Pred. | TP | FP | FN | P | R | F1 | Halluc. |
|---|---|---|---|---|---|---|---|---|
| Frozen pipeline (exp11) | 16 | 14 | 2 | 27 | 0.875 | 0.342 | 0.491 | 0 |
| Table stage alone | 40 | 23 | 17 | 18 | 0.575 | 0.561 | 0.568 | 0 |
| **Merged (frozen ∪ table stage)** | 44 | 26 | 18 | 15 | **0.591** | **0.634** | **0.612** | **0** |

**Δ vs the frozen pipeline: recall +0.293, F1 +0.121, precision −0.284.**
Still **0 hallucinations** — every predicted number appears in the frozen source
text, and the Validator's safety property survives the new extraction context.

The merge drops 12 table-stage claims that restate a frozen one. That the
row-level stage independently re-derived 12 of experiment 11's 16 claims is a
useful consistency signal: the two contexts agree where they overlap.

## 5. Recall on the two experiment-11 error buckets

Both buckets are taken from `exp11/analysis.json`'s own catalog by exact FN-line
prefix, so this experiment cannot quietly redefine what counted as a miss.

### 5.1 The 17 table-grid misses — the target of this pilot

| View | Recall |
|---|---|
| **Value level** (property-agnostic, unit-class-guarded, grid units only) | **17/17 = 1.000** |
| **Exact name** (the paper metric) | **10/17 = 0.588** |

**Every grid cell was read correctly.** Row × column structure, cell values and
page attribution are perfect across all four grids, including the page-split
§7.6. The remaining gap is entirely a **naming** gap, and it is confined to two
grids:

| Grid | Family the model chose | Correct family | Cost |
|---|---|---|---|
| §7.6 discharge-temperature | `rel_discharge_capacity` | ✔ | — |
| §7.9 discharge-rate | `rel_discharge_capacity` | ✔ | — |
| §7.7 charge-temperature | `rel_discharge_capacity` | `rel_charge_capacity` | 5 FN + 5 FP |
| §7.8 charge-rate | `rel_charge_capacity`, `axis_kind=current_a` | `rel_capacity`, `axis_kind=charge_mode` | 2 FN + 2 FP |

§7.7 is a defensible reading of an ambiguous instruction rather than a
misreading of the table: the grid *measures* capacity on discharge (10 A, 2.5 V
cut-off) while *varying* the charge temperature, and the family gloss given to
the model — "capacity measured on DISCHARGE" vs "capacity resulting from the
CHARGE condition the column names" — does not cleanly separate those. The
resulting names collide with §7.6's, which is why they land as FPs with
identical names and values rather than as novel wrong names.

§7.8 is the harder failure: the model took the *charge-current* row (1.25 A / 4 A)
as the axis instead of the standard/rapid protocol distinction, producing
`rel_charge_capacity_1_25a_pct` and `rel_charge_capacity_4a_pct`. It read the
right values from the right row — it did not fall for the decoy `Cut-off | 125mA
| 100mA` row — but it labelled the axis by its current rather than its protocol.

Two of the three risks flagged before the run did **not** materialise: the model
never emitted a spurious claim for §7.7's phantom seventh column (`Discharge
temperature 25℃`, a fixed condition rather than a data column), and it resolved
the §7.9 family ambiguity correctly.

### 5.2 The 9 missed specification rows — secondary arm

| View | Recall |
|---|---|
| Value level (spec-row units only) | 5/9 = 0.556 |
| Exact name (paper metric) | 2/9 = 0.222 |

Only **6 of these 9 are reachable at all**: `rated_capacity_ah` (§7.4),
`ex_factory_soc_pct` and `ex_factory_ocv_range_v` (§7.12) are stated in prose,
not in the page-3 specification table, and no table-grounding pass can reach
them. Against the reachable 6 the value-level recall is 4/6.

| Gold claim | Outcome |
|---|---|
| `storage_temp_range_c = [-30, 45]` | **recovered, exact name** — the §2.2 line-break fix |
| `storage_temp_range_c = [-30, 60]` | **recovered, exact name** — same |
| `cell_height_mm = 64.85` | value recovered, named `height_mm` — out-of-vocab by construction (§3) |
| `cell_diameter_mm = 18.33` | value recovered, named `diameter_mm` — same |
| `std_charge_time_min = 180` | **lost to a silent unit conversion** — emitted as `std_charge_time_h = 3` |
| `rapid_charge_time_min = 60` | same — emitted as `fast_charge_time_h = 1` |
| `rated_capacity_ah`, `ex_factory_soc_pct`, `ex_factory_ocv_range_v` | unreachable (prose, not the table) |

## 6. Precision: where the 18 false positives come from

| Source | Count |
|---|---|
| Inherited from the frozen pipeline (`charge_voltage_v`, `storage_capacity_remaining_pct`) | 2 |
| Grid family/axis errors (§7.7 ×5, §7.8 ×2) — duplicates of correctly-read cells | 7 |
| Spec-row over-extraction: cut-off currents emitted under four different names (`discharge_cutoff_v`, `std_charge_current_a`, `charge_cutoff_current_a` = 0.125 A, `std_discharge_current_a` = 0.1 A), `max_charge_current_a` = 4 A | 5 |
| Unit-converted charge times (`std_charge_time_h` = 3, `fast_charge_time_h` = 1) | 2 |
| Correct values under non-canonical names (`height_mm`, `diameter_mm`) | 2 |

So **7 of the 18 are one root cause** — the family choice on two grids — and
they are duplicates of cells the model read correctly. A further 2 are values it
read correctly under names the frozen vocabulary does not contain. The genuinely
new over-extraction is the 5 cut-off/limit currents: given a single isolated
specification row, the model emits every number in it as a claim, including the
CV cut-off currents that the gold does not treat as separate claims. Row-level
isolation buys structure and costs restraint — the row no longer competes with
the rest of the document for the model's attention, so everything in it looks
salient.

### 6.1 A Validator finding worth reporting

The Validator behaved exactly as in experiment 11 where it was tested before —
it rejected `std_charge_time_h = 180 min` as implausible against the `[0.05, 50]
h` bounds — and it was left untouched here.

But `std_charge_time_h = 3` and `fast_charge_time_h = 1`, the **silently
converted** forms of the same two claims, passed every stage. Value-in-source is
a number-presence test, so a converted value survives whenever the converted
number happens to appear somewhere in the document — and "3" and "1" certainly
do. The claim is then plausible by construction, because conversion is what put
it inside the bounds.

This is a real limitation of the deterministic Validator, surfaced by this
experiment and not previously visible: **the value-in-source gate cannot detect a
unit conversion whose result collides with an unrelated number in the document.**
It is stated here rather than fixed, because fixing it would mean changing the
frozen Validator.

## 7. What this means for the paper

1. **The identified fix is now a validated fix.** Row-level table grounding
   recovers 17/17 of the table-grid cells at value level and lifts document
   recall from 0.342 to 0.634. Reviewer 2's comment 2.5 is answered with a
   measurement, not a plan.
2. **Report the precision cost honestly.** 0.875 → 0.591. F1 still improves
   (0.491 → 0.612), and 0 hallucinations survive, but the pilot trades precision
   for recall and the paper should say so.
3. **The residual gap is naming, not reading.** 7 of the 17 grid cells are read
   correctly and named wrongly, from two family choices on two grids. That is a
   prompt/vocabulary problem with an obvious next step (constrain the family by
   the axis the grid varies), not evidence that table grounding fails.
4. **Page-boundary splits deserve a sentence in the extraction section.** §7.6
   was unrecoverable from flat text for a structural reason, not a modelling one.
5. **Row isolation reduces restraint.** The new over-extraction (5 cut-off
   currents) is the mirror image of the conditions second pass's eagerness in
   experiment 11 §2 — a smaller context makes every number in it look like a
   claim.

## Files

| File | Contents |
|---|---|
| `table_grid.py` | deterministic table extraction, page-boundary stitch, classification, extraction units, the grid namer — **no LLM** |
| `run_grounding.py` | 3 cached row-level runs + unchanged Validator + consensus (makes LLM calls) |
| `score.py` | scoring against the 41-claim gold + both error buckets (**no LLM path**) |
| `tables_extracted.json` | the deterministic stage's output: regions, stitch log, the 15 units verbatim |
| `predictions_table.json` | the 40 consensus + Validator claims of the table stage |
| `runs_raw.json` | the 3 raw runs, Validator reports, consensus-only set, token usage |
| `results.json` | all metrics, both buckets with per-claim matches, family choices, merge log, deltas |
| `tests/test_exp12_table_grid.py` | 15 unit tests for the deterministic stage (no PDF, no API) |

Raw LLM responses are cached under `outputs/extraction_raw/tables/`. Like the
other datasheet artifacts they quote a copyrighted document verbatim and are not
part of the public release.
