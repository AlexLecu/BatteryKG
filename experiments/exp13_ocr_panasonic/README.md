# Experiment 13 — OCR pilot on the image-only Panasonic marketing sheet

> **Exploratory pilot. One document. Not integrated into the pipeline.**
> Reviewer 2 (round 2, comment 2.5) listed "row-level table grounding / OCR /
> vision" as the identified fixes. Grounding is validated in experiment 12; this
> measures the OCR branch. It is deliberately a measurement, not a pipeline
> change: nothing here is wired into `src/`, and the paper's Limitations 3 still
> stands as written unless the paper chooses to cite these numbers.

**Result in one line:** OCR makes the document extractable — **P = 1.000,
R = 0.643, F1 = 0.783, 0 hallucinations** on 14 gold claims — and the residual
errors are almost entirely **the extractor's**, not OCR's: 14/14 gold values are
legible in the OCR text, and the Validator caught **3/3** of the model's silent
unit-conversion errors.

## Setup

| | |
|---|---|
| Document | Panasonic NCR18650B marketing sheet (doc code 2G23X0KYKU), 4 pages, **no text layer** |
| Gold | `data/claims/panasonic_ncr18650b.yaml`, 14 claims, hand-annotated by eye 2026-07-11 |
| OCR | tesseract 5.5.2 + poppler 26.07.0, via `pytesseract` / `pdf2image` |
| Snapshot | `panasonic_ncr18650b_ocr.txt`, 3,181 chars, 4 page markers |
| Pipeline | **frozen and unmodified** — `extractor.SYSTEM_PROMPT`, 3 runs, temperature 0, seed 42, `validate_claims`, `consensus` |
| Cost | **3 LLM calls**, 4,449 prompt + 2,022 completion tokens |

Reproduce:

```
python -m experiments.exp13_ocr_panasonic.ocr                  # OCR, deterministic, no LLM
python -m experiments.exp13_ocr_panasonic.run_ocr_extraction   # 3 cached runs (LLM)
python -m experiments.exp13_ocr_panasonic.score                # re-score only, no LLM path
pytest tests/test_exp13_ocr.py                                 # no PDF, no API
```

### In the public release

`panasonic_ncr18650b_ocr.txt` is **not released**: it is the marketing sheet's
text reproduced verbatim, the same rule that withholds
`data/claims/extracted_text/*.txt`. `score.py` reads it, so on the public tree
it raises `FileNotFoundError` until you rebuild the snapshot:

```
# install the binaries first: brew install tesseract poppler
# download the datasheet listed in data/README.md, then:
python -m experiments.exp13_ocr_panasonic.ocr
```

The OCR stage is deterministic and its configuration is pinned in
`ocr_meta.json`, so the rebuilt snapshot reproduces the published numbers.
`results.json`, `predictions_final.json` and `runs_raw.json` all ship, and
`tests/test_exp13_ocr.py` needs neither the snapshot nor the PDF.

## 1. The document really has no text layer

All four pages: **0 characters, 0 word boxes, 0 char objects, one full-page
raster each**. The frozen pipeline's own snapshot of it,
`data/claims/extracted_text/panasonic_ncr18650b.txt`, is 66 bytes — four page
markers and nothing else:

```
'=== PAGE 1 ===\n\n\n=== PAGE 2 ===\n\n\n=== PAGE 3 ===\n\n\n=== PAGE 4 ===\n'
```

That file is **left untouched** by this experiment; it is the artifact that
documents the limitation. The OCR snapshot is written inside this folder
instead, in the same format (`pdf_text.PAGE_MARK` is imported, not re-declared)
so the frozen pipeline consumes it without modification.

## 2. The OCR stage, and two non-obvious findings

### 2.1 The raster is 144 dpi — rendering higher makes it worse

The embedded image is 1684×1190 for an 842×595 pt page: exactly 144 dpi.
Rendering at 300 or 400 dpi upscales a 144 dpi source, adding interpolation blur
rather than detail. Measured on the 15 page-1 target values: **300 dpi → 5/15,
400 dpi → worse, native 144 dpi → 9/15.**

### 2.2 Table rules were eating the value column

Tesseract's layout analysis merges short value cells into the ruled borders
around them and drops them: at default settings `Nominal Voltage` came back with
no value at all. Erasing horizontal/vertical ink runs longer than 60 px before
OCR is the single decisive step in the pipeline.

| Preprocessing | Confident tokens (conf ≥ 60) | Page-1 target values |
|---|---|---|
| No rule removal | 21–54 | 0–9 / 15 |
| **Rule removal** | **73–85** | **12–14 / 15** |

### 2.3 Configuration was chosen without reference to the gold

The config was selected by maximising **tokens returned with tesseract
confidence ≥ 60** over an 18-point sweep (rule removal off/60/100 × upscale
1/2/3× × psm 4/6) — a gold-independent criterion. The gold-hit count is reported
above only as a cross-check. The two rankings agree on the decision that
matters: every rule-removal variant beats every non-rule-removal variant on
both. Mean confidence *alone* was rejected as a criterion because it rewards
reading little and reading it confidently — the highest-mean-confidence variant
returned 23 tokens and 0 target values.

Final: native 144 dpi → rule removal (runs ≥ 60 px) → 3× LANCZOS →
`--oem 1 --psm 6 -c preserve_interword_spaces=1`.

## 3. OCR quality: numbers survive, units do not

12 of the 13 page-1 specification values came through verbatim. **One numeric
corruption:** `Charging Voltage 4.2V` → `Charaing Voltage — 42Vv`, the decimal
point lost. Unit strings fared much worse than numbers — `mAh`→`mAn`,
`Wh/l`→`Whil`, `Wh/kg`→`Whikg` — and one non-gold value was misread
(`65.10mm`→`69.10mm`). Chart pages survive as noise plus their annotation lines.

**The graph annotation was captured, twice and verbatim:**

```
page 2:  Discharge:CC:Variable Current (E.V.:2.50V)
page 3:  Discharge:CC:3.25A(E.V.:2.50V)
```

So `discharge_cutoff_v = 2.5`, the one claim the gold marks as read from a graph,
is **annotation text inside the figure, not a value estimated off a curve** — and
OCR reads it cleanly. It is therefore not unrecoverable-from-text in principle,
which is why it is reported as its own denominator rather than excluded.

### 3.1 Value-in-source runs against the OCR snapshot

The OCR text is the only "source" this document has, so `value_in_source` is
checked against it. An OCR error in a value can therefore cause a rejection, and
that is intended: such rejections are a finding about OCR quality.

Measured **before any LLM call**, using the Validator's own `value_in_source`:
**all 14/14 gold values are present in the OCR snapshot.** No gold claim is
blocked by that gate.

`charge_voltage_v = 4.2` passes despite the corrupted page-1 cell, because the
charge-protocol annotation `Charge:CC-CV:1.625A-4.2V` on pages 2, 3 and 4
supplies a correct `4.2` and the gate reads the whole document. Cross-page
redundancy rescued a corrupted table cell — and, as §5 shows, it rescued the
extraction too.

## 4. Results

| | Value |
|---|---|
| Gold claims | 14 |
| Predictions (consensus + Validator) | 9 |
| TP / FP / FN | 9 / 0 / 5 |
| **Precision** | **1.000** |
| **Recall** | **0.643** |
| **F1** | **0.783** |
| Hallucinations | **0** |

Recall by denominator:

| Denominator | Recall | |
|---|---|---|
| All gold claims | 9/14 = **0.643** | |
| **Specification table only** | **9/13 = 0.692** | the fair test of OCR on structured content |
| Graph annotation only | 0/1 = 0.000 | legible in the OCR text; the extractor never proposed it (§5) |

All three runs returned the same 12 claims and the Validator accepted the same 9
in each, so consensus changed nothing — the same near-zero disagreement at
temperature 0 already reported in experiment 11 §1.

## 5. Error taxonomy — all 5 misses

| Bucket | n | Claims |
|---|---|---|
| `wrong_value` | 3 | `rated_capacity_ah`, `nominal_capacity_ah`, `std_charge_current_a` |
| `not_extracted` | 2 | `minimum_capacity_ah`, `discharge_cutoff_v` |
| `ocr_value_lost` | **0** | — |
| `validator_rejected` | 0 | — |
| `consensus_dropped` | 0 | — |

**Not one miss is attributable to OCR.** The taxonomy's OCR bucket is empty.

### 5.1 The model converted units, and got the conversions wrong

| OCR text | Model emitted | Correct |
|---|---|---|
| `Min.3200mAn` / `Min.3250mAh` | `rated_capacity_ah = [0.0032, 0.00325] Ah` | 3.2 / 3.25 Ah |
| `Typ.3350mAh` | `nominal_capacity_ah = 0.00335 Ah` | 3.35 Ah |
| `Std. 1625mA` | `std_charge_current_a = 0.1625 A` | 1.625 A |

Every one is a botched mAh→Ah / mA→A conversion — off by 10³ in two cases and
10⁴ in the third — despite hard rule 1 of the frozen prompt ("Extract ONLY what
the document explicitly states. Never infer, convert…").

**The Validator caught all three**, by two different stages: `0.00335` and
`0.1625` by **value-in-source** (neither number appears in the OCR text), and
`[0.0032, 0.00325]` by **plausibility bounds** (`[0.1, 10] Ah`). That is why
precision is 1.000 with 0 hallucinations.

This is the direct counterpart of the Validator limitation surfaced in
experiment 12 §6.1, where `std_charge_time_h = 3` (a silent 180 min → 3 h
conversion) passed because the converted value happened to appear elsewhere in
the document. Here the converted values are absurd enough not to collide with
anything, so the same gate fires. **Value-in-source catches unit conversions
exactly when the converted number is not coincidentally present** — experiment
12 found the failure case, this one the success case.

### 5.2 Two claims never proposed

- **`minimum_capacity_ah = 3.25`** — not independently missed so much as
  *absorbed*: the model folded the two capacity rows (`Min.3200mAh` and
  `Min.3250mAh`) into a single range claim, `rated_capacity_ah = [0.0032,
  0.00325]`. Two distinct gold claims became one. The taxonomy labels it
  `not_extracted` because no run ever proposed the property, which is literally
  true; the cause is the merge.
- **`discharge_cutoff_v = 2.5`** — a genuine **extractor** miss, not an OCR
  miss. `E.V.:2.50V` is sitting legibly in the OCR text on two pages; the model
  simply never treated a chart annotation as a specification. This is the one
  place where the three-denominator split earns its keep: the graph subset
  scores 0/1, and the reason has nothing to do with the graph or with OCR.

### 5.3 A prediction that did not materialise

Before the run I predicted the model would read the corrupted `42Vv` cell and
emit `charge_voltage_v = 42`, to be rejected by the `[0, 5] V` plausibility
bounds. It did not: **`charge_voltage_v = 4.2` was extracted correctly** in all
three runs. The page-1 cell is corrupt, but pages 2–4 carry
`Charge:CC-CV:1.625A-4.2V` intact, and the model used the redundant statement.
The Validator never had to fire. Recorded because it is a real property of
multi-page marketing sheets — repeated protocol annotations give OCR errors
somewhere to be corrected from — and because the prediction was wrong.

## 6. What this means for the paper

1. **OCR moves this document from "not extractable" to "extractable with
   P = 1.000, R = 0.643".** Limitations 3 currently says the sheet "is outside
   the current text-based pipeline and requires OCR or a vision model". That
   remains true of the *pipeline*; this pilot shows what the OCR branch would
   buy if integrated.
2. **The recall gap is the extractor's, not OCR's.** 14/14 values legible,
   0 misses attributable to OCR. Three of the five misses are botched unit
   conversions the prompt already forbids; one is a claim merge; one is a chart
   annotation the extractor ignores.
3. **The Validator did real work here** — 3/3 unit-conversion errors caught,
   precision 1.000. In experiment 11 it improved no metric because there was
   nothing to reject. Together with experiment 12 §6.1 the three experiments now
   bracket the value-in-source gate's power precisely.
4. **Report it as exploratory.** One document, one OCR engine, no integration.
   Vision models remain the untested alternative — a vision model would be the
   natural way to attack §5.2's chart-annotation miss, since that claim is
   *visible* in the figure but not *tabular* anywhere.

## Files

| File | Contents |
|---|---|
| `ocr.py` | render → rule removal → upscale → tesseract; writes the frozen snapshot — **no LLM** |
| `run_ocr_extraction.py` | frozen pipeline, 3 cached runs, unchanged Validator + consensus (LLM) |
| `score.py` | three-denominator scoring + error taxonomy (**no LLM path**) |
| `panasonic_ncr18650b_ocr.txt` | the frozen OCR snapshot, 3,181 chars, 4 page markers |
| `ocr_meta.json` | tesseract/poppler versions, config, per-page token counts and confidences |
| `predictions_final.json` | the 9 consensus + Validator claims |
| `runs_raw.json` | 3 raw runs, Validator reports, usage |
| `results.json` | metrics, three denominators, error taxonomy, Validator reports |
| `tests/test_exp13_ocr.py` | rule removal + subset derivation (no PDF, no API) |

Raw LLM responses are cached as `outputs/extraction_raw/panasonic_ncr18650b_ocr_*`
— a distinct document key, so no existing artifact is read or written.
