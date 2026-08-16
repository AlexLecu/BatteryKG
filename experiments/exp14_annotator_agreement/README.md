# Experiment 14 — second-annotator agreement on the claims gold standard

Agreement between the reference gold standard (**A1**) and an independent
second annotation of the same five datasheets (**A2**).

**Annotators:** A1 = first author (reference gold standard), A2 = second author.
Both annotation sets were produced independently by manual annotation against
`ANNOTATION_GUIDELINE.md`; A2 was blind to the reference annotations. Metrics
reported on the independent sets (pre-adjudication).

Purpose: quantify how far a second, independent annotation of the same five
datasheets agrees with the existing hand-labelled gold standard, and produce a
claim-by-claim disagreement list as the input for a human adjudication session.
This experiment **adjudicates nothing** and writes to neither annotation set.

## Inputs (read-only)

| role | path |
|---|---|
| reference set (62 claims) | `data/claims/*.yaml` |
| reference set, held-out (41 claims) | `data/gold/samsung_inr18650_25r_gold.yaml` |
| second set (164 claims) | `data/gold/second_annotator/*_annotation.yaml` |
| source PDFs (page-range check only) | `data/claims/datasheets/*.pdf` — not released |

The second annotator worked from `data/gold/second_annotator/ANNOTATION_GUIDELINE.md`
and its 27-property pre-Samsung vocabulary, blind to both reference files.

Both annotation sets and the guideline are released. The source PDFs are not
(copyright; URLs and retrieval dates in `data/README.md`) — `sanity.py`'s
page-range check is the only thing that reads them, and it skips silently when
they are absent.

## Run

```
python -m experiments.exp14_annotator_agreement.sanity   # step 1: schema + conventions
python -m experiments.exp14_annotator_agreement.score    # steps 2-5: agreement + adjudication list
```

Both accept an optional directory argument holding the five returned files, so a
different annotator's set scores with:

```
python -m experiments.exp14_annotator_agreement.score path/to/returned
```

No LLM path exists in this experiment; both scripts are pure file reads.

`adjudication_prep.py` additionally reads `data/claims/extracted_text/*.txt`,
which is **not released** (derived verbatim from the copyrighted datasheets).
Regenerate it first with `python -m src.agents.pdf_text` after downloading the
five datasheets listed in `data/README.md`; `sanity.py` and `score.py` — which
produce every number quoted in the paper — need no such input.

## Method

**Value test.** `src.agents.evaluation.values_match` — the same function the
paper metric uses: 2 % relative tolerance, unit-scaled (mAh↔Ah, mA↔A, mV↔V),
ranges compared element-wise after sorting.

**Two views**, each matching greedy 1:1 in reference-document order:

| view | matches when |
|---|---|
| exact | property names are identical **and** the values match |
| value-level | the values match **and** the units belong to the same physical class (`capacity`, `voltage`, `current`, `pct`, `temperature`, …) |

The value-level pass runs only over what the exact pass left unmatched, so it is
a strict superset. The unit-class guard is what stops `3.0 Ah` matching `3.0 V`
in a property-agnostic comparison.

**Tie-break.** Where several candidates match a value (routine inside a
characteristic grid, where many cells carry the same percentage), the partner
with the most similar *conditions* wins — shared parsed condition fields first,
then condition-text token overlap, then same page. This does not change any
count; it makes the pairing shown in the adjudication list the right cell.

**Metrics.** Neither set is truth, so agreement is reported as positive specific
agreement (F1 over the matched set, symmetric between annotators) with Jaccard
alongside, plus the two one-sided coverages, which separate "we contradict each
other" from "one of us recorded more".

**Strata** for the clustering question: `characteristic grid` (property is a
`rel_*` / relative / retention-at-temperature name), `read from graph` (notes or
conditions say so), otherwise `specification row`.

## Results — A1 vs A2

103 reference claims vs 164 second-set claims.

| view | matched | F1 | Jaccard |
|---|---:|---:|---:|
| exact property + value | 76 | 0.569 | 0.398 |
| value-level | 101 | 0.757 | 0.608 |

| direction | share |
|---|---:|
| reference claims with a value-level counterpart in the second set | **101/103 = 0.981** |
| second-set claims with a value-level counterpart in the reference | 101/164 = 0.616 |

Disagreement structure: 25 naming-only (22 of them the Samsung characteristic
grids, where the reference uses bespoke `rel_*_pct` names and the second
annotator used the shipped `capacity_retention_at_temp_pct` with the varying
temperature in the conditions), **1** genuine value disagreement, 1
reference-only claim, 62 second-set-only claims, and 7 matched pairs with a
conflicting condition field.

Chance-corrected agreement, on the two decisions where Cohen's κ is definable
(both annotators labelling the same, already matched item — the extraction step
itself has no shared item set and no negative class, so κ does not apply to it):

| decision | items | observed | κ |
|---|---:|---:|---:|
| which property name | 101 | 0.752 | **0.744** |
| property name, characteristic grids excluded | 80 | 0.900 | **0.895** |
| conditions stated vs `unspecified` | 101 | 0.950 | 0.0 (prevalence paradox — quote the 0.95) |

Krippendorff's α over the aligned item set (union of both annotations, `∅` for
the annotator who did not record a claim):

| α | items | observed | α |
|---|---:|---:|---:|
| property name incl. `∅` — what to extract *and* what to call it | 166 | 0.458 | **0.427** |
| naming only, matched items | 101 | 0.752 | **0.744** |
| presence only | 166 | 0.608 | −0.24 — degenerate, do not quote |

α_U (Krippendorff's *unitizing* α) is **not** reported: it is defined over a
continuum and needs each unit's offsets, which these annotations do not carry —
and the image-only Panasonic sheet has no text continuum at all. Recovering
spans by string-matching the quotations back into the extracted text would make
the figure a measurement of the string matcher. See the module docstring in
`alpha.py` for the full argument and the caveats that must travel with the
numbers above.

Full tables: `agreement_report.md` (§5b κ, §5c α, §6 adjudication list),
`results.json`, `sanity_report.md`. For the adjudication session itself,
`adjudication_prep.md` groups the 62 second-set-only claims into 7 categories
with gold precedent and source-text verification, so whole categories can be
resolved in one decision.

## Known limits

- **F1 is dominated by set-size asymmetry**, not by contradiction: the second
  annotator recorded ~60 % more claims (dimensions, charge cut-off currents,
  individual storage-duration rows, plot-derived values, test-criterion rows).
  Whether those belong in the gold standard is an adjudication question, not a
  scoring one.
- Matching is greedy in document order, not optimal assignment. With the
  condition tie-break it reproduces the sensible pairing on every grid checked,
  but it is not guaranteed globally optimal.
- The condition comparison compares parsed fields by exact string equality, so
  `"capacity >= 1,500mAh (60% of nominal)"` and `"capacity >= 1500 mAh, i.e.
  60% of nominal capacity"` count as conflicting. Read §6's conflict table
  before treating those as real disagreements.
- Omissions are compared as property-name sets; the reference set records
  omissions for only two documents, so that column is thin by construction.

## Files

| file | contents |
|---|---|
| `common.py` | document table, loading, unit classes, strata |
| `sanity.py` | step 1 — schema + convention check → `sanity_report.md`, `sanity.json` |
| `score.py` | steps 2-5 — agreement + adjudication list → `agreement_report.md`, `results.json` |
| `alpha.py` | Krippendorff's α (nominal, aligned item set); `python -m …alpha` prints per document |
| `adjudication_prep.py` | locates, categorises and verifies the second-set-only claims → `adjudication_prep.md` |
| `../../tests/test_exp14_alpha.py` | hand-computable α cases pinning the formula |
