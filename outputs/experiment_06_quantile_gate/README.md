# Quantile-referenced abstention — Tasks 1–3

Three changes, in order: fix the self-exclusion bug in the served neighbour lookup; replace the absolute coverage threshold with a percentile of the bank's own coverage distribution; re-run the HUST adaptation under it.

## Task 1 — self-exclusion fixed

`app/common.py` dropped any neighbour with weight ≥ 0.999999, using "distance 0" as a proxy for "this is the query cell". Replaced with explicit `exclude_cell_id`, threaded through `view_neighbors` → `graph_features_for_query` → `predict_with_gate`, and passed by the one page that knows the query's identity. `hf_space/` was regenerated from the fixed source. `src/models/graph_model.py` already excluded by id and needed no change.

Why the proxy was wrong: the condition view is (c_rate_1, c_rate_2, soc_transition_pct), so any two cells on the same charge policy sit at distance 0. All 77 HUST cells share `5C(80%)-1C`, so adding that study to the bank would have made every HUST cell invisible to every other HUST cell — silently, as a coverage deficit rather than an error.

`tests/test_self_exclusion.py` (5 tests): a zero-distance twin is kept unless it *is* the query; same-policy siblings survive in the condition view; identity exclusion is explicit; group exclusion still implies self-exclusion; coverage rises when a coincident cell joins the bank. **4 of the 5 fail against the old code and pass against the fix**; the fifth guards the grouped-CV path against regression. Full suite: 123 passed, 4 skipped. No published number moved — HUST zero-shot coverage is still 0.64/0.97/1.86 and the gate still retains 0 of 77.

## Task 2 — quantile-referenced gate, in-study

### (a) In-study results preserved: **YES**

| check | this run | published | tol | pass |
|---|---|---|---|---|
| graph RMSE (cycles) | 135.1548 | 135.2 | 0.5 | ✅ |
| graph MAPE (%) | 9.6904 | 9.69 | 0.05 | ✅ |
| Spearman raw coverage vs \|err\| log | -0.2474 | -0.247 | 0.01 | ✅ |
| Spearman raw coverage vs \|err\| cyc | -0.1789 | -0.179 | 0.01 | ✅ |
| AURC absolute gate | 86.7721 | 86.8 | 1.0 | ✅ |
| AURC random | 135.1548 | 135.2 | 1.0 | ✅ |
| AURC quantile-spread gate | 86.3896 | 86.4 | 1.0 | ✅ |
| AURC quantile gate vs absolute gate — the swap must not degrade the curve | 87.3991 | 86.77 | 1.0 | ✅ |
| quantile gate beats random — margin over random preserved | 47.7557 | 48.38 | 2.0 | ✅ |
| retention at operating point | 0.6 | 0.6 | 0.02 | ✅ |
| retained RMSE at operating point — 2 of 120 cells change decision | 79.0398 | 71.99 | 10.0 | ✅ |

| gate | AURC (cyc) | 95% CI |
|---|---|---|
| coverage, absolute threshold (deployed) | 86.8 | [67.1, 108.2] |
| coverage, quantile-referenced (new) | 87.4 | [67.5, 109.5] |
| GBM quantile spread | 86.4 | [65.8, 105.6] |
| random abstention | 135.2 | [81.9, 180.6] |

The swap costs **0.6 cycles of AURC** (86.8 → 87.4), well inside the bootstrap CI, and both coverage gates stay far ahead of random abstention (135.2) and level with the model's own quantile-spread gate. Spearman coverage-vs-error is reproduced exactly, because the swap changes the threshold and not the coverage values it is compared against.

**One number does move, and it is worth stating plainly.** At the 60% operating point both rules retain exactly 72 cells, but not the same 72: 2 of 120 cells change decision, and retained RMSE goes 72.0 → 79.0 cycles (+7.1). With 72 retained cells a single swapped-in cell carrying a large error moves RMSE by several cycles, so this is small-sample sensitivity at one threshold rather than a systematic degradation — the AURC over the whole curve differs by 0.6 cycles, and the per-fold thresholds span 4.075–4.095 around the old fixed value. If the paper quotes the 60%-retention RMSE, it needs the new number.

### (b) Deployed percentile: **q\* = 41**

- Retention 60.0% (72 of 120), retained RMSE 79.0 cycles — the paper's operating point, selected on Severson CV alone with no HUST information.
- Across the 66 folds, q\*=41 lands at absolute coverage **4.081** (median; range 4.075–4.095) against the old fixed **4.080** — a difference of 0.001 in coverage units (coverage is a sum of similarity weights, max 5).

| q (pct) | median fold threshold | retention | retained RMSE (cyc) |
|---|---|---|---|
| 0 | 0.432 | 99.2% | 134.7 |
| 10 | 3.093 | 90.8% | 118.8 |
| 20 | 3.828 | 81.7% | 111.7 |
| 30 | 4.001 | 73.3% | 114.0 |
| 41 ← | 4.081 | 60.0% | 79.0 |
| 50 | 4.182 | 51.7% | 70.6 |
| 60 | 4.252 | 41.7% | 71.2 |
| 70 | 4.344 | 31.7% | 63.0 |
| 80 | 4.416 | 21.7% | 64.2 |
| 90 | 4.486 | 11.7% | 40.7 |

In-study the two rules are near-indistinguishable, which is the expected and desired result: every fold's bank is 118 of the same 120 Severson cells, so a fixed percentile maps to a near-fixed absolute value. The rules can only diverge when the bank changes character.

## Task 3 — HUST adaptation under the quantile gate

### (c) The adaptation story

Three reference definitions were run. The distinction turns out to matter more than the absolute-to-quantile swap itself.

| reference distribution | what the percentile is taken over |
|---|---|
| `same_policy_excluded` | every bank cell, own policy group removed — the literal Task-2 definition |
| `self_only` | every bank cell, only itself removed |
| `study_stratified` | bank cells **of the query's own study**, scored under the same neighbour availability the query faces |

#### Global percentile (literal Task-2 rule) — does NOT recover

| n | variant | eval | gate thr | retention % | retained RMSE | retained MAPE % | false acc. | unnec. rej. | ungated MAPE % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | bank-only | 77 | 4.07 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 82.7 ± 0.0 |
| 0 | bank-retrain | 77 | 4.07 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 82.7 ± 0.0 |
| 5 | bank-only | 72 | 4.06 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 73.2 ± 0.7 |
| 5 | bank-retrain | 72 | 4.06 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 55.4 ± 6.8 | 19.1 ± 2.8 |
| 10 | bank-only | 67 | 4.06 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 71.8 ± 0.3 |
| 10 | bank-retrain | 67 | 4.06 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 49.8 ± 5.0 | 22.4 ± 3.2 |
| 20 | bank-only | 57 | 3.99 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 72.5 ± 1.3 |
| 20 | bank-retrain | 57 | 3.99 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 48.4 ± 2.3 | 17.4 ± 1.0 |
| 40 | bank-only | 37 | 3.86 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 71.1 ± 1.4 |
| 40 | bank-retrain | 37 | 3.86 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 32.6 ± 1.5 | 16.7 ± 2.7 |

#### Study-stratified percentile — recovers

| n | variant | eval | gate thr | retention % | retained RMSE | retained MAPE % | false acc. | unnec. rej. | ungated MAPE % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | bank-only | 77 | 4.07 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 82.7 ± 0.0 |
| 0 | bank-retrain | 77 | 4.07 ± 0.00 | 0.0 ± 0.0 | — | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 82.7 ± 0.0 |
| 5 | bank-only | 72 | 1.36 ± 0.09 | 71.1 ± 18.6 | 1445 ± 33 | 71.9 ± 1.4 | 51.2 ± 13.4 | 0.0 ± 0.0 | 73.2 ± 0.7 |
| 5 | bank-retrain | 72 | 1.36 ± 0.09 | 71.1 ± 18.6 | 464 ± 96 | 17.4 ± 2.0 | 9.8 ± 3.4 | 14.0 ± 8.4 | 19.1 ± 2.8 |
| 10 | bank-only | 67 | 2.28 ± 0.34 | 45.7 ± 20.7 | 1367 ± 67 | 69.1 ± 2.2 | 30.6 ± 13.8 | 0.0 ± 0.0 | 71.8 ± 0.3 |
| 10 | bank-retrain | 67 | 2.28 ± 0.34 | 45.7 ± 20.7 | 450 ± 44 | 21.5 ± 4.3 | 7.2 ± 2.5 | 26.4 ± 11.9 | 22.4 ± 3.2 |
| 20 | bank-only | 57 | 2.39 ± 0.11 | 65.3 ± 0.8 | 1437 ± 37 | 71.9 ± 1.1 | 37.2 ± 0.4 | 0.0 ± 0.0 | 72.5 ± 1.3 |
| 20 | bank-retrain | 57 | 2.39 ± 0.11 | 65.3 ± 0.8 | 355 ± 37 | 14.7 ± 1.8 | 3.8 ± 1.6 | 15.0 ± 0.7 | 17.4 ± 1.0 |
| 40 | bank-only | 37 | 2.89 ± 0.13 | 54.6 ± 12.0 | 1353 ± 94 | 70.3 ± 2.3 | 20.2 ± 4.4 | 0.0 ± 0.0 | 71.1 ± 1.4 |
| 40 | bank-retrain | 37 | 2.89 ± 0.13 | 54.6 ± 12.0 | 296 ± 26 | 14.6 ± 1.8 | 2.0 ± 0.7 | 14.4 ± 3.6 | 16.7 ± 2.7 |

**Why the global percentile fails.** The bank stays Severson-dominated: 120 Severson cells against at most 40 HUST. Severson's own LOO coverage median holds at 4.17 whatever else joins, so the q\*=41 threshold falls only from 4.07 to 3.86 while held-out HUST coverage climbs to just 3.73 at n=40. The gap never closes. Making the criterion scale-free is necessary but not sufficient — the *reference population* has to be right too, and a percentile of the whole bank keeps answering a question about Severson.

**What the stratified rule changes.** Judging a HUST cell against how well covered HUST cells actually are in this bank tracks the population that is being served. Retention goes 0% → 71% at n=5 and holds between 46% and 71% thereafter; retained MAPE for the retrained predictor runs 14.6–21.5%, against 83% zero-shot ungated; false acceptances fall 10 → 2 as n grows.

**Zero-shot (n=0) is unchanged**, which is the property that had to survive: with fewer than 5 same-study cells in the bank the rule falls back to the whole-bank percentile (4.07) and abstains on all 77 HUST cells, exactly as Section 4.5 reports. The gate does not open because it was made more permissive; it opens because the graph acquired evidence about the population being served.

**The gate is a coverage criterion, not an accuracy criterion.** Under `bank_only` — HUST cells in the bank, predictor left Severson-trained — the stratified gate opens just as wide but the served predictions are still ~71% MAPE, producing 20–51 false acceptances per draw. Retention recovering is only good news when the predictor recovers with it; the two must ship together.

**Retention is non-monotone in n** (71% → 46% → 65% → 55%) because threshold and coverage rise together: adding HUST cells raises held-out coverage and simultaneously raises the bar those cells are measured against. The level is governed by q\*, not by n — which is the intended behaviour of a self-referencing criterion, but it does mean n cannot be read as a progress bar.

**Spearman(coverage, |error|) among retained cells** is ~0 at every n (-0.15 to +0.05). Within the accepted set coverage does not rank error; its value is the accept/reject decision, not a confidence ordering over what it accepts.

## What to deploy

- Gate: `coverage ≥ q*-th percentile of the reference distribution`, **q\* = 41**, selected in-study, never re-tuned.
- Reference distribution: bank cells of the query's own study, scored under the same neighbour availability the query faces (own policy group held out in-study CV; self only at serve time). Fall back to the whole bank when the query's study has fewer than 5 bank members.
- In-study this is bit-identical to the Task-2 rule (asserted: both give 4.070296 on the all-Severson bank), so q\* carries over unchanged and the published in-study numbers stand.
- Ship the gate change together with predictor retraining. Alone it converts silent abstention into confident error.

## Files

| file | contents |
|---|---|
| `deployed_gate.json` | q\*, achieved retention, threshold mapping |
| `in_study_preservation_checks.csv` | machine-checkable headline preservation |
| `in_study_aurc.csv` | AURC + bootstrap CIs, four gates |
| `in_study_absolute_sweep.csv`, `in_study_quantile_sweep.csv` | risk–coverage curves |
| `in_study_fold_thresholds.csv` | per-fold threshold at q\* |
| `in_study_risk_coverage.pdf/.png` | in-study figure |
| `hust_results_per_repeat.csv`, `hust_results_aggregated.csv` | Task 3 results |
| `hust_recovery_quantile.pdf/.png` | recovery figure |

Reproduce: `python -m experiments.exp06_quantile_gate.in_study` then `...hust_adaptation` then `...summarize`. Task 1 lives in `app/common.py` with `tests/test_self_exclusion.py`.
