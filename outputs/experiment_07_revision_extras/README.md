# Revision extras — adaptation table, cost-sensitive gate, target sensitivity

Three follow-ups for the major revision. Nothing here modifies a published experiment output; the two LaTeX tables are new files in `paper/tables/`.

## Task 1 — adaptation table

`paper/tables/tab_adaptation.tex`, generated from the exp06 aggregate (study-stratified quantile gate, `bank+retrain` variant, mean ± std over 5 seeded repeats, including the n=0 all-abstain row). Booktabs/`tabularx` styling matches the existing tables; the caption states the gate, the transferred percentile, and the repeat count.

## Task 2 — cost-sensitive operating point

Cost model, both terms dimensionless fractions on [0, 1]:

```
cost(q) = r * frac_abstained(q) + errshare_retained(q),   r = c_abst / c_err
```

`errshare_retained` is the share of the population's total absolute error still incurred because those cells were served. The normalisation makes the endpoints readable: **serving everything costs exactly 1, abstaining on everything costs exactly r**, so r = 1 is where a blanket answer and a blanket refusal break even, and r is literally "how many average-error predictions is one abstention worth".

### The gate does concentrate error

| q | retention | share of total error served | ratio |
|---|---|---|---|
| 0 | 99.2% | 98.1% | 0.99 |
| 20 | 81.7% | 68.0% | 0.83 |
| 41 | 60.0% | 44.5% | 0.74 |
| 60 | 41.7% | 28.6% | 0.69 |
| 80 | 21.7% | 13.6% | 0.63 |
| 99 | 1.7% | 0.6% | 0.35 |

At the deployed q\*=41 the gate abstains on 40% of cells and removes 56% of the population's total absolute error. The ratio column is below 1 everywhere, so abstention is always removing more error than cells — the gate is doing real work at every operating point.

### Optimal operating point vs cost ratio

| r | optimal q | retention | RMSE retained (cyc) | MAPE retained (%) |
|---|---|---|---|---|
| 0.01 | 99 | 1.7% | 28.5 | 4.63 |
| 0.10 | 99 | 1.7% | 28.5 | 4.63 |
| 0.50 | 88 | 14.2% | 39.7 | 4.35 |
| 0.79 | 48 | 53.3% | 69.5 | 7.03 |
| 1.00 | 33 | 70.8% | 77.3 | 7.46 |
| 1.58 | 14 | 85.0% | 110.1 | 7.77 |
| 3.16 | 0 | 99.2% | 134.7 | 8.75 |
| 10.00 | 0 | 99.2% | 134.7 | 8.75 |
| 100.00 | 0 | 99.2% | 134.7 | 8.75 |

### Interpretation

Because the error-share curve is close to (though everywhere below) the retention curve, total cost is nearly linear in q and the optimum is essentially bang-bang: abstain on almost everything while abstentions are cheap, answer everything once they are expensive, with a narrow transition in between. The optimum leaves q≈94 at r≈0.50, passes q=48 at r≈0.77 and q=33 at r≈0.90, and reaches q=0 by r≈2.46.

**The deployed q\*=41 corresponds to an implicit cost ratio of roughly r ≈ 0.8–1.0.** It is never exactly cost-optimal on the integer-q grid — the optimum jumps over it — but it costs at most 5% above the best available q for r ∈ [0.77, 1.02], with a minimum excess of 3.5% at r = 0.90. In plain terms the paper's operating point implicitly prices one abstention at about the cost of one average-error prediction — a defensible default, and one worth stating rather than leaving implied. Outside that band the fixed setting gets expensive quickly, which is the honest caveat: an application that can tolerate wrong answers cheaply, or one that cannot tolerate them at all, should move q.

Figure: `cost_sensitive.pdf/.png`. Data: `cost_curve_by_q.csv`, `cost_optimal_by_ratio.csv`, `cost_excess_of_deployed_q.csv`.

## Task 3 — sensitivity of the cycle-life target

Current configuration (`src/ingestion/qc.py`): rolling-median smoothing window **5** (centered, `min_periods=1`), threshold-proximity tolerance **1%**, **linear** interpolation of the crossing. The re-implementation used here is asserted to reproduce `cycle_life_table()` exactly for all 124 cells before any variant is run.

| variant | labelled | changed | med \|Δ\| | max \|Δ\| | base RMSE | graph RMSE | Spearman ρ | AURC |
|---|---|---|---|---|---|---|---|---|
| paper configuration | 124 | — | — | — | 141.3 | 135.2 | -0.247 | 86.8 |
| smoothing window 1 | 124 | 5 | 0.0938 | 0.796 | 141.9 | 136.4 | -0.239 | 87.1 |
| smoothing window 3 | 124 | 1 | 1.5e-03 | 1.5e-03 | 141.3 | 135.2 | -0.247 | 86.8 |
| smoothing window 7 | 124 | 0 | — | — | 141.3 | 135.2 | -0.247 | 86.8 |
| smoothing window 9 | 124 | 0 | — | — | 141.3 | 135.2 | -0.247 | 86.8 |
| proximity rule 0.5% | 124 | 0 | — | — | 141.3 | 135.2 | -0.247 | 86.8 |
| proximity rule 2% | 124 | 0 | — | — | 141.3 | 135.2 | -0.247 | 86.8 |
| proximity rule none (censor) | 43 | 0 | — | — | 59.1 | 58.6 | -0.433 | 28.4 |
| nearest cycle | 124 | 43 | 0.236 | 0.474 | 142.0 | 135.5 | -0.254 | 86.3 |

### Verdict: the target is robust to all three choices

- **Smoothing window.** Widening it to 7 or 9 changes no cell at all; window 3 moves one cell by 0.002 cycles; removing smoothing entirely (window 1) moves 5 cells by at most 0.80 cycles. Largest headline movement across the whole axis: 0.5 cycles of baseline RMSE and 0.009 of Spearman ρ.
- **Interpolation.** Nearest-cycle instead of linear moves 43 cells — exactly the cells that have a genuine crossing — by a median of 0.24 and at most 0.47 cycles. These are sub-cycle differences; baseline RMSE moves +0.7 cycles and AURC -0.5.
- **Proximity tolerance.** Tightening to 0.5% or loosening to 2% changes **nothing at all** — not one cell, not one metric.

### The one choice that matters, and why it is not a robustness problem

Removing the proximity rule leaves only **43 of 124** cells with a label. That is not fragility in the estimator; it is a property of the dataset. The Severson cells are cycled until they reach 80% of nominal and then stopped, so the last recorded capacity sits essentially *on* the threshold and the smoothed curve of **81 of the 124 cells never dips strictly below it**. The proximity rule exists precisely to label those cells at their final cycle rather than mark them incomplete.

The apparent improvement in that row (baseline RMSE 59.1 vs 141.3) is **not** a better model: it is a different, shorter-lived subpopulation — the 43 cells that degrade fast enough to cross the threshold before the test ends. Metrics computed on it are not comparable to the paper's, and the row is included to document the dependency, not as an alternative configuration.

Section 3.1 already states the rule and the reason for it (cells terminated at the nominal threshold, final capacities clustering at 0.880–0.883 Ah). What this analysis adds is the count: **81 of 124 labels come from the fallback**, so for 65% of cells the effective definition is "the cycle at which the test was stopped, having reached 80% of nominal" rather than an interpolated crossing. That is worth one sentence in the appendix, because it also explains why the smoothing window and the interpolation rule are almost irrelevant here — they only act on the 43 cells that cross strictly.

Table: `paper/tables/tab_target_sensitivity.tex`. Data: `target_sensitivity.csv`.

## Task 3 (addition) — Kaplan-Meier under the initial-capacity EOL

Under the initial-capacity definition 46 of 124 cells reach threshold and 78 are right-censored at their last observed cycle. The product-limit estimate uses all 124: population survival falls to ~0.61 and then plateaus, so the median is not reached within the observation window, whereas discarding the censored cells and taking the median of the completers alone would report 493 cycles — an estimate that is wrong by construction, because the cells omitted are precisely the long-lived ones. That is the argument for the survival-aware treatment the reviewer asked about.

The batch-stratified curves separate with log-rank χ²=180.9 (df=2, p=5e-40), but **this is not evidence that the batches degrade differently** and should not be reported as such: censoring here is administrative and batch-dependent. The initial-capacity threshold sits below the nominal one (these cells start above nameplate), so registering an event requires cycling past the study's own stopping point — which batch 2 did (median final capacity 0.826 Ah, below its 0.858 Ah threshold; 43/43 events) and batches 1 and 3 did not (0.881 and 0.880 Ah against thresholds of 0.864 and 0.853; 3/41 and 0/40 events). The log-rank test is measuring the test schedule. Kaplan-Meier assumes censoring independent of the failure process, and across batches that assumption fails here.

Figure: `paper/figures/km_survival.pdf/.png` (copy in this directory). Data: `survival_data.csv`, `survival_km_all.csv`, `survival_km_batch{1,2,3}.csv`, `survival_summary.json`.

## Files

| file | contents |
|---|---|
| `paper/tables/tab_adaptation.tex` | Task 1 table |
| `paper/tables/tab_target_sensitivity.tex` | Task 3 table |
| `cost_curve_by_q.csv` | retention, error share, RMSE per percentile |
| `cost_optimal_by_ratio.csv` | cost-optimal q per cost ratio |
| `cost_excess_of_deployed_q.csv` | excess cost of q\* vs the optimum |
| `cost_sensitive.pdf/.png` | Task 2 figure |
| `target_sensitivity.csv` | Task 3 sensitivity grid |
| `survival_*.csv/.json` | Kaplan-Meier data + summary |
| `paper/figures/km_survival.pdf/.png` | Kaplan-Meier figure |

Reproduce: `python -m experiments.exp07_revision_extras.{tab_adaptation,cost_sensitive,target_sensitivity,survival,summarize}`.
