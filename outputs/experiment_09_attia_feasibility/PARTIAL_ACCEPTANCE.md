# Attia 2020 — partial acceptance under the deployed gate

The first external population the gate does not refuse outright: **17 of 45 cells retained** (37.8%) at threshold 4.0703. Deployed artifacts applied unchanged; nothing retrained, nothing added to the bank.

| configuration | n | RMSE (cyc) | MAPE (%) | FA | UR |
|---|---|---|---|---|---|
| gate-retained | 17 | 141.7 | 21.5 | 4 | — |
| gate-abstained (had it answered) | 28 | 197.8 | 19.1 | — | 21 |
| graph model, ungated | 45 | 178.7 | 20.0 | 11 | — |
| graph-free GBM, ungated | 45 | 168.2 | 18.8 | — | — |

FA = served with APE > 30%. UR = abstained though APE ≤ 30%.

## Does the gate select the cells it should?

Retained RMSE 141.7 vs abstained 197.8 (ungated 178.7); retained MAPE 21.5% vs abstained 19.1%.

**The two metrics disagree, and the disagreement is the finding.** The gate retains cells with lower absolute error but *higher* percentage error. That is what happens when the retained set is systematically shorter-lived: the same absolute error is a larger fraction of a smaller lifetime. Read with the lifetime-proxy result below, the RMSE advantage is substantially a lifetime effect rather than evidence that the gate found the predictable cells.

Spearman(coverage, |error|) on the retained set: **ρ = -0.127** (p = 0.626). In-study the same correlation is −0.247: more coverage, less error.
Across all 45: ρ = +0.002 (p = 0.989).

### Lifetime-proxy check

Spearman(coverage, cycle life) = **-0.643** (p = 1.86e-06), against +0.055 in-study (experiment 01). Coverage is behaving as a lifetime proxy on this population, which the in-study test explicitly ruled out for Severson — the retained subset is partly being selected for how long the cells live, not only for how predictable they are.

## Calibration of the q16–q84 band

| set | n | empirical coverage | nominal | median width (cyc) | model over-predicted | model under-predicted |
|---|---|---|---|---|---|---|
| retained | 17 | 6% | 68% | 94 | 16 | 0 |
| abstained | 28 | 32% | 68% | 278 | 19 | 0 |
| all | 45 | 22% | 68% | 211 | 35 | 0 |

The band is badly miscalibrated out of distribution — 22% empirical against a 68% nominal band — and the misses are **one-sided**: 35 of 45 cells fall below the band and 0 above it. The model does not merely have wide uncertainty here, it systematically predicts these cells to live longer than they do, and its own uncertainty band does not admit the possibility.

## Feature completeness caveat

8 of 19 model inputs are undefined for every Attia cell: `batch`, `c_rate_1`, `c_rate_2`, `condition_coverage_train`, `condition_nbr_mean_weight`, `condition_nbr_wmean_log_life`, `condition_nbr_wstd_log_life`, `soc_transition_pct`. policy grammar cannot express Attia's four-step CC protocols; the condition similarity view follows from those features. The 11 that remain are all 7 early-cycle features and all 4 behaviour-view graph features — the informative ones — but the error figures above are produced with 8 inputs absent, which is not the condition the in-study numbers were measured under.

## Artifacts

| file | contents |
|---|---|
| `attia_partial_acceptance.json` | full result record |
| `attia_partial_acceptance_per_cell.csv` | per-cell coverage, prediction, APE, band |
| `paper/tables/tab_attia_partial.tex` | paper table |

## Minimal adaptation (n=5)

Five labelled Attia cells added to the bank at random, predictor (point and q16/q84) refitted on Severson + 5, gate unchanged at q*=41, evaluated on the remaining 40. Mean ± std over 5 seeded repeats. With five same-study cells present the stratified rule now draws its reference from Attia itself (`same_study/self_only`) instead of the whole-bank fallback.

| | zero-shot (n=0, 45 cells) | adapted (n=5, 40 cells) |
|---|---|---|
| gate threshold | 4.070 | 3.864 ± 0.289 |
| retention % | 37.8 | 54.5 ± 19.6 |
| retained RMSE | 141.7 | 100.5 ± 24.2 |
| retained MAPE % | 21.5 | 11.1 ± 5.1 |
| false acceptances | 4 | 1.4 ± 1.5 |
| unnecessary rejections | 21 | 15.2 ± 5.8 |
| ungated RMSE (all evaluated) | 178.7 | 146.8 ± 20.0 |

### The two diagnostics after adaptation

- **Spearman(coverage, cycle life), all evaluated**: -0.611 ± 0.066 (reference: -0.643 zero-shot, +0.055 in-study)
- **Spearman(coverage, cycle life), retained only**: -0.498 ± 0.033 (reference: —)
- **Spearman(coverage, |error|), retained only**: -0.134 ± 0.193 (reference: -0.127 zero-shot, -0.247 in-study)

The lifetime-proxy confound **persists** after adaptation (-0.611 vs -0.643 zero-shot). Five cells are not enough to stop coverage tracking lifetime on this population; the retained subset is still being chosen partly for how long its cells live.

### Band calibration after retraining

| set | empirical coverage | nominal | model over-predicted (of 40) |
|---|---|---|---|
| retained | 47% ± 27% | 68% | — |
| all evaluated | 43% ± 22% | 68% | 17.8 ± 11.0 |

Zero-shot the band covered 6% of retained cells with all 35 misses on one side (over-prediction). 

Artifacts: `attia_adaptation.json`, `attia_adaptation_per_repeat.csv`, `paper/tables/tab_attia_adaptation.tex`.
