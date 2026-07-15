# Experiment 01 — cycle-life prediction with graph features + coverage-gated abstention

**Target:** `log10(cycle_life_nominal)` (cycle life to 80 % of nominal 1.1 Ah).
**Protocol:** leave-one-policy-group-out CV, grouping key `policy_group_id` — 66 folds over 120 cells.
**Exclusions:** the 4 QC-anomalous cells ['b1c0', 'b1c18', 'b2c12', 'b2c44'] are removed from both train and test (gross capacity spikes; `is_anomalous` in the dataset). Two of them (b2c12, b2c44) were their policy group's only member, so the fold count is 66 rather than the nominal 68.
**Models:** identical XGBoost config (no tuning, seed 42). Baseline = early-cycle (cycles 10-100) + parsed policy features + batch. Graph = baseline + per-fold, train-neighbors-only KG features from both SIMILAR_TO views (weighted-mean/std neighbor log-life, mean edge weight, coverage_train; k=5).

## Overall metrics

| model | RMSE (log10) | RMSE (cycles) | MAPE (%) |
|---|---|---|---|
| baseline | 0.061 | 141.3 | 9.8 |
| graph | 0.061 | 135.2 | 9.7 |

## Per-batch metrics

| batch | n | baseline RMSE cyc | graph RMSE cyc | baseline MAPE % | graph MAPE % |
|---|---|---|---|---|---|
| 1 | 39 | 163.4 | 155.5 | 8.9 | 9.0 |
| 2 | 41 | 45.0 | 47.5 | 9.2 | 9.5 |
| 3 | 40 | 178.4 | 170.0 | 11.3 | 10.6 |

## Paired per-fold comparison (fold RMSE, log space)

- graph wins **38** folds, baseline wins **28**, ties 0 (of 66)
- Wilcoxon signed-rank on paired fold RMSEs: statistic=990.0, p=0.461

## Coverage-gated abstention (behavior view)

Risk-coverage sweep over `behavior_coverage_train` (sum of top-5 edge weights into the training fold). Random-abstention baseline = mean RMSE of 500 seeded random subsets at the same retention. See `risk_coverage.png`.

| threshold | retained | frac | RMSE retained (cyc) | RMSE random (cyc) |
|---|---|---|---|---|
| 0.432 | 120 | 1.00 | 135.2 | 135.2 |
| 3.025 | 110 | 0.92 | 128.2 | 135.0 |
| 3.670 | 100 | 0.83 | 111.0 | 134.9 |
| 3.930 | 90 | 0.75 | 112.8 | 134.1 |
| 4.055 | 80 | 0.67 | 77.3 | 134.4 |
| 4.088 | 70 | 0.58 | 72.7 | 133.8 |
| 4.182 | 60 | 0.50 | 70.1 | 132.7 |
| 4.237 | 50 | 0.42 | 71.2 | 133.2 |
| 4.310 | 40 | 0.33 | 62.2 | 131.7 |
| 4.380 | 30 | 0.25 | 64.2 | 127.9 |
| 4.458 | 20 | 0.17 | 52.9 | 126.5 |
| 4.492 | 10 | 0.08 | 32.7 | 119.2 |

## Do low-coverage cells make the largest errors?

Spearman rank correlations against `behavior_coverage_train` (n = 120):

- |error| in **log space** (the model's target): **rho = -0.247** (p = 0.00644)
- |error| in cycles: rho = -0.179 (p = 0.0505)
- actual cycle life (confound check): rho = 0.055 (p = 0.548) — coverage is NOT a proxy for lifetime magnitude, so the risk-coverage gain is not an artifact of preferring short-lived cells.

Lower coverage -> larger errors, as the abstention gate assumes.

### 10 largest graph-model errors

| cell | fold (policy group) | batch | abs err (cyc) | behavior cov_train | condition cov_train |
|---|---|---|---|---|---|
| b3c38 | pg050 | 3 | 774.3 | 3.966 | 3.297 |
| b1c2 | pg007 | 1 | 556.3 | 3.045 | 2.539 |
| b1c1 | pg007 | 1 | 430.8 | 2.149 | 2.539 |
| b1c3 | pg026 | 1 | 429.8 | 3.445 | 2.884 |
| b3c10 | pg018 | 3 | 277.7 | 4.079 | 1.173 |
| b1c4 | pg026 | 1 | 274.6 | 3.215 | 2.884 |
| b3c21 | pg009 | 3 | 273.5 | 2.168 | 3.091 |
| b3c22 | pg049 | 3 | 215.4 | 4.302 | 4.040 |
| b3c16 | pg018 | 3 | 197.2 | 2.935 | 1.173 |
| b3c6 | pg009 | 3 | 197.2 | 2.361 | 3.091 |

## Files

- `experiment_01_predictions.csv` — per-cell CV predictions, both models
- `pred_vs_actual.png`, `risk_coverage.png`

*Seeds fixed (42) in model, folds (deterministic group order), and random-abstention draws.*

## Robustness

Sensitivity of the abstention result. Same predictions as above (graph model, seed 42); only the gating signal varies. AURC = mean RMSE (cycles) across retention levels 20%-100%, lower is better; 95% CIs from 1000 fold-bootstrap resamples.

**Hybrid gate selection rule (no test-set tuning):** keep a cell iff behavior coverage >= t_cov AND quantile spread <= t_spread. For each target retention level and each CV fold, the threshold pair is selected on the OTHER folds' cells only (among pairs meeting the retention target there, minimise retained RMSE; ties -> higher retention), then applied to the held-out fold. Candidate thresholds are signal quantiles plus keep-all endpoints.

### AURC (area under risk-coverage)

| gate | AURC (cyc) | 95% CI |
|---|---|---|
| behavior k=3 | 86.9 | [68.0, 106.2] |
| behavior k=5 | 86.8 | [67.1, 108.2] |
| behavior k=10 | 85.3 | [64.5, 102.7] |
| condition k=5 | 118.6 | [69.2, 161.8] |
| mean(views) k=5 | 100.7 | [63.7, 134.3] |
| GBM quantile spread | 86.4 | [65.8, 105.6] |
| hybrid (coverage + spread) | 109.7 | [67.2, 152.7] |
| random abstention | 135.2 | [81.9, 180.6] |

### Spearman(gating signal, |error|)

| signal | rho (log-space err) | p | rho (cycles err) | p |
|---|---|---|---|---|
| behavior k=5 | -0.247 | 0.00644 | -0.179 | 0.0505 |
| condition k=5 | -0.098 | 0.289 | -0.187 | 0.0404 |
| mean(views) k=5 | -0.215 | 0.0184 | -0.239 | 0.00859 |
| GBM quantile spread (spread vs err; positive = informative) | 0.134 | 0.146 | 0.179 | 0.0503 |

### Plots

- `risk_coverage_k_sensitivity.png` — k in {3, 5, 10}, behavior view
- `risk_coverage_view_sensitivity.png` — behavior vs condition vs mean vs the GBM's own quantile-spread gate (all k=5)
