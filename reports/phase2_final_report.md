# Phase 2 — Final research report

> **HISTORICAL (Phase 1 / Phase 2).** Kept as a research record. Operating-point metrics here use the Phase-1 threshold (0.659) and the original validation events. Final results: [`final_project_quality_report.md`](final_project_quality_report.md) and `reports/final/*.json`.

*All numbers below are read from files produced by the pipeline (`reports/phase2/*`, `experiments/phase2/results.csv`); none were typed in by hand. Failure events are simulated on top of real transformer measurements (ETDataset).*

## 1. Original system (Phase 1)

GRU + temporal attention, 31,425 parameters, 65 causal stationary features, 12 h window, 2 h horizon. Reported test results were **reproduced exactly** from the saved artefacts: PR-AUC 0.326, ROC-AUC 0.845, precision 0.318, recall 0.487, F1 0.385, 0.115 false alarms per asset-day, median lead 120 min (`reports/phase2/reproduction.md`).

## 2. Problems discovered

1. **The validation→test gap was mostly sampling luck plus selection bias, not seasonal shift.** Re-simulating the failure events 8 times on the same real data, the unchanged model scores PR-AUC (detectable positives) 0.450 on validation vs 0.444 on test (gap +0.006 ± 0.039). The original validation events had been used for every Phase-1 decision: on fresh validation events the deployed model (seed 42) scores 0.398 and the configuration 0.360 ± 0.034 (3 seeds), versus 0.553 on the original validation events. (`reports/phase2_distribution_shift.md`)
2. **Misses are not only a sensor-signal problem.** Test recall by detectability class: {'A': '16/22', 'B': '7/9', 'C': '1/2', 'D': '0/7'}; several strong-signal (class A) thermal failures were missed. (`reports/detectability_analysis.md`)
3. **Raw scores were uncalibrated** (weighted-BCE training) yet shown as probabilities.
4. **No per-prediction uncertainty**, despite large seed variance.
5. **Weak cross-asset transfer** (one direction tested in Phase 1).

## 3. Experiments (3 seeds each, selection on fresh validation realisations)

### Ablation

| configuration                        | seeds    | val PR-AUC (fresh events)   | test PR-AUC   | test PR-AUC (re-sim)   | recall        | precision     | F1            | FA/day        | lead (min)   | params       | latency (ms)   |
|:-------------------------------------|:---------|:----------------------------|:--------------|:-----------------------|:--------------|:--------------|:--------------|:--------------|:-------------|:-------------|:---------------|
| Raw signals only                     | 3        | 0.190 ± 0.011               | 0.189 ± 0.026 | 0.157 ± 0.016          | 0.614 ± 0.059 | 0.027 ± 0.005 | 0.051 ± 0.009 | 1.104 ± 0.093 | 300 ± 65     | 20,289       | 0.43           |
| Temporal features (all engineered)   | 3        | 0.333 ± 0.025               | 0.291 ± 0.038 | 0.312 ± 0.062          | 0.465 ± 0.059 | 0.280 ± 0.010 | 0.348 ± 0.014 | 0.128 ± 0.028 | 135 ± 15     | 33,345       | 0.41           |
| Stationary features (Phase-1)        | 3        | 0.360 ± 0.034               | 0.324 ± 0.021 | 0.359 ± 0.013          | 0.489 ± 0.070 | 0.284 ± 0.066 | 0.352 ± 0.043 | 0.150 ± 0.096 | 150 ± 30     | 31,425       | 0.41           |
| Stationary, no calendar              | 3        | 0.342 ± 0.011               | 0.311 ± 0.009 | 0.337 ± 0.021          | 0.511 ± 0.027 | 0.198 ± 0.017 | 0.285 ± 0.021 | 0.259 ± 0.055 | 175 ± 17     | 30,657       | 0.41           |
| Robust (24 h-relative) features      | 3        | 0.345 ± 0.020               | 0.277 ± 0.019 | 0.336 ± 0.023          | 0.438 ± 0.031 | 0.283 ± 0.017 | 0.343 ± 0.003 | 0.128 ± 0.015 | 115 ± 9      | 31,617       | 0.41           |
| GRU (no attention)                   | 3        | 0.347 ± 0.012               | 0.303 ± 0.027 | 0.361 ± 0.026          | 0.461 ± 0.059 | 0.337 ± 0.049 | 0.385 ± 0.015 | 0.096 ± 0.038 | 140 ± 37     | 27,265       | 0.40           |
| GRU + attention (deployed)           | 3        | 0.360 ± 0.034               | 0.324 ± 0.021 | 0.359 ± 0.013          | 0.489 ± 0.070 | 0.284 ± 0.066 | 0.352 ± 0.043 | 0.150 ± 0.096 | 150 ± 30     | 31,425       | 0.41           |
| Compact GRU (h=32)                   | 3        | 0.342 ± 0.023               | 0.286 ± 0.008 | 0.350 ± 0.011          | 0.475 ± 0.046 | 0.313 ± 0.126 | 0.363 ± 0.069 | 0.125 ± 0.094 | 145 ± 35     | 9,537        | 0.33           |
| LSTM + attention                     | 3        | 0.310 ± 0.038               | 0.309 ± 0.054 | 0.322 ± 0.028          | 0.513 ± 0.041 | 0.202 ± 0.066 | 0.283 ± 0.056 | 0.287 ± 0.094 | 135 ± 15     | 39,809       | 0.40           |
| TCN                                  | 3        | 0.327 ± 0.054               | 0.337 ± 0.039 | 0.352 ± 0.032          | 0.542 ± 0.005 | 0.181 ± 0.069 | 0.266 ± 0.078 | 0.343 ± 0.151 | 150 ± 13     | 33,697       | 0.27           |
| Lightweight transformer              | 3        | 0.283 ± 0.041               | 0.267 ± 0.050 | 0.287 ± 0.020          | 0.569 ± 0.022 | 0.184 ± 0.054 | 0.276 ± 0.064 | 0.365 ± 0.096 | 130 ± 38     | 23,489       | 0.26           |
| Multi-horizon hazard head (2 h head) | 3        | 0.359 ± 0.013               | 0.296 ± 0.020 | 0.345 ± 0.017          | 0.447 ± 0.066 | 0.288 ± 0.094 | 0.338 ± 0.061 | 0.130 ± 0.064 | 118 ± 56     | 31,524       | 0.41           |
| Raw score (no calibration)           | deployed | unchanged (monotone map)    | unchanged     | unchanged              | unchanged     | unchanged     | unchanged     | unchanged     | unchanged    | Brier 0.0248 | ECE 0.0390     |
| + isotonic calibration               | deployed | unchanged (monotone map)    | unchanged     | unchanged              | unchanged     | unchanged     | unchanged     | unchanged     | unchanged    | Brier 0.0131 | ECE 0.0045     |

### Decision table (challenger vs incumbent, fresh-validation PR-AUC)

| group        | candidate        |   seeds |   mean_val_rs |   diff_vs_incumbent |   se_diff |   required_gain | replaces_incumbent   |
|:-------------|:-----------------|--------:|--------------:|--------------------:|----------:|----------------:|:---------------------|
| architecture | gru              |       3 |        0.3466 |             -0.0133 |    0.0209 |          0.0209 | False                |
| features     | robust           |       3 |        0.3447 |             -0.0152 |    0.023  |          0.023  | False                |
| architecture | gru_compact      |       3 |        0.3425 |             -0.0174 |    0.0238 |          0.0238 | False                |
| features     | stationary_nocal |       3 |        0.3424 |             -0.0175 |    0.0208 |          0.0208 | False                |
| features     | engineered       |       3 |        0.3331 |             -0.0268 |    0.0246 |          0.0246 | False                |
| architecture | tcn              |       3 |        0.3272 |             -0.0327 |    0.037  |          0.037  | False                |
| architecture | lstm_attention   |       3 |        0.3099 |             -0.05   |    0.0295 |          0.0295 | False                |
| architecture | transformer      |       3 |        0.2828 |             -0.0771 |    0.0308 |          0.0308 | False                |
| features     | raw              |       3 |        0.19   |             -0.17   |    0.0207 |          0.0207 | False                |

Rule: (declared before the full results were seen, see reports/phase2_audit.md §7):
a challenger replaces the incumbent only if its mean *fresh-realisation* validation PR-AUC
(val_rs, 3 seeds) exceeds the incumbent's by more than max(0.01, 1 standard error of the
difference).  Test metrics are never used for the decision.

**Outcome: no challenger met the rule — the Phase-1 GRU+attention model is kept.**

### Multi-horizon hazard head

| horizon   |   multi-horizon head |   dedicated single-horizon model |
|:----------|---------------------:|---------------------------------:|
| 30 min    |                0.142 |                            0.163 |
| 60 min    |                0.24  |                            0.255 |
| 120 min   |                0.359 |                            0.36  |
| 240 min   |                0.434 |                            0.409 |

Fresh-validation PR-AUC. Rule: adopt only if the 2 h head is within 0.01 and every head within 0.02 of a dedicated model (fresh validation). **Not adopted.**

## 4. Cross-asset generalisation (test period of the target asset)

| setting                           | features   | target   |   seeds |   pr_auc |   pr_auc_std |   recall |   precision |    f1 |   fa_day |   lead_min |   event_recall |   pr_auc_resim |
|:----------------------------------|:-----------|:---------|--------:|---------:|-------------:|---------:|------------:|------:|---------:|-----------:|---------------:|---------------:|
| robust:ETTm2->ETTm1               | robust     | ETTm1    |       3 |    0.163 |        0.007 |    0.242 |       0.261 | 0.251 |    0.061 |      105   |          0.283 |          0.128 |
| robust:in-domain(both)->ETTm1     | robust     | ETTm1    |       3 |    0.296 |        0.03  |    0.483 |       0.286 | 0.359 |    0.108 |      170   |          0.567 |        nan     |
| stationary:ETTm2->ETTm1           | stationary | ETTm1    |       3 |    0.242 |        0.059 |    0.402 |       0.208 | 0.269 |    0.14  |      117.5 |          0.533 |          0.185 |
| stationary:in-domain(both)->ETTm1 | stationary | ETTm1    |       3 |    0.336 |        0.031 |    0.517 |       0.309 | 0.382 |    0.134 |      160   |          0.6   |        nan     |
| robust:ETTm1->ETTm2               | robust     | ETTm2    |       3 |    0.086 |        0.019 |    0.562 |       0.066 | 0.117 |    0.642 |      170   |          0.846 |          0.124 |
| robust:in-domain(both)->ETTm2     | robust     | ETTm2    |       3 |    0.281 |        0.022 |    0.392 |       0.28  | 0.324 |    0.148 |       85   |          0.795 |        nan     |
| stationary:ETTm1->ETTm2           | stationary | ETTm2    |       3 |    0.133 |        0.014 |    0.54  |       0.089 | 0.152 |    0.488 |      177.5 |          0.872 |          0.167 |
| stationary:in-domain(both)->ETTm2 | stationary | ETTm2    |       3 |    0.327 |        0.021 |    0.46  |       0.265 | 0.325 |    0.167 |      150   |          0.821 |        nan     |

`in-domain(both)` = the same architecture trained on both assets, scored on the target asset only. Note: the target asset's scaler is fitted on its own (unlabelled) training-period history.

### Which features generalise?

Cross-asset PSI of each (per-asset scaled) feature between the two transformers' training periods; PSI > 0.25 = the feature behaves differently on the other asset. Importance = drop in test PR-AUC when shuffled (deployed model, Phase-1 explainability).

**Most asset-specific features:**

| feature      |   cross_asset_psi |   importance_pr_auc_drop |
|:-------------|------------------:|-------------------------:|
| LULL_rstd4   |             4.008 |                  nan     |
| LUFL_rstd4   |             2.964 |                  nan     |
| LULL_rstd16  |             2.381 |                  nan     |
| LUFL_diff1   |             2.36  |                  nan     |
| LULL_range16 |             2.311 |                  nan     |
| LULL_diff4   |             2.148 |                  nan     |
| MUFL_rstd16  |             1.989 |                  nan     |
| OT_diff1     |             1.773 |                    0.026 |

**Most important features and their transferability:**

| feature          |   cross_asset_psi |   importance_pr_auc_drop |
|:-----------------|------------------:|-------------------------:|
| HUFL_rstd4       |             0.112 |                    0.068 |
| hour_cos         |             0     |                    0.057 |
| OT_resid         |             0.054 |                    0.039 |
| OT_volratio      |             0.065 |                    0.035 |
| OT_zbase         |             0.039 |                    0.03  |
| HULL_rstd4       |             0.161 |                    0.029 |
| OT_diff1         |             1.773 |                    0.026 |
| OT_resid_rmean16 |             0.054 |                    0.025 |
| MULL_rstd16      |             0.172 |                    0.02  |
| MULL_spikes16    |             0     |                    0.019 |

2 of the 15 most important features shift strongly between assets (PSI > 0.25); median PSI of all features 0.143.

## 5. Robustness to harder fault signatures (deployed model, no retraining)

| variant                                |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('pr_auc_detectable', 'mean') |   ('pr_auc_detectable', 'std') |   ('event_recall', 'mean') |   ('event_recall', 'std') |   ('lead_median_min', 'mean') |   ('lead_median_min', 'std') |   ('fa_day', 'mean') |   ('fa_day', 'std') |
|:---------------------------------------|---------------------:|--------------------:|--------------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------------------------:|-----------------------------:|---------------------:|--------------------:|
| standard (new seeds)                   |                0.39  |               0.034 |                           0.468 |                          0.043 |                      0.762 |                     0.028 |                        165    |                       58.481 |                0.104 |               0.02  |
| weaker signatures (amp ×0.5–1.0)       |                0.3   |               0.028 |                           0.339 |                          0.031 |                      0.537 |                     0.085 |                        136.25 |                       20.902 |                0.107 |               0.009 |
| delayed onset (≤ 50 % of precursor)    |                0.41  |               0.023 |                           0.467 |                          0.034 |                      0.781 |                     0.097 |                        158.75 |                       44.854 |                0.107 |               0.01  |
| intermittent signatures (30 % dropout) |                0.292 |               0.058 |                           0.334 |                          0.076 |                      0.593 |                     0.087 |                        103.75 |                       12.016 |                0.104 |               0.026 |
| all of the above + irregular growth    |                0.164 |               0.041 |                           0.186 |                          0.048 |                      0.357 |                     0.077 |                        107.5  |                       27.928 |                0.1   |               0.019 |

Each variant re-simulates the events (3 seeds) with the stated realism knobs; val and test periods pooled.

## 6. Calibration

Chosen by within-validation cross-fitting: **isotonic**.

| method      |   brier |    ece |
|:------------|--------:|-------:|
| identity    |  0.0248 | 0.039  |
| temperature |  0.025  | 0.033  |
| platt       |  0.0126 | 0.0044 |
| isotonic    |  0.0131 | 0.0045 |

Calibration is monotone: alarms, PR-AUC and lead times are unchanged; only the displayed probability changes.

## 7. Uncertainty (model confidence)

| method                    |   auroc_conf_vs_error_all |   auroc_conf_vs_false_alarm |   alarm_precision_high_conf |   alarm_precision_low_conf |
|:--------------------------|--------------------------:|----------------------------:|----------------------------:|---------------------------:|
| seed ensemble (3 members) |                     0.873 |                       0.733 |                       0.494 |                      0.143 |
| MC dropout (20 passes)    |                     0.882 |                       0.726 |                       0.451 |                      0.155 |

Confidence = 1 − 2·σ of three seed-ensemble members. Low-confidence alarms are much more often false — useful to triage alerts. It is model confidence, not physical certainty.

### Rule-based monitors before missed failures

In the 6 h before a failure, the backend's rule-based anomaly monitors fired before 100% of the *missed* failures that had a precursor and 62% of the warned ones — but also in 24% of random event-free 6 h windows (n = 402). A rule+model hybrid could recover misses but would add many false alarms; it was not validated and is not deployed.

## 8. False alarms

| split   | category                            |   episodes |
|:--------|:------------------------------------|-----------:|
| test    | load fluctuation (benign, injected) |          5 |
| test    | seasonal / real-data anomaly        |          9 |
| test    | thermal bump (benign, injected)     |          9 |
| test    | unknown                             |          1 |
| val     | load fluctuation (benign, injected) |          1 |
| val     | outage-like / post-outage recovery  |          1 |
| val     | seasonal / real-data anomaly        |          3 |
| val     | thermal bump (benign, injected)     |         10 |
| val     | unknown                             |          2 |

See `reports/false_alarm_analysis.md` for the base-rate check.

## 9. Efficiency of the operational engine

| stage        |   median ms |   p95 ms |
|:-------------|------------:|---------:|
| ingest       |      0.0073 |   0.0081 |
| features     |      8.5444 |   8.9075 |
| sequence     |      0.0006 |   0.0009 |
| inference    |      0.1324 |   0.1588 |
| confidence   |      1.3735 |   1.6308 |
| calibration  |      0.0808 |   0.099  |
| explain      |      3.9673 |   4.3901 |
| risk_engine  |      0.0437 |   0.0575 |
| alert_engine |      0.0122 |   0.0162 |
| total        |     14.2583 |  14.6094 |

Measured over 20904 engine steps; ONNX Runtime inference: True. Feature computation dominates; model inference is < 0.5 ms.

## 10. Final model

**GRU + temporal attention (Phase-1 weights), now served with ONNX Runtime, isotonic calibration, 3-seed confidence, integrated-gradients attribution.**

Test (original events): PR-AUC 0.326, recall 0.487, precision 0.318, F1 0.385, event recall (detectable) 0.727, median lead 120 min. Expected performance on fresh event realisations (3-seed mean): validation 0.360, test 0.359.

## 11. Negative results (kept on purpose)

* Adaptive 7-day robust normalisation (Phase 1) and 24 h-relative 'robust' features (Phase 2) did not improve fresh-validation PR-AUC.
* Removing calendar features did not help.
* No alternative architecture (GRU, compact GRU, LSTM+attention, TCN, lightweight transformer) beat the incumbent by the required margin.
* int8 dynamic quantisation is ~3× smaller but slower on this ARM CPU.
* Harder, intermittent or weaker fault signatures sharply reduce detection (section 5).

## 12. Limitations

* Failures are simulated; absolute metrics describe the simulated signatures, not real failure rates.
* Two assets only; cross-asset transfer is weak.
* 40 test failures → wide confidence intervals; seed variance ≈ ±0.03 PR-AUC.
* No voltage/current/frequency/weather channels exist in the data; none were fabricated.
* Health score is a derived application indicator; attributions are associational, not causal.
