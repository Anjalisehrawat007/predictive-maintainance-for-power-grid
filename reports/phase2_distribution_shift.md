# Why is test worse than validation? — quantitative analysis

> **HISTORICAL (Phase 1 / Phase 2).** Kept as a research record. Operating-point metrics here use the Phase-1 threshold (0.659) and the original validation events. Final results: [`final_project_quality_report.md`](final_project_quality_report.md) and `reports/final/*.json`.

Model: `best_attention_model` (Phase-1 selection), threshold 0.659. Validation = 2017-11-22 → 2018-03-09 (winter), test = 2018-03-10 → 2018-06-26 (spring).

## 1. Headline gap

| | val | test |
|---|---|---|
| PR-AUC | 0.553 | 0.326 |
| event recall (detectable) | 0.889 | 0.727 |
| false alarms / asset-day | 0.082 | 0.115 |

## 2. Controlled experiment: is the gap the *period* or the *particular failures*?

The real sensor data were kept; the failure/benign events were **re-simulated 8 times** with new seeds (same simulator settings); the Phase-1 model and preprocessor were applied unchanged.

| split   |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('pr_auc_detectable', 'mean') |   ('pr_auc_detectable', 'std') |   ('recall', 'mean') |   ('recall', 'std') |   ('event_recall_detectable', 'mean') |   ('event_recall_detectable', 'std') |   ('fa_day', 'mean') |   ('fa_day', 'std') |
|:--------|---------------------:|--------------------:|--------------------------------:|-------------------------------:|---------------------:|--------------------:|--------------------------------------:|-------------------------------------:|---------------------:|--------------------:|
| test    |                0.384 |               0.028 |                           0.444 |                          0.036 |                0.542 |               0.067 |                                 0.761 |                                0.058 |                0.102 |               0.011 |
| val     |                0.383 |               0.047 |                           0.45  |                          0.039 |                0.513 |               0.063 |                                 0.748 |                                0.043 |                0.109 |               0.021 |

Validation minus test PR-AUC (detectable positives): **+0.006 ± 0.039**; test was lower in **4/8** re-simulations.
→ The gap is **not** systematic across re-simulations: a large part of it is sampling variability of the particular simulated events in the original benchmark.

## 3. Seasonal variability of the real signals (local σ = std of deviation from 24 h mean)

|                   |   test |   train |   val |
|:------------------|-------:|--------:|------:|
| ('ETTm1', 'HUFL') |   1.62 |       1 |  1.29 |
| ('ETTm1', 'HULL') |   1.25 |       1 |  0.87 |
| ('ETTm1', 'OT')   |   0.69 |       1 |  0.59 |
| ('ETTm2', 'HUFL') |   0.74 |       1 |  1.09 |
| ('ETTm2', 'HULL') |   0.84 |       1 |  0.95 |
| ('ETTm2', 'OT')   |   1.16 |       1 |  0.95 |

Values are relative to the training period. Injected signatures are scaled to the *training* σ, so a ratio > 1 means the same fault is buried in more natural variability (lower effective SNR).

## 4. Feature drift of model inputs (scaled; reference = training rows of the same asset)

|                   |   mean_psi |   n_psi_gt_0_25 |   mean_ks |
|:------------------|-----------:|----------------:|----------:|
| ('ETTm1', 'test') |      0.143 |              16 |     0.112 |
| ('ETTm1', 'val')  |      0.068 |               4 |     0.065 |
| ('ETTm2', 'test') |      0.052 |               1 |     0.059 |
| ('ETTm2', 'val')  |      0.139 |               7 |     0.089 |

Top-10 drifting features in test:

| asset   | feature       |   psi |    ks |
|:--------|:--------------|------:|------:|
| ETTm1   | MUFL_range16  | 0.442 | 0.266 |
| ETTm1   | LUFL_volratio | 0.425 | 0.181 |
| ETTm1   | HUFL_range16  | 0.413 | 0.268 |
| ETTm1   | OT_range16    | 0.407 | 0.266 |
| ETTm1   | MUFL_rstd16   | 0.403 | 0.251 |
| ETTm1   | LULL_range16  | 0.399 | 0.251 |
| ETTm1   | LULL_rstd16   | 0.392 | 0.256 |
| ETTm1   | OT_rstd16     | 0.366 | 0.25  |
| ETTm1   | HUFL_rstd16   | 0.366 | 0.255 |
| ETTm1   | LUFL_spikes16 | 0.362 | 0.25  |

PSI > 0.25 is conventionally 'major shift'.

## 5. Failure mix, detectability and precursor statistics

| split   |   intermittent_arcing |   load_oscillation |   sudden |   thermal_runaway |
|:--------|----------------------:|-------------------:|---------:|------------------:|
| test    |                    12 |                  9 |        7 |                12 |
| train   |                    55 |                 43 |       25 |                58 |
| val     |                    14 |                 10 |        5 |                12 |

| split   |   A |   B |   C |   D |
|:--------|----:|----:|----:|----:|
| test    |  22 |   9 |   2 |   7 |
| train   | 108 |  42 |   6 |  25 |
| val     |  28 |   7 |   1 |   5 |

Median signal strength of precursors (see detectability report for units):

| split   |   intermittent_arcing |   load_oscillation |   thermal_runaway |
|:--------|----------------------:|-------------------:|------------------:|
| test    |                   5.5 |               2.93 |              3.86 |
| train   |                   6   |               3.35 |              3.59 |
| val     |                   6   |               2.3  |              4.27 |

Mean precursor length (h): {'test': 10.61, 'train': 10.19, 'val': 12.23}

Positive (label) rate: train 0.0155, val 0.0164, test 0.0160

## 6. Threshold transfer and calibration

| split   |   recall@thr |   precision@thr |   fpr@thr |   median_score_pos |   median_score_neg |   p99_score_neg |   thr_for_recall_0.7 (diagnostic) |   brier |    ece |   mean_score |   positive_rate |
|:--------|-------------:|----------------:|----------:|-------------------:|-------------------:|----------------:|----------------------------------:|--------:|-------:|-------------:|----------------:|
| train   |       0.7017 |          0.3752 |    0.0184 |             0.8854 |             0.0146 |          0.825  |                            0.6608 |  0.0238 | 0.0497 |       0.0652 |          0.0155 |
| val     |       0.7012 |          0.3517 |    0.0216 |             0.913  |             0.0141 |          0.8641 |                            0.6593 |  0.0268 | 0.0534 |       0.0686 |          0.0164 |
| test    |       0.4875 |          0.3184 |    0.017  |             0.6179 |             0.0118 |          0.7742 |                            0.0515 |  0.0248 | 0.039  |       0.0543 |          0.016  |

## 7. Per asset × month (PR-AUC at sample level)

| split   | asset   | month   |   pos |   pr_auc |   recall |
|:--------|:--------|:--------|------:|---------:|---------:|
| val     | ETTm1   | 2017-11 |     8 |    0.868 |    1     |
| val     | ETTm1   | 2017-12 |    40 |    0.421 |    0.425 |
| val     | ETTm1   | 2018-01 |    48 |    0.527 |    0.646 |
| val     | ETTm1   | 2018-02 |    32 |    0.469 |    0.594 |
| val     | ETTm1   | 2018-03 |    16 |    0.853 |    1     |
| val     | ETTm2   | 2017-11 |     8 |    0.196 |    0.25  |
| val     | ETTm2   | 2017-12 |    48 |    0.372 |    0.417 |
| val     | ETTm2   | 2018-01 |    64 |    0.658 |    0.859 |
| val     | ETTm2   | 2018-02 |    48 |    0.772 |    0.958 |
| val     | ETTm2   | 2018-03 |    16 |    0.896 |    1     |
| test    | ETTm1   | 2018-03 |    24 |    0.489 |    0.458 |
| test    | ETTm1   | 2018-04 |    48 |    0.501 |    0.542 |
| test    | ETTm1   | 2018-05 |    56 |    0.327 |    0.536 |
| test    | ETTm1   | 2018-06 |    32 |    0.562 |    0.562 |
| test    | ETTm2   | 2018-03 |    32 |    0.508 |    0.625 |
| test    | ETTm2   | 2018-04 |    48 |    0.153 |    0.375 |
| test    | ETTm2   | 2018-05 |    40 |    0.357 |    0.4   |
| test    | ETTm2   | 2018-06 |    40 |    0.505 |    0.425 |
