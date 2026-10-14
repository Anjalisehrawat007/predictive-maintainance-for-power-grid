# Model-selection protocol (final)

*Written before the final protocol evaluation was run (2026-10-03). Results are appended by
`python -m src.final.run` in the "Results" section; the rules above it were not changed afterwards.*

## Why the protocol changed

Phase 1 tuned every choice (features, regularisation, architecture, loss, window, horizon, Optuna,
checkpoint, seed, threshold, calibrator) on **one** realisation of simulated validation failures. Phase 2
showed this was optimistic: the selected model scored PR-AUC 0.553 on those events but **0.36 ± 0.03** on
freshly simulated validation events (same real sensor data, new failure realisations). The validation
events had effectively become part of the training signal.

## The protocol

| Element | Rule |
|---|---|
| Data for every decision | **Fresh validation realisations**: the real sensor data of the validation period (2017-11-22 → 2018-03-09) with failure/disturbance events re-simulated using seeds 7000, 7010, 7020, 7030, 7040 (same simulator settings as the reference benchmark). |
| Test data | Original test period (2018-03-10 → 2018-06-26) and its re-simulated variants. **Reported only. Never used for any decision.** |
| Repetitions | Every candidate: **3 training seeds × 5 fresh validation realisations = 15 measurements**. Report mean, std, min, max. |
| Primary metric | Mean PR-AUC over the 15 measurements (threshold-free, imbalance-aware). Operational metrics (recall, precision, F1, false alarms/day, lead time) are reported alongside. |
| Replacing the incumbent | A challenger replaces GRU + attention only if its mean exceeds the incumbent's by more than **max(0.01, 1 standard error of the difference)**. More complex models must clear the same bar; ties keep the simpler/incumbent model. |
| Warning threshold | Chosen on the **pooled** fresh validation realisations: maximise F1 subject to recall ≥ 0.70 (sample level). |
| Critical threshold | Lowest threshold whose pooled fresh-validation precision ≥ 0.90 (never below the warning threshold). |
| Alarm rule | Threshold exceeded for 2 consecutive 15-min steps; alarm gaps ≤ 1 h merge into one episode. |
| Calibration | Isotonic regression fitted on the pooled fresh validation realisations (monotone, so alarms are unaffected). |
| Confidence | 1 − 2·σ of three seed-ensemble members; bins HIGH ≥ 0.9, MEDIUM 0.7–0.9, LOW < 0.7. |

## What "fresh" does and does not remove

It removes over-fitting to a particular set of simulated failure *events*. It does **not** create new real
sensor data: all realisations share the same real validation-period measurements, so the estimate is still
conditional on that period and on the simulator's assumptions.

## Results (appended by `python -m src.final.run`)

Fresh-validation metrics over 3 seeds × 5 realisations (mean ± std [min, max]); thresholds are each run's validation-selected threshold. Test columns are for reporting only.

| Candidate | PR-AUC | PR-AUC (predictable) | Recall | Precision | F1 | FA/day | Lead (min) | Test PR-AUC | Params | Size KB | Latency p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU + attention (incumbent) | 0.360 ± 0.056 [0.257, 0.475] | 0.406 ± 0.057 [0.301, 0.537] | 0.570 ± 0.058 [0.475, 0.655] | 0.300 ± 0.048 [0.208, 0.366] | 0.390 ± 0.043 [0.297, 0.459] | 0.135 ± 0.056 [0.058, 0.226] | 156 ± 42 | 0.324 ± 0.021 | 31,425 | 127 | 0.42 / 0.44 / 0.46 |
| GRU | 0.347 ± 0.044 [0.269, 0.428] | 0.391 ± 0.044 [0.320, 0.487] | 0.522 ± 0.068 [0.416, 0.622] | 0.366 ± 0.070 [0.263, 0.478] | 0.421 ± 0.037 [0.354, 0.460] | 0.081 ± 0.045 [0.029, 0.155] | 146 ± 43 | 0.303 ± 0.027 | 27,265 | 110 | 0.37 / 0.39 / 0.40 |
| Compact GRU (h=32) | 0.342 ± 0.038 [0.282, 0.406] | 0.387 ± 0.041 [0.293, 0.456] | 0.512 ± 0.078 [0.372, 0.654] | 0.354 ± 0.073 [0.230, 0.446] | 0.408 ± 0.041 [0.334, 0.467] | 0.090 ± 0.063 [0.014, 0.178] | 147 ± 41 | 0.286 ± 0.008 | 9,537 | 40 | 0.33 / 0.36 / 0.37 |
| TCN | 0.327 ± 0.059 [0.244, 0.442] | 0.370 ± 0.070 [0.286, 0.515] | 0.609 ± 0.045 [0.543, 0.676] | 0.166 ± 0.061 [0.084, 0.255] | 0.256 ± 0.078 [0.145, 0.360] | 0.403 ± 0.174 [0.207, 0.653] | 185 ± 41 | 0.337 ± 0.039 | 33,697 | 140 | 0.26 / 0.27 / 0.29 |
| LSTM + attention | 0.310 ± 0.059 [0.201, 0.411] | 0.347 ± 0.070 [0.210, 0.466] | 0.591 ± 0.046 [0.503, 0.657] | 0.191 ± 0.060 [0.121, 0.294] | 0.284 ± 0.066 [0.195, 0.391] | 0.254 ± 0.074 [0.135, 0.332] | 173 ± 49 | 0.309 ± 0.054 | 39,809 | 159 | 0.41 / 0.43 / 0.45 |
| Lightweight transformer | 0.283 ± 0.048 [0.194, 0.363] | 0.317 ± 0.053 [0.227, 0.408] | 0.608 ± 0.056 [0.516, 0.683] | 0.158 ± 0.045 [0.091, 0.209] | 0.248 ± 0.060 [0.157, 0.319] | 0.385 ± 0.095 [0.256, 0.539] | 181 ± 52 | 0.267 ± 0.050 | 23,489 | 102 | 0.26 / 0.27 / 0.28 |

**Decision (rule above):**

| Candidate | Δ mean PR-AUC vs incumbent | SE | required | replaces? |
|---|---|---|---|---|
| GRU + attention (incumbent) | +0.0000 | 0.0279 | 0.0279 | no |
| GRU | -0.0133 | 0.0209 | 0.0209 | no |
| Compact GRU (h=32) | -0.0174 | 0.0238 | 0.0238 | no |
| LSTM + attention | -0.0500 | 0.0295 | 0.0295 | no |
| TCN | -0.0327 | 0.0370 | 0.0370 | no |
| Lightweight transformer | -0.0771 | 0.0308 | 0.0308 | no |

**Outcome:** no candidate meets the rule — GRU + attention is kept.
