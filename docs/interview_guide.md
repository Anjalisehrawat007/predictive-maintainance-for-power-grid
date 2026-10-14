# Interview guide — SmartGrid AI

These answers describe what is **actually implemented** in this repository. Numbers come from
`reports/final/final_metrics.json`, `reports/final/protocol_summary.json`, `reports/final/performance.json` and
`models/phase2/deployment.json`. Test-period metrics refer to the held-out period 2018-03-10 → 2018-06-26 at the
deployed operating point, unless a number is labelled "Phase 1" (historical).

---

## Model choice

### 1. Why a GRU?
Failure precursors are temporal patterns. Examples: oil temperature rising more than load explains, oscillation
amplitude that grows, a spike rate that increases. A recurrent model reads a 12-hour window (48 × 15 min × 65
causal features) and can represent "this is growing" directly. A GRU has fewer gates than an LSTM, so fewer
parameters and less overfitting with about 180 training failures. In the selection protocol, the GRU family
was at the top: GRU + attention 0.360, plain GRU 0.347, compact GRU 0.342. Evidence that temporal modelling
matters: on raw signals a GRU reached about 0.24 validation PR-AUC against about 0.08 for gradient boosting.
With rich engineered features the gap to strong tabular baselines shrinks, because the features already
summarise history.

### 2. Why temporal attention?
Additive attention pools a weighted summary of all hidden states next to the last state. A precursor that peaked
a few hours ago is then not "forgotten". It costs about 4k parameters (31,425 vs 27,265) and almost no latency.
It had the best mean fresh-validation PR-AUC (0.360 ± 0.056), but the gain over plain GRU (+0.013) is inside
one standard error. **I don't claim attention is significantly better.** It was kept because nothing beat it
by the pre-declared margin.

### 3. Why not LSTM, TCN or a Transformer?
All three were trained under the same protocol (3 seeds × 5 fresh simulations). Mean fresh-validation PR-AUC
was 0.310 for LSTM + attention, 0.327 for TCN and 0.283 for a small Transformer, against 0.360 for the
incumbent. With few positive events and high seed variance, the GRU's simple inductive bias of "evolving state"
generalised best. Transformers pay off with much more data or longer contexts. TCN had good recall but low
precision (0.17 at its own threshold).

---

## Evaluation and leakage

### 4. Why PR-AUC instead of accuracy?
About 1.6 % of 15-minute steps are positive, so "never fails" scores 98.4 % accuracy. PR-AUC measures ranking
quality on the rare class without fixing a threshold. Operationally I also report:
- event recall over all failures and over predictable failures;
- false alarms per transformer-day;
- lead time of the first reliable warning;
- calibration (Brier, ECE).

### 5. How did you prevent temporal leakage?
- **Splits:** chronological 70/15/15 per transformer, with an embargo. No label window crosses into the next
  split, and validation/test input windows start inside their own split.
- **Features:** backward-looking only. Tests perturb every future row on real data and assert past features are
  bit-identical, for every feature variant.
- **Fitting:** scalers, the thermal-residual regression and spike thresholds are fitted on training rows only
  (tested).
- **Simulator metadata:** a test permutes all `sim_*` and label columns and asserts the features don't change.
- **Decisions:** thresholds, calibration and model selection use validation only. See Q20.

All of this is in [`tests/test_temporal_integrity.py`](../tests/test_temporal_integrity.py) and
`src/preprocessing/leakage.py`.

---

## Simulated failures

### 6. How were failure events generated?
`src/data/simulation.py` is configured by `configs/failure_simulation.yaml` and is seeded, so it regenerates
bit-for-bit (tested). On top of the real readings it injects four failure mechanisms, each with a 3–18 h
precursor whose magnitude is relative to the channel's real local variability:

| Mechanism | Share | Precursor |
|---|---|---|
| Thermal runaway | 35 % | oil temperature above what load explains |
| Load oscillation | 25 % | growing HV/MV oscillation |
| Intermittent arcing | 25 % | rising reactive-load spike rate |
| Sudden | 15 % | none |

After a trip the asset goes offline for 2–8 h, and those rows are never scored. Harmless look-alikes
(0.25/day: thermal bumps, load surges, short oscillations, sensor glitches) never lead to failure, so the model
must separate degradation from ordinary disturbance. Real and simulated values are stored separately.

### 7. Why use simulated failures?
ETDataset has real transformer load and oil-temperature data but **no failure labels**. I reviewed alternatives
(`reports/dataset_review.md`). The closest public candidate derives "faults" from the transformer's own
temperature alarms, which makes predicting them from temperature almost circular. Real sensors plus documented,
seeded precursors give a controllable benchmark: real noise, seasonality and load behaviour, with known ground
truth. The cost is that results describe detection of *simulated signatures*, and I state that everywhere.

### 8. What does "predictable failure" mean?
For every failure I measured how much precursor signal actually reached the sensors. That is the peak injected
amplitude relative to real local variability before onset, or the visible spike count for arcing. Failures fall
into classes A (strong), B (moderate), C (weak) and D (sudden, no precursor). The class thresholds were fixed
before looking at recall. "Predictable" means classes A–C: 33 of the 40 test failures.

### 9. Why are some failures undetectable?
The 7 sudden test failures (class D) have no precursor by construction, so no sensor-based model can warn of
them. That is why I report both 62.5 % of all failures warned (25/40) and 75.8 % of predictable failures
(25/33). Some strong-signal failures were still missed (17/22 class A were warned). Those are model or threshold
errors, not sensor limits, and the Failure Intelligence page lists them for replay.

---

## Operating point and calibration

### 10. Why did the threshold change from 0.659 to 0.241?
Both use the same rule: maximise F1 subject to recall ≥ 0.70 on validation. Phase 1 applied it to **one** set of
simulated validation events, which every other decision had also been tuned on. When I re-simulated the events,
the model's scores on true positives were lower. Reaching 70 % recall on pooled fresh validation needed a lower
raw threshold, **0.241**, which is about 4 % calibrated risk. The test set played no part. A test re-derives
this threshold from the fresh validation simulations and asserts it equals the deployed value in
`models/phase2/deployment.json`.

### 11. Why did recall improve while precision fell?
A lower threshold raises more alarms. On the same test data:

| | Phase 1 (0.659) | Final (0.241) |
|---|---|---|
| Recall | 0.487 | 0.575 |
| Precision | 0.318 | 0.171 |
| Median lead | 2 h | 3 h |
| False alarms per day | 0.115 | 0.25 |

PR-AUC (0.326) and ROC-AUC (0.845) don't change, because they don't depend on the threshold. I present this
openly as a trade-off chosen by a stated rule, not as an improvement.

### 12. Why calibrate probabilities?
Weighted BCE (pos_weight ≈ 7.9) inflates raw scores, so the raw score ranks well but isn't a probability.
Operators need a number they can interpret: "about 4 % chance of failure in the next 2 h" must mean roughly 4 %.
Calibration also makes "expected failures in the next 2 h" (the sum of risks) meaningful in the UI.

### 13. What is isotonic calibration?
It fits a monotone, non-decreasing step function from raw score to observed frequency (pool-adjacent-violators).
It was fitted on the pooled fresh **validation** simulations, and chosen over Platt and temperature scaling by
time-split cross-fitting inside validation. Because it is monotone, the ranking, the alarms and PR-AUC are
unchanged; only the displayed probability changes. Its plateaus can reach 1.0, so the UI caps the display at
"≥ 99 %".

### 14. What does the Brier score mean here?
It is the mean squared error between predicted probability and the 0/1 outcome per step; lower is better.
On test, isotonic calibration cut it from 0.0248 to **0.0125**, and expected calibration error from 0.039 to
**0.003**. With 1.6 % prevalence the Brier score is dominated by negatives, so I report it together with ECE and
the reliability diagram.

### 15. What does the confidence score represent?
Confidence = 1 − 2·σ of the probabilities from three independently seeded models. It measures **model
agreement**, not certainty about the physical transformer. Measured on test, HIGH-confidence alarm steps (≥ 0.9)
were correct 63 % of the time, against 13 % for MEDIUM and 9 % for LOW. So HIGH vs not-HIGH is useful for
triage, and MEDIUM vs LOW is not.

---

## Errors and robustness

### 16. What caused false alarms?
There were 52 test false-alarm episodes (0.25 per transformer-day), each categorised by a rule checked against
a base rate:

| Category | Episodes | Base rate in quiet windows |
|---|---|---|
| Harmless injected disturbances | 24 | — |
| Unusual real sensor behaviour (|z| ≥ 3) | 15 | 22 % |
| Model error | 11 | — |
| Sensor noise | 1 | 3 % |
| Slow thermal behaviour | 1 | 2 % |

I fixed one category whose base rate was 43 %, because a category that common in normal operation explains
nothing.

### 17. What happens under degraded sensor conditions?
I built four stress profiles that make signatures weaker, delayed, intermittent, irregular, asset-varied and
overlapping. On the test period (5 re-simulations × 3 seeds each), from NORMAL to SEVERE:
- PR-AUC fell from 0.38 to 0.20;
- predictable failures warned fell from 86 % to 66 %;
- median lead fell from 197 to 117 min.

Training on MODERATE signatures helped recall under stress but cost 0.09 PR-AUC in normal conditions, with no
mean gain, so it was not adopted.

### 18. Why is cross-asset generalisation weak?
The two transformers operate in different regimes (ETTm2 loads are about 5× larger, with different thermal
dynamics), and many features shift strongly between them. Trained on T1 and tested on T2, PR-AUC was 0.133;
T2 → T1 gave 0.242; in-domain is about 0.33. Relative ("robust") features transferred worse. A label-free
domain-invariant subset (27 features with PSI < 0.1) changed transfer by −0.005, so it was not adopted. Real
fleet use would need per-asset fine-tuning or hierarchical models and many more assets.

### 19. Why wasn't the best HPO model deployed?
The Phase-2 Optuna winner scored 0.58 on one seed and the original validation events. Over 3 seeds it averaged
0.527, the same as the incumbent, with 79 % more parameters. Under the protocol, a challenger must beat the
incumbent's mean by more than max(0.01, 1 SE), so it was rejected. The single-seed number was selection noise.
Keeping a model is a valid outcome of a fair comparison.

### 20. How did you prove test-set isolation?
`tests/test_temporal_integrity.py` scrambles every `test_*` column of the experiment logs and re-runs the
selection logic (Phase-1 `selection_score`, Phase-2 `decide`), asserting the decisions are unchanged. Two more
tests support this:
- the protocol run log contains no test columns at all;
- the deployed threshold and calibrator are re-derived from the pooled fresh validation simulations and must
  equal `deployment.json`.

---

## Serving and engineering

### 21. How does streaming inference work?
Each new reading, in `backend/intelligence.py`, goes through these steps:
1. Append to a 114-row ring buffer.
2. Compute causal features for the newest row only (NumPy).
3. Apply the per-asset training-fitted scaler.
4. Push into the 48-step sequence buffer.
5. Run ONNX inference.
6. Apply isotonic calibration and compute 3-seed confidence.
7. Compute integrated-gradients attribution.
8. Run the risk engine: 1 h trend, derived health, status with 2-step persistence.
9. Run the alert engine: events and the alert lifecycle.

Sessions warm up from the preceding rows in one vectorised pass, which a test shows equals step-by-step
warm-up.

### 22. Why compute only the newest feature row?
Recomputing about 80 pandas rolling windows over the whole buffer for every observation took 7.2 ms p50. That
was most of the step. Only the newest row is new, so `transform_last` computes just that row in NumPy: 0.51 ms,
14× faster. A test checks it matches the full pandas computation (within 1e-8 relative difference), and alarm
statuses were identical over all 20,904 test steps. The full engine step went from 14.3 ms to **5.5 ms p50**.

### 23. Why ONNX?
The same weights run in 0.072 ms per window in ONNX Runtime against 0.391 ms in eager PyTorch (5.4×), single
thread, and scoring doesn't need PyTorch. At startup the engine compares ONNX output with PyTorch on random
inputs and uses ONNX only if they match to 1e-4. I also tried int8 dynamic quantisation: models became about 3×
smaller but *slower* on this ARM CPU, so it isn't deployed.

### 24. What are p50, p95 and p99?
They are the 50th, 95th and 99th percentiles of latency over many calls after warm-up. p50 is the typical
case; p95 and p99 show the tail a user occasionally feels. I report percentiles rather than means because
latency distributions are skewed. On an Apple M2 with one thread, the engine step is 5.49 / 5.68 / 6.07 ms
(`src/final/benchmark.py`, idle machine).

### 25. How does FastAPI serve the model?
`backend/app.py` exposes:
- `/api/health` and `/api/ready` (503 until model, data and engine log are loaded);
- `/api/model` (features, thresholds, calibration, weights hash);
- `/api/metrics` (per-route latency percentiles);
- `/api/predict` (validated batch scoring);
- streaming sessions, Incident Replay and the alert endpoints.

Inputs are Pydantic schemas with bounds: finite values, strictly increasing timestamps, at most 5,000 rows,
step size 1–64. Middleware adds `X-Request-ID` and `X-Response-Time-ms` and logs one structured JSON line per
request. The built React app is served by the same process.

### 26. How is the model loaded?
Once, in the FastAPI lifespan handler at startup: model, preprocessing, calibrator, 3 ensemble members and the
ONNX session (after the equivalence check). Its thresholds come from `models/phase2/deployment.json`.
`AlertPolicy.from_config` is the single resolver used by the API and all CLI tools, and a test asserts both
resolve the same thresholds. If loading fails, the readiness probe returns 503 with the error, and
model-dependent routes return 503 instead of crashing.

### 27. What security issues did you find?
- **Path traversal** in static-file serving (Q28).
- **Unvalidated request bodies:** fixed with Pydantic schemas; malformed IDs return 422, unknown ones 404.
- **Stack traces in error responses:** a global handler now returns only `detail` plus `request_id`.
- **CORS allowed every method:** now GET/POST/DELETE, with env-configured origins.
- **A local `eval()` in analysis code:** replaced with `ast.literal_eval`.

`pip-audit` and `npm audit` report 0 known vulnerabilities. There are no secrets in the project. I did not add
authentication: this is a local single-user demo, and I say so rather than bolting on decorative auth.

### 28. What was the path-traversal issue?
The single-page-app fallback route served `dist/<path>` without checking where the resolved path pointed.
`/..%2F..%2Fconfigs%2Fconfig.yaml` returned the project config; I reproduced it before fixing. Now the resolved
path must be inside `dist/` (`Path.resolve().is_relative_to(dist)`), `/api/*` never falls back to the SPA, and
anything else gets the shell. A regression test uses a temporary build directory with a secret file next to
it, and asserts several encodings of `../` can't reach it.

---

## Framing

### 29. Why shouldn't the system claim blackout prevention?
- The failure labels are simulated.
- The system was never deployed on a real grid.
- Two transformers are not a fleet.
- Even on the benchmark, 37.5 % of test failures were not warned in time.
- 17 % of alarm steps are correct at the chosen operating point.

The honest claim: on real sensor data with documented simulated precursors, a calibrated early-warning model
warned 75.8 % of predictable failures a median 3 h ahead at 0.25 false alarms per transformer-day. It is a
decision-support prototype. Claiming more would be the first thing a domain expert takes apart.

### 30. What would you do with real industrial failure labels?
- Replace simulated labels with real trip and maintenance records, and keep the leakage-safe pipeline and
  selection protocol.
- Evaluate rolling-origin over several years and many assets, grouping assets in splits.
- Add the channels that matter physically: DGA, winding temperature, voltage, current, ambient temperature.
- Use per-asset fine-tuning or hierarchical models for transfer, and calibrate per asset class.
- Pick thresholds from a cost model agreed with operators (cost of a missed failure vs. a site visit).
- Monitor drift in production.
