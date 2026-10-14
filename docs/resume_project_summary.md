# Resume project summary — SmartGrid AI

Every number below is in `reports/final/*.json` and the README. The wording deliberately avoids "deployed",
"prevented outages", "accuracy" and "real failures". The sensor data are real (ETDataset); the failure events
are **simulated**. Say "simulated failures" whenever you quote a detection number.

---

## A. ATS-focused
**SmartGrid AI — Predictive Maintenance & Early-Warning Platform** · Python, PyTorch, ONNX Runtime, FastAPI,
React, TypeScript
- Built a time-series anomaly early-warning system (GRU + temporal attention, deep learning) on real
  power-transformer sensor data with simulated failure events. It reached test PR-AUC 0.326 and ROC-AUC 0.845,
  and warned 75.8 % of predictable failures a median 3 h ahead.
- Developed a real-time streaming inference pipeline with FastAPI, ONNX Runtime and a React/TypeScript
  dashboard: 5.5 ms p50 per observation, ONNX inference 5.4× faster than PyTorch.
- Implemented leakage-safe model evaluation, probability calibration (Brier score 0.025 → 0.013),
  explainability (integrated gradients) and automated testing (pytest, Vitest, CI-style lint, type and
  security checks).

## B. Software-engineering focused (FAANG-style)
- Designed and built a streaming ML service in FastAPI: startup model loading, readiness and metrics endpoints,
  Pydantic-validated inputs, request IDs and structured JSON logs. Serves a GRU model through ONNX Runtime at
  5.5 ms p50 / 6.1 ms p99 per observation.
- Cut per-observation latency from 14.3 ms to 5.5 ms by replacing full pandas rolling-window recomputation with
  an incremental NumPy newest-row path. It matches the full recomputation within 1e-8 relative difference and
  gave identical alarm decisions over all 20,904 test steps.
- Found and fixed a path-traversal vulnerability in static-file serving, with a regression test. Added
  86 automated tests (76 pytest + 10 Vitest), including a temporal-integrity suite proving no test data
  reaches model selection.

## C. ML/AI focused
- Built a calibrated early-warning model (GRU + temporal attention, 31k parameters) over 12-hour windows of
  65 causal features. It warned 75.8 % of predictable simulated transformer failures a median 3 h ahead at
  0.25 false alarms per transformer-day (test PR-AUC 0.326).
- Showed that single-split validation overstated PR-AUC (0.55 vs 0.36). Replaced it with a 3-seed ×
  5-simulation selection protocol; under that protocol no TCN, LSTM or Transformer challenger beat the model.
- Added isotonic calibration (ECE 0.039 → 0.003), seed-ensemble confidence (HIGH-confidence alarms 63 % vs
  9 % precise), integrated-gradients attribution and stress-test robustness analysis.

## D. Research-focused
- Designed a leakage-safe benchmark for transformer-failure early warning: real ETDataset sensor streams plus a
  seeded, documented fault simulator. Detectability classes separate predictable failures (75.8 % warned) from
  ones with no precursor (0 % warned, by construction).
- Quantified selection bias. Single-realisation validation gave PR-AUC 0.553 against 0.36 on re-simulated
  events, which motivated a repeated fresh-simulation protocol with a pre-declared improvement margin.
- Reported negative results:
  - weak cross-transformer transfer (PR-AUC 0.13/0.24 vs 0.33 in-domain);
  - no gain from domain-invariant features (−0.005);
  - robustness training traded 0.09 normal-condition PR-AUC for no mean stress gain.

## E. Balanced (off-campus resume)
**SmartGrid AI — Real-time predictive maintenance for power transformers**
- Built an end-to-end early-warning system on real transformer sensor data with simulated failures:
  - GRU + temporal attention with causal feature engineering and leakage-safe temporal evaluation;
  - calibrated risk (Brier 0.025 → 0.013);
  - result: test PR-AUC 0.326, 75.8 % of predictable failures warned a median 3 h ahead.
- Served it through a FastAPI streaming engine with ONNX Runtime (5.5 ms p50 per observation, 2.6× faster
  after optimisation) and a React/TypeScript operations dashboard. The dashboard includes Incident Replay and a
  guided demo on held-out data.
- Hardened and tested the system: fixed a path-traversal vulnerability, added input validation and
  observability, and wrote 86 automated tests including temporal-integrity checks that prove test-set isolation.

---

## 30-second explanation
"SmartGrid AI is an early-warning system for power-transformer failures. It uses real sensor data from two
transformers. Because there are no public failure labels, I inject documented, simulated failure precursors. A
GRU with temporal attention reads the last 12 hours and estimates the risk of failure in the next 2 hours; that
risk is calibrated, explained, and streamed through a FastAPI engine to a React dashboard in about 5 ms per
reading. Most of my effort went into evaluation: leakage-safe splits, a protocol that caught over-optimistic
validation, and honest reporting. It warns about three-quarters of the predictable failures, about 3 hours
early."

## 2-minute technical explanation
"The data is ETDataset: two transformers, seven channels (load and oil temperature) every 15 minutes, with no
failure labels. A seeded simulator injects four mechanisms with 3–18 hour precursors: thermal runaway, load
oscillation, arcing, and sudden failures with no precursor. It also adds harmless look-alikes, so the model has
to tell degradation from noise. Simulated values are kept separate, and tests prove the model never sees them.

Features are strictly causal and fitted on training data only. Splits are chronological with an embargo. The
model is a GRU with additive temporal attention, 31k parameters, predicting failure within 2 hours.

The key methodological finding: Phase 1 tuned everything on one set of simulated validation events and reported
PR-AUC 0.55; re-simulating those events gave 0.36. So I rebuilt selection around 3 seeds × 5 fresh simulations
with a pre-declared margin. No TCN, LSTM or Transformer beat the GRU, so I kept it. Re-deriving the threshold
under the protocol lowered it from 0.659 to 0.241. That trades precision (0.32 → 0.17) for recall and lead time,
and I report the trade-off.

On top of the model there's isotonic calibration (Brier halved), a 3-seed confidence score where only HIGH
confidence is meaningfully more precise, and integrated-gradients attributions labelled as associational.

For serving, I replaced a pandas rolling-window recomputation with an incremental NumPy newest-row path and
moved inference to ONNX. A step went from 14 to 5.5 ms p50. FastAPI loads the model once, validates inputs,
and exposes readiness and metrics. I also found and fixed a path-traversal bug.

Limitations I raise myself: simulated labels, only two transformers with weak transfer between them, and 40
test failures."

## Numbers you can defend (and where they come from)
| Claim | Value | Source |
|---|---|---|
| Test PR-AUC / ROC-AUC | 0.326 / 0.845 | `reports/final/final_metrics.json` |
| Predictable / all simulated failures warned | 75.8 % / 62.5 % | same |
| Median lead time | 3 h (180 min) | same |
| False alarms | 0.25 per transformer-day | same |
| Precision / recall at the operating point | 0.171 / 0.575 | same |
| Fresh-validation PR-AUC | 0.360 ± 0.056 | `reports/final/protocol_summary.json` |
| Brier / ECE | 0.0248 → 0.0125 / 0.039 → 0.003 | `final_metrics.json` |
| Engine step latency | 5.49 ms p50, 6.07 ms p99 (Apple M2, 1 thread) | `reports/final/performance.json` |
| ONNX vs PyTorch | 0.072 vs 0.391 ms p50 | same |
| Tests | 76 pytest + 10 Vitest, 0 skipped | `reports/final_project_quality_report.md` |
