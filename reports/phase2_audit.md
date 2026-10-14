# Phase 2 · Audit of the existing system (before any change)

> **HISTORICAL (Phase 1 / Phase 2).** Kept as a research record. Operating-point metrics here use the Phase-1 threshold (0.659) and the original validation events. Final results: [`final_project_quality_report.md`](final_project_quality_report.md) and `reports/final/*.json`.

Audited: repository layout, `README.md`, `configs/config.yaml`, all of `src/`, `models/`, `experiments/`,
`reports/`, `dashboard/`, `app.py`, `stream.py`, `predict.py`, `evaluate.py`, and the tests.
Reproduction of the headline numbers is in `reports/phase2/reproduction.md`: **every reported test
metric reproduces exactly** from the saved artefacts.

## 1. Data flow (as implemented)

```
data/raw/ETTm{1,2}.csv (real, 15 min)
  → src/data/simulation.FaultSimulator   (seeded failure + benign-event injection; sim_* columns)
  → data/processed/{asset}.parquet + events.csv
  → src/features/engineering.FeatureEngineer (causal rolling/diff/residual features; fit on train rows)
  → src/preprocessing/scaling.AssetScaler (per-asset, train-only)  + splits (chronological, embargo)
  → src/training/trainer.WindowBatcher   (on-the-fly (B, 48, 65) windows)
  → src/models/recurrent.SequenceClassifier (GRU + additive attention, 31,425 params)
  → sigmoid → raw score (uncalibrated; weighted-BCE trained)
  → threshold 0.659 (val: max F1 s.t. recall ≥ 0.7) / critical 0.962 (val precision ≥ 0.9)
  → src/streaming/alerts.AlertPolicy (2 consecutive steps) → NORMAL / WARNING / CRITICAL
  → src/streaming/engine.StreamingPredictor (ring buffers; 8.4 ms/obs) → stream.py CSV logs
  → dashboard/main.py (Streamlit, 8 views)
```

## 2. What already works — keep unchanged

| Component | Evidence |
|---|---|
| Chronological split + embargo, train-only fitting, causal features | 6/6 automated leakage checks pass; unit tests for causality, lookback sufficiency and train-only fits |
| Streaming engine ≡ batch inference | `tests/test_streaming.py`; replay reproduced the offline event metrics exactly |
| Experiment tracking | 125 logged runs, per-stage decisions in `experiments/stage_log.json` |
| Selection discipline | Validation-only selection; minimum-gain rule rejected the lucky Optuna winner |
| Saved artefacts | `models/*.pt`, `.onnx`, `int8`, ensemble members, `preprocessor.pkl`, `model_metadata.json` |
| Reported numbers | Reproduced exactly (see reproduction report) |

**Must not be changed in Phase 2:** the Phase-1 processed dataset (`data/processed`), split definitions,
target rows, saved Phase-1 models/metadata, `experiments/results.csv`, `reports/*` from Phase 1, and the
leakage checks. All Phase-2 work goes to `experiments/phase2/`, `models/phase2/`, `reports/phase2/`,
`backend/`, and `frontend/`.

## 3. Current weaknesses

1. **Validation → test gap.** PR-AUC 0.553 → 0.326, event recall 0.89 → 0.73, and recall at the fixed
   threshold 0.70 → 0.49. Not yet explained quantitatively.
2. **Optimistic validation estimate.** The selected model is the *best of 3 seeds* on validation
   (0.553 vs a 3-seed mean of 0.527). Its validation number is therefore biased upward.
3. **Uncalibrated probabilities.** The model was trained with weighted BCE (pos_weight ≈ 7.9), which
   inflates scores. The dashboard displays the raw score as "failure probability".
4. **No uncertainty estimate.** Seed variance is large (±0.03 PR-AUC), but no per-prediction confidence
   is exposed.
5. **Weak cross-asset transfer.** ETTm1→ETTm2 PR-AUC 0.14 vs 0.32 in-domain. Only one direction was tested.
6. **Single horizon (2 h).** There is no estimate of *when* within the window.
7. **Recall is not split by detectability.** Sudden failures are excluded only as a group. Weak-signal
   precursors (low SNR relative to the period's natural variability) have not been quantified.
8. **False alarms only coarsely categorised.** 10 of 24 test false alarms are labelled "no injected
   event"; these are not examined further.

## 4. Performance bottlenecks

| Stage | Cost | Note |
|---|---|---|
| Streaming feature computation | ≈ 7.9 ms of 8.4 ms per observation | Recomputes pandas rolling features over a 114-row buffer every step |
| Model inference (PyTorch, 1 thread) | 0.40 ms | ONNX Runtime 0.07 ms is available but not used in the engine |
| Training | ≈ 7 s/epoch CPU | MPS slower for these small RNNs |
| Dashboard | Re-runs scripts on each interaction; images are static PNGs | Live page re-renders everything per tick |

## 5. Scientific limitations

* Failures are **simulated**. Signature magnitudes are expressed in σ of the *training* period. When natural
  variability is higher in a later season, the same injected magnitude yields a lower SNR. This is a
  plausible mechanism for the val/test gap, still to be verified.
* Simulator signatures follow deterministic smooth envelopes (`s^p`), always start exactly at onset, and
  always involve the same channels per mechanism. Real precursors are more irregular. The simulator also
  has an artefact: after an outage the observed signal blends back to the real one over 1 h, which can
  create step-like transients.
* Two assets only, a single held-out season, and 40 test failures, which gives wide confidence intervals.
* Phase-1 model comparisons used 2–3 seeds; the ablation used a mix of configurations (the feature
  stage ran *before* regularisation was tuned).

## 6. Frontend limitations (Streamlit dashboard)

* Generic layout, with metric tiles and static PNG figures. Few interactions are synchronised.
* The live view advances only through Streamlit reruns: no shared global clock, no timeline scrubbing,
  and no linked views.
* No topology view, alert lifecycle (acknowledge / investigate), incident-replay storyline, event feed,
  calibrated risk, confidence or health indicator.
* Status colours exist, but there is no reduced-motion handling or keyboard-first navigation.

## 7. Phase-2 plan derived from this audit

1. Quantify the val/test gap: shift, SNR, failure mix, threshold transfer, calibration.
2. Classify each failure by detectability and report recall per class.
3. Stress-test robustness on re-simulated benchmarks (new seeds, harder signatures) *without*
   changing the Phase-1 benchmark.
4. Run 3-seed experiments: robust features, cross-asset (both directions), TCN, temporal transformer,
   compact GRU, LSTM-attention, and a multi-horizon hazard head.
5. Calibration (temperature, isotonic; fitted on validation only) and seed-ensemble / MC-dropout
   uncertainty.
6. Keep the Phase-1 model unless a challenger is convincingly better (3-seed mean, minimum gain).
7. Product layer: a FastAPI backend wrapping the *real* streaming engine, and a React/TypeScript
   operations-center frontend.
