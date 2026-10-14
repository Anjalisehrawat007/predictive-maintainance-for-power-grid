# Final resume readiness — SmartGrid AI

### Project Name
**SmartGrid AI: Real-Time Predictive Maintenance & Early-Warning Platform for Power Transformers**

### One-line Resume Description
Leakage-safe, calibrated GRU + temporal-attention early-warning system on real transformer sensor data
(simulated failure events), served by a 5.5 ms-per-reading FastAPI/ONNX streaming engine and a
React/TypeScript operations dashboard with incident replay.

### 3 Resume Bullets
- Built a calibrated early-warning model (GRU + temporal attention, 31k params) on real transformer sensor
  streams with simulated failures. It warned 75.8 % of predictable failures a median 3 h ahead at 0.25 false
  alarms per transformer-day (test PR-AUC 0.326, ROC-AUC 0.845; Brier 0.025 → 0.013).
- Designed leakage-safe evaluation (chronological splits with an embargo, 16 temporal-integrity tests) and found
  that single-split validation overstated PR-AUC (0.55 vs 0.36). A 3-seed × 5-simulation protocol replaced it;
  under it the GRU beat TCN, LSTM and Transformer challengers.
- Shipped a FastAPI + ONNX Runtime streaming engine (14.3 → 5.5 ms p50 per reading) and a React/TypeScript
  dashboard with Incident Replay. Fixed a path-traversal vulnerability; 86 automated tests (pytest + Vitest).

### ATS Keywords
Python · PyTorch · GRU · LSTM · temporal attention · time-series forecasting · anomaly detection · predictive
maintenance · deep learning · machine learning · ONNX Runtime · model calibration · isotonic regression ·
explainable AI (integrated gradients) · FastAPI · REST API · Pydantic · React · TypeScript · Vite ·
Tailwind CSS · streaming inference · real-time systems · latency optimisation · NumPy · pandas ·
scikit-learn · pytest · Vitest · data leakage prevention · model evaluation · PR-AUC · security hardening ·
reproducibility · Docker (configuration only)

### Best Metrics to Mention
| Metric | Value | Say it as |
|---|---|---|
| Predictable-failure recall | 75.8 % (25/33) | "of predictable **simulated** failures" |
| Median lead time | 3 h | "median 3 h before the (simulated) failure" |
| PR-AUC / ROC-AUC (test) | 0.326 / 0.845 | prevalence ≈ 1.6 %, so a random model scores PR-AUC ≈ 0.016 |
| False alarms | 0.25 per transformer-day | at the chosen operating point |
| Calibration | Brier 0.0248 → 0.0125, ECE 0.039 → 0.003 | isotonic, fitted on validation only |
| Latency | 5.49 ms p50 / 6.07 ms p99 per reading; ONNX 0.072 ms vs PyTorch 0.391 ms | Apple M2, 1 thread |
| Speed-up | 14.3 → 5.5 ms (features 7.2 → 0.51 ms) | NumPy newest-row features + ONNX |
| Tests | 76 pytest + 10 Vitest, 0 skipped | incl. 16 temporal-integrity tests |

Avoid precision (0.17) as a headline. Mention it when asked about trade-offs.

### Technical Interview Talking Points
1. **Why simulated failures**: there are no public labels, and the alternative is circular. Real sensors plus a
   seeded, documented simulator give a controllable benchmark.
2. **Selection bias found and fixed**: 0.553 → 0.36 on fresh events led to the 3 × 5 protocol, and the
   threshold moved 0.659 → 0.241 (a recall/precision trade-off, stated openly).
3. **Test isolation is *tested***: decisions are unchanged when test columns are scrambled, and the threshold
   and calibrator are re-derived from validation only.
4. **Predictable vs all failures**: 7 sudden failures are undetectable by construction (75.8 % vs 62.5 %).
5. **Calibration and confidence**: isotonic is monotone, so alarms are unchanged; only HIGH confidence is
   meaningfully more precise (63 % vs 9 %).
6. **Streaming optimisation**: newest-row NumPy features (14×), ONNX (5.4×), equivalence tests.
7. **Security**: the path-traversal bug was reproduced, fixed and regression-tested; Pydantic bounds; no stack
   traces.
8. **Negative results**: cross-asset transfer, domain-invariant features, robustness training and HPO were all
   rejected by the protocol.

Full Q&A: [`docs/interview_guide.md`](../docs/interview_guide.md) (30 questions).

### Honest Limitations
- Failure labels are **simulated**, at an accelerated rate. No real-world failure prediction is claimed.
- Two transformers only, and cross-asset transfer is weak (0.13/0.24 vs 0.33 PR-AUC in-domain).
- 40 test failures give wide uncertainty; seed variance spans fresh-validation PR-AUC 0.26–0.47.
- Precision is 0.17 at the chosen operating point.
- No voltage, current, frequency, weather or maintenance data.
- The Docker image was not built or tested (Docker was unavailable).
- No authentication; this is a local demo.
- Trained weights are not published (their status under ETDataset's CC BY-ND 4.0 is unresolved), so a fresh clone trains them locally (~20–30 min).

### GitHub Presentation Checklist
- [x] README: summary, disclaimer, metrics strip, architecture diagrams, results, screenshots, three-command quick start
- [x] Real screenshots (`docs/screenshots/`) captured from the running app
- [x] `LICENSE` (MIT, code only) and `DATASET.md` (CC BY-ND 4.0 data, not redistributed; attribution and citation)
- [x] `.gitignore`: venv, node_modules, caches, `.env`, logs, raw and adapted dataset files, build output
- [x] No secrets, no absolute local paths, no Windows-invalid file names
- [x] Interview guide, resume summary and quality reports present and consistent with `reports/final/*.json`
- [x] Copyright name in `LICENSE` matches the git identity (Anjali Sehrawat)
- [x] Trained weights excluded (unresolved CC BY-ND status); `make data` retrains them (~20–30 min, bit-identical on the dev machine)
- [ ] On GitHub: set the repository description and topics

### Demo Checklist
1. `make demo` and open <http://127.0.0.1:8000>. Check the header shows **SYSTEM ONLINE**.
2. Click **Demo mode** and walk the 8 steps:
   - live stream: risk rises through the warning line on the real engine;
   - explanation panel;
   - Incident Replay;
   - Model Lab;
   - latency panel.
3. In **Incident Replay**, press **REPLAY INCIDENT** at 8×. The result card shows FIRST RELIABLE WARNING
   3h 00m BEFORE FAILURE ✓ matches.
4. Open **Failure Intelligence**: 40 failures, 62.5 % / 75.8 % recall, and the false-alarm explorer (click
   "inspect").
5. Open **Model Lab**: the protocol table (why GRU + attention was kept) and the robustness curve.
6. Say the disclaimer out loud: *real sensor data, simulated failures, decision-support prototype*.

Files to show in an interview:
- `tests/test_temporal_integrity.py`
- `backend/intelligence.py` (streaming engine)
- `src/features/engineering.py` (`transform_last`)
- `reports/model_selection_protocol.md`
- `models/phase2/deployment.json`
- `backend/app.py` (the path-safe SPA route)
