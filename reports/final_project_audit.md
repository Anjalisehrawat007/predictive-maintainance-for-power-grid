# Final project audit

Scope: every Python package (`src/`, `backend/`, root scripts, `dashboard/`), the React frontend,
configuration, tests, experiment artefacts, documentation and dependencies, plus the claims made in the
README and the UI. Tools: manual review, `ruff` (E, F, W, B, UP, SIM, I), `tsc`, `oxlint`, `pip-audit`,
`npm audit`, the test suites, and targeted experiments that re-checked the claims.

Status legend: **Fixed** · **Mitigated** (reduced, documented) · **Accepted** (deliberate, documented).

## 1. Bugs and security

| # | Finding | Evidence | Status |
|---|---|---|---|
| S1 | **Path traversal in the SPA fallback route.** `/{path}` served `dist/<path>` without checking the resolved path, so `/..%2F..%2Fconfigs%2Fconfig.yaml` returned files from outside the web root. | Reproduced: the old handler returned `configs/config.yaml`. | **Fixed**: the resolved path must be inside `dist/`, and `/api/*` never falls back to the SPA. Regression test `test_path_traversal_blocked`. |
| S2 | Unvalidated request bodies (`dict = Body(...)`): `hours_before`, `n`, `start_index`, alert IDs and ack notes were unbounded or untyped. | Code review. | **Fixed**: Pydantic schemas with bounds (`backend/schemas.py`); malformed IDs → 422, unknown → 404. Tests in `tests/test_api_robustness.py`. |
| S3 | Unhandled exceptions could return stack traces. | Code review. | **Fixed**: a global handler returns `{"detail", "request_id"}`, and the error is logged as structured JSON. |
| S4 | CORS was allowed for all methods. | Code review. | **Fixed**: GET/POST/DELETE only; origins are configurable through `SGPM_CORS_ORIGINS`. |
| B1 | `tests/test_backend.py` used `TestClient(app)` without the lifespan, so it only passed if another module had loaded the model first (order-dependent). | It failed once the model moved to startup loading. | **Fixed** (`with TestClient(app)`). |
| B2 | Engine warm-up didn't update the anomaly-monitor state, so the first live step re-reported pre-existing deviations as "new" (wrong "subtle anomaly at −6 h" in replays). | Seen in Incident Replay. | **Fixed** in Phase 2 (state is updated during warm-up). |
| B3 | Production-only route clash: the Vite bundle path `/assets/*` swallowed the UI route `/assets/:id` (404). | Browser test against the built app. | **Fixed** (bundle moved to `/_app`). |
| B4 | The stale-cache risk: the browser could show cached model output while the backend was down. | Browser test. | **Fixed** (`cache: 'no-store'`). |
| B5 | The FA "sensor noise" criterion (any channel flat for 2 h) had a **43 %** base rate in normal operation, so the category was meaningless. | Base-rate measurement. | **Fixed**: only channels that are normally never flat count; the base rate is now 2.8 %. |
| B6 | Calibrated risk could display as **100 %** (an isotonic plateau). | UI review. | **Fixed**: displayed as "≥ 99 %"; this is not a certainty claim. |

## 2. Methodology (ML)

| # | Finding | Status |
|---|---|---|
| M1 | Phase-1 selection reused one set of simulated validation failures for every decision → optimistic validation (0.553 vs 0.36 on fresh events). | **Fixed**: a formal protocol (`reports/model_selection_protocol.md`) with 3 seeds × 5 fresh validation simulations; it is the primary selection methodology. |
| M2 | The thresholds and calibrator had also been fitted on those same original validation events. | **Fixed**: re-derived on pooled fresh validation under the same objective. The warning threshold moved from 0.659 to 0.241, which trades precision for recall and lead time (documented, not hidden). |
| M3 | No test proved that decisions ignore test data. | **Fixed**: `tests/test_temporal_integrity.py` scrambles every `test_*` column and asserts that the selection outcomes are unchanged. |
| M4 | The "robust" and "adaptive" feature variants and the training-on-harder-signatures experiment did not improve validation. | **Accepted** as negative results (kept in code and reports, not deployed). |
| M5 | The simulator is idealised (smooth envelopes, clean onset). | **Mitigated**: realism knobs and four stress profiles (`configs/failure_simulation.yaml`), plus a formal robustness curve. The reference benchmark is unchanged (bit-for-bit test). |
| M6 | Weak cross-asset transfer. | **Accepted** and analysed (both directions; invariant-feature strategy tested). |

## 3. Engineering quality

| # | Finding | Status |
|---|---|---|
| E1 | 91 lint findings (unused imports, dead variables, unsorted imports, an ambiguous variable name). | **Fixed**: `ruff check .` is clean under the `pyproject.toml` config. |
| E2 | Batch inference duplicated between `predict.py` and the tests. | **Fixed**: `src/streaming/inference.py` is shared by the CLI, the API and the tests. |
| E3 | No readiness/model/metrics endpoints, no request IDs, unstructured logs. | **Fixed**: `/api/ready`, `/api/model`, `/api/metrics`; `X-Request-ID`, `X-Response-Time-ms`; JSON logs; per-route p50/p95/p99. |
| E4 | The model was loaded lazily on the first request. | **Fixed**: loaded once at startup (lifespan); readiness reports failures; requests degrade with a 503. |
| E5 | Streaming features recomputed ~80 pandas rolling windows per observation (≈ 8.5 ms of a 14.3 ms step). | **Fixed**: NumPy newest-row path, 14× faster, equality-tested, with 100 % identical alarm statuses. |
| E6 | Loose dependency ranges (`>=`). | **Fixed**: exact pins (`requirements.txt`, `requirements-dev.txt`); `pip-audit` and `npm audit` find 0 known vulnerabilities. |
| E7 | No containerisation or env template. | **Mitigated**: `Dockerfile`, `docker-compose.yml`, `.env.example`. **Docker was not available on the development machine, so the image was not built here.** |
| E8 | The simulation parameters were mixed into the main config. | **Fixed**: `configs/failure_simulation.yaml` documents every mechanism, merged at load. |
| E9 | No frontend unit tests or performance trace. | **Fixed**: Vitest tests for formatting and adapters; `window.__sgPerf` trace used by the benchmark. |
| E10 | `backend/export_static.py` (full static snapshot for Vercel) would have published per-step, fault-injected ETDataset sensor values. | **Replaced** by `backend/export_results.py`: a results-only static site (aggregate JSON, guarded against sensor rows), deployed to Vercel. |
| E11 | The legacy Streamlit dashboard (`dashboard/`, `app.py`) duplicates part of the React UI. | **Accepted**: kept as an internal research/debug view and labelled as legacy in the README. |
| E12 | `src/streaming/engine.py` (Phase-1 `stream.py`) still uses the pandas feature path. | **Accepted**: CLI-only reproduction tool; the served engine (`backend/intelligence.py`) uses the fast path. |

## 4. Claims and UI text checked

* "Failure prediction" → always qualified as **simulated** failures; the disclosure is shown in the header, the Methodology page and the README.
* Recall is reported both over **all** failures and over **predictable** failures (the 7 sudden test failures have no precursor).
* "Health score" → labelled as an application-defined composite, not a grid-reliability standard.
* "Confidence" → labelled as **model confidence** (seed agreement), not physical certainty. Measured: only the HIGH bin is meaningfully more precise.
* Attribution → labelled "what the model relied on", not causes.
* No claims of real-grid deployment, blackout prevention or accuracy-based headline numbers.

## 5. Final polish pass (resume / demo / GitHub readiness)

| # | Finding | Status |
|---|---|---|
| P1 | **Two threshold sources.** CLI tools (`predict.py`, `stream.py`) still resolved the Phase-1 threshold 0.659 from `model_metadata.json`, while the API used 0.241 from `deployment.json`. | **Fixed**: `AlertPolicy.from_config` is the single resolver (config override → `deployment.json` → Phase-1 fallback); a test asserts the API and CLI agree. |
| P2 | Incident list and Incident Replay showed an "offline evaluation lead time" from the **Phase-1** detectability file next to the engine's final-OP lead time. | **Fixed**: taken from `reports/final/test_per_failure.csv`; a test asserts it equals the engine for all 40 test failures. |
| P3 | Model Lab presented the deployed model with Phase-1 operating-point metrics (precision 0.318, recall 0.487) without saying so. | **Fixed**: those rows are labelled "Phase-1 OP"; a separate "deployed OP" row shows the final test metrics. |
| P4 | Confidence-bin cut points (0.9/0.7) were duplicated in evaluation code and the UI; the UI hard-coded "2 h", "12 h", "48 × 15 min". | **Fixed**: `CONFIDENCE_BINS` is defined once and served by `/api/system` and `/api/model`; window and horizon labels come from model metadata. |
| P5 | `/api/system → benchmark` exposed a Phase-1 table (including Phase-1 OP metrics). | **Fixed**: latency only, from `reports/final/performance.json`. |
| P6 | The Methodology page lost its stage descriptions and timings after a backend rename, and quoted Phase-2 confidence precision (49 % / 14 %). | **Fixed**: both now come from final data (63 % / 13 % / 9 %). |
| P7 | `eval()` on CSV content in `src/analysis/eda.py`. | **Fixed**: `ast.literal_eval`. |
| P8 | Checkpoint file names contained `:` and `>` (cannot be checked out on Windows). | **Fixed**: renamed; the writer now sanitises names. |
| P9 | ONNX models were git-ignored, so a fresh clone silently served PyTorch. | **Fixed**: ONNX files (~0.8 MB) are committed. |
| P10 | One test (`test_spa_shell_is_not_cached`) was skipped on fresh clones because it needed a local frontend build. | **Fixed**: the SPA route reads `DIST` per request; tests use a temporary build directory. Path-traversal test strengthened (secret file next to dist). 0 skipped. |
| P11 | **Dataset licence.** ETDataset is CC BY-ND 4.0 (no sharing of adapted material); the engine log and stream CSVs contained row-level sensor values with simulated faults injected. | **Fixed**: git-ignored and rebuilt by `make data`; `DATASET.md` documents source, licence, citation and the unsettled status of trained weights. |
| P12 | Committed experiment logs contained absolute local paths (including the user name). | **Fixed**: paths made relative. |
| P13 | `make setup` failed on a machine where `python3.11` is an inactive pyenv shim. | **Fixed**: `.python-version` plus a working-interpreter fallback in the Makefile; verified from a clean copy. |
| P14 | UI: overlapping chart labels, raw internal model name in the header, clipped ruler tick, recall shown rounded differently from the docs. | **Fixed**. |
