# Performance benchmark

Machine: Apple M2 · macOS-26.5.2-arm64-arm-64bit · Python 3.11.9 · torch 2.14.1 · 1 intra-op thread for model calls (streaming setting). All times in **milliseconds**; percentiles over n calls after warm-up.

## Serving path (one observation, one transformer)

| Stage | p50 | p95 | p99 | n |
|---|---|---|---|---|
| features · pandas (114-row buffer) | 7.185 | 7.357 | 7.690 | 300 |
| features · NumPy newest row (deployed) | 0.511 | 0.541 | 0.562 | 2000 |
| model · PyTorch eager | 0.391 | 0.424 | 0.485 | 2000 |
| model · ONNX Runtime (deployed) | 0.072 | 0.079 | 0.083 | 2000 |
| confidence · 3 seed members | 1.176 | 1.216 | 1.259 | 1000 |
| calibration · isotonic | 0.036 | 0.037 | 0.047 | 5000 |
| attribution · integrated gradients (12 steps) | 3.445 | 3.519 | 3.606 | 500 |
| engine step · total (ingest→alert) | 5.490 | 5.678 | 6.073 | 1450 |

* Feature computation: the NumPy newest-row path is **14.1× faster** than recomputing pandas rolling windows over the buffer (same values: ≤ 1e-8 relative difference, tested; identical alarm statuses over all 20,904 test-period steps).
* Model inference: ONNX Runtime p50 0.072 ms vs PyTorch 0.391 ms (**5.4×**). ONNX is the deployed path (verified numerically identical at load).
* The full step is dominated by integrated-gradients attribution and the 3-member confidence ensemble — both optional for pure scoring (`/api/predict` skips them).

Batch throughput (PyTorch, 4 threads, batch 512): **37,373 windows/s** (p50 13.7 ms per batch).

## API

| Stage | p50 | p95 | p99 | n |
|---|---|---|---|---|
| in-process · GET /api/health | 0.565 | 0.643 | 0.685 | 300 |
| in-process · GET /api/snapshot | 4.061 | 4.178 | 4.438 | 200 |
| in-process · POST step n=1 (2 assets) | 12.769 | 13.062 | 13.269 | 300 |
| in-process · POST step n=4 (2 assets) | 48.823 | 49.481 | 49.725 | 150 |
| HTTP · GET /api/health | 0.599 | 0.679 | 0.724 | 300 |
| HTTP · GET /api/timeline (gzip) | 98.330 | 98.758 | 98.911 | 30 |
| HTTP · POST step n=1 (2 assets) | 13.130 | 13.420 | 13.670 | 300 |

In-process = FastAPI TestClient (no network); HTTP = real uvicorn server on localhost. A step for two transformers runs the engine twice and returns full records (sensors, scores, attribution, health, timings).

## Before → after (this iteration)

| | before | after |
|---|---|---|
| engine step p50 (ingest → alert, incl. attribution & confidence) | 14.26 ms (pandas features; engine-log run before this iteration) | 5.49 ms |
| feature computation p50 | 7.18 ms | 0.51 ms |