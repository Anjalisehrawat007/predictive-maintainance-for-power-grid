"""Performance benchmark: p50 / p95 / p99 for every stage of the serving path.

    python -m src.final.benchmark [--http http://127.0.0.1:8000]

Single process, CPU, 1 intra-op thread for model calls (the streaming engine's setting).  Run on an
otherwise idle machine; numbers are hardware-dependent and reported together with the machine info.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from backend.intelligence import SENSORS, AssetStream, load_deployment
from src.data.dataset import load_processed
from src.final.common import FINAL
from src.phase2.common import REPORTS, write_md
from src.utils.config import load_config


def pct(times_ms) -> dict:
    a = np.asarray(times_ms, dtype=float)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


def timeit(fn, n: int, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return pct(ts)


def run(http: str | None) -> dict:
    torch.set_num_threads(1)
    cfg = load_config()
    dep = load_deployment(cfg)
    dfs, _ = load_processed(cfg)
    df = dfs["ETTm1"]
    fe, art = dep.art.fe, dep.art
    i = 62000
    buf = df.iloc[i - fe.lookback + 1: i + 1]
    arr = buf[SENSORS].to_numpy()
    ts0 = buf.timestamp.iloc[-1]
    x = torch.from_numpy(art.features(df.iloc[i - fe.lookback - art.seq_len: i + 1], "ETTm1")[-art.seq_len:].astype(np.float32))[None]
    cpu = platform.processor() or platform.machine()
    try:                                              # macOS: human-readable CPU name
        import subprocess
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2).stdout.strip() or cpu
    except Exception:
        pass
    out: dict = {"machine": {"platform": platform.platform(), "processor": cpu,
                             "python": platform.python_version(), "torch": torch.__version__, "threads": 1}}
    st = {}
    st["features · pandas (114-row buffer)"] = timeit(lambda: fe.transform(buf, "ETTm1"), 300)
    st["features · NumPy newest row (deployed)"] = timeit(lambda: fe.transform_last(arr, ts0, "ETTm1"), 2000)
    with torch.no_grad():
        st["model · PyTorch eager"] = timeit(lambda: art.model(x), 2000)
    if dep.onnx is not None:
        xn = x.numpy()
        st["model · ONNX Runtime (deployed)"] = timeit(lambda: dep.onnx.run(None, {"x": xn}), 2000)
    with torch.no_grad():
        st["confidence · 3 seed members"] = timeit(lambda: [m(x) for m in dep.members], 1000)
    st["calibration · isotonic"] = timeit(lambda: dep.calibrator([0.3]), 5000)
    s = AssetStream(dep, "ETTm1")
    st["attribution · integrated gradients (12 steps)"] = timeit(lambda: s._attribution(x), 500)
    # full engine step over real consecutive observations
    eng = AssetStream(dep, "ETTm1")
    eng.prime(df.iloc[i - fe.lookback - art.seq_len: i])
    steps = []
    for j in range(i, i + 1500):
        r = df.iloc[j]
        t = time.perf_counter(); eng.update(r["timestamp"], r, bool(r["offline"]), j); steps.append((time.perf_counter() - t) * 1000)
    st["engine step · total (ingest→alert)"] = pct(steps[50:])
    out["stages_ms"] = st
    # batch throughput
    with torch.no_grad():
        xb = x.repeat(512, 1, 1)
        torch.set_num_threads(4)
        b = timeit(lambda: art.model(xb), 50, 5)
        torch.set_num_threads(1)
    out["batch_512_pytorch_4_threads"] = b | {"windows_per_s": 512 * 1000 / b["p50"]}
    # API
    from fastapi.testclient import TestClient

    from backend.app import app
    api = {}
    with TestClient(app) as c:
        lo = c.get("/api/system").json()["dataset"]["test_index"][0]
        sid = c.post("/api/sessions", json={"start_index": lo + 500}).json()["id"]
        api["in-process · GET /api/health"] = timeit(lambda: c.get("/api/health"), 300)
        api["in-process · GET /api/snapshot"] = timeit(lambda: c.get(f"/api/snapshot?index={lo + 700}"), 200)
        api["in-process · POST step n=1 (2 assets)"] = timeit(lambda: c.post(f"/api/sessions/{sid}/step", json={"n": 1}), 300)
        api["in-process · POST step n=4 (2 assets)"] = timeit(lambda: c.post(f"/api/sessions/{sid}/step", json={"n": 4}), 150)
    if http:
        import httpx
        with httpx.Client(base_url=http, timeout=10) as h:
            try:
                lo = h.get("/api/system").json()["dataset"]["test_index"][0]
                sid = h.post("/api/sessions", json={"start_index": lo + 500}).json()["id"]
                api["HTTP · GET /api/health"] = timeit(lambda: h.get("/api/health"), 300)
                api["HTTP · GET /api/timeline (gzip)"] = timeit(lambda: h.get("/api/timeline", headers={"Accept-Encoding": "gzip"}), 30, 3)
                api["HTTP · POST step n=1 (2 assets)"] = timeit(lambda: h.post(f"/api/sessions/{sid}/step", json={"n": 1}), 300)
                h.delete(f"/api/sessions/{sid}")
            except Exception as e:  # server not running → skip, recorded
                api["HTTP"] = {"skipped": str(e)}
    out["api_ms"] = api
    (FINAL / "performance.json").write_text(json.dumps(out, indent=1))
    return out


def write_report(o: dict, frontend: dict | None) -> None:
    def tab(d):
        rows = ["| Stage | p50 | p95 | p99 | n |", "|---|---|---|---|---|"]
        for k, v in d.items():
            if "p50" in v:
                rows.append(f"| {k} | {v['p50']:.3f} | {v['p95']:.3f} | {v['p99']:.3f} | {v['n']} |")
            else:
                rows.append(f"| {k} | {v} | | | |")
        return "\n".join(rows)
    st = o["stages_ms"]
    fp, fn = st["features · pandas (114-row buffer)"], st["features · NumPy newest row (deployed)"]
    mp, mo = st["model · PyTorch eager"], st.get("model · ONNX Runtime (deployed)")
    L = ["# Performance benchmark\n",
         f"Machine: {o['machine']['processor']} · {o['machine']['platform']} · Python {o['machine']['python']} · torch {o['machine']['torch']} · "
         "1 intra-op thread for model calls (streaming setting). All times in **milliseconds**; percentiles over n calls after warm-up.\n",
         "## Serving path (one observation, one transformer)\n", tab(st), "",
         f"* Feature computation: the NumPy newest-row path is **{fp['p50'] / fn['p50']:.1f}× faster** than recomputing pandas rolling "
         "windows over the buffer (same values: ≤ 1e-8 relative difference, tested; identical alarm statuses over all 20,904 test-period steps).",
         (f"* Model inference: ONNX Runtime p50 {mo['p50']:.3f} ms vs PyTorch {mp['p50']:.3f} ms (**{mp['p50'] / mo['p50']:.1f}×**). "
          "ONNX is the deployed path (verified numerically identical at load)." if mo else ""),
         "* The full step is dominated by integrated-gradients attribution and the 3-member confidence ensemble — both optional "
         "for pure scoring (`/api/predict` skips them).",
         f"\nBatch throughput (PyTorch, 4 threads, batch 512): **{o['batch_512_pytorch_4_threads']['windows_per_s']:,.0f} windows/s** "
         f"(p50 {o['batch_512_pytorch_4_threads']['p50']:.1f} ms per batch).\n",
         "## API\n", tab(o["api_ms"]), "",
         "In-process = FastAPI TestClient (no network); HTTP = real uvicorn server on localhost. A step for two transformers runs the "
         "engine twice and returns full records (sensors, scores, attribution, health, timings).\n"]
    if frontend:
        L += ["## Frontend (measured in the browser)\n", "| Metric | Value |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in frontend.items()]
        L.append("")
    L += ["## Before → after (this iteration)\n",
          "| | before | after |", "|---|---|---|",
          f"| engine step p50 (ingest → alert, incl. attribution & confidence) | 14.26 ms (pandas features; engine-log run before this iteration) | {st['engine step · total (ingest→alert)']['p50']:.2f} ms |",
          f"| feature computation p50 | {fp['p50']:.2f} ms | {fn['p50']:.2f} ms |"]
    write_md(REPORTS / "performance_benchmark.md", "\n".join(L))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", default=None)
    ap.add_argument("--frontend-json", default=None)
    a = ap.parse_args()
    res = run(a.http)
    fe_json = json.loads(Path(a.frontend_json).read_text()) if a.frontend_json else None
    write_report(res, fe_json)
    for k, v in res["stages_ms"].items():
        print(f"{k:45s} p50 {v['p50']:.3f}  p95 {v['p95']:.3f}  p99 {v['p99']:.3f}")
    for k, v in res["api_ms"].items():
        print(k, {kk: round(vv, 2) for kk, vv in v.items() if isinstance(vv, float)} if isinstance(v, dict) else v)
