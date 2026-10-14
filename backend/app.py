"""SmartGrid AI — inference/API layer over the Python ML system.

Run:  .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000

The deployed model, calibrator and confidence ensemble are loaded ONCE at startup
(lifespan) and shared by every request.  Endpoint map (all under /api):

  GET  /health               liveness            GET  /ready      readiness (model + data + engine log)
  GET  /model                model metadata      GET  /metrics    request + engine latency p50/p95/p99
  GET  /system               UI bootstrap        GET  /assets     transformer assets
  GET  /timeline /log /snapshot                  precomputed engine log (global time scrubber)
  POST /sessions             start a stream      POST /sessions/{id}/step     advance the real engine
  GET  /incidents            held-out failures   POST /incidents/{asset}/{id}/replay   incident replay session
  GET  /alerts               alerts              POST/DELETE /alerts/{id}/ack
  POST /predict              batch scoring of posted sensor rows
  GET  /analytics/failures /models /methodology /patterns   research data for the UI
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.intelligence import EVENT_RULES, HEALTH_WEIGHTS, SENSORS, load_deployment
from backend.observability import METRICS, RequestContextMiddleware, log_event, setup_logging
from backend.schemas import AckBody, PredictBody, ReplayBody, ScenarioBody, SessionCreate, StepBody, UploadBody
from backend.sessions import Session, SessionManager
from backend.store import ASSET_LABEL, PUBLIC, R, Store, _clean, _json, records
from src.streaming.calibration import CONFIDENCE_BINS
from src.utils.config import PROJECT_ROOT

setup_logging(os.environ.get("SGPM_LOG_LEVEL", "INFO"))
STORE = Store()
SESSIONS = SessionManager()
_STATE: dict = {"dep": None, "error": None, "loaded_at": None}
DISCLOSURE = ("Sensor measurements are real (ETDataset, two power transformers, 15-min SCADA data, 2016–2018). "
              "Failure events are SIMULATED because the source dataset does not contain real failure labels.")
ALERT_ID = re.compile(r"^(log-\d{1,6}|s-[0-9a-f]{10}-\d{1,6})$")


def _load_model() -> None:
    t = time.perf_counter()
    try:
        _STATE["dep"] = load_deployment(STORE.cfg)
        _STATE["error"] = None
        _STATE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log_event("model_loaded", model=_STATE["dep"].name, onnx=_STATE["dep"].onnx is not None,
                  ms=round((time.perf_counter() - t) * 1000, 1))
    except Exception as e:  # keep the API up; /ready and model endpoints report the problem
        _STATE["dep"], _STATE["error"] = None, f"{type(e).__name__}: {e}"
        log_event("model_load_failed", logging.ERROR, error=_STATE["error"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()                         # model, calibrator, ensemble, ONNX session: loaded once
    _ = STORE.data, STORE.log             # warm data caches so the first UI request is fast
    yield


app = FastAPI(title="SmartGrid AI API", version="2.1", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get(
    "SGPM_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o],
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    log_event("unhandled_error", logging.ERROR, path=request.url.path, error=type(exc).__name__, request_id=rid)
    return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})


@app.exception_handler(RequestValidationError)
async def invalid(request: Request, exc: RequestValidationError):
    errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()][:10]
    return JSONResponse(status_code=422, content={"detail": "invalid request", "errors": errs,
                                                  "request_id": getattr(request.state, "request_id", None)})


def dep():
    if _STATE["dep"] is None:
        raise HTTPException(503, f"model unavailable: {_STATE['error'] or 'not loaded'}")
    return _STATE["dep"]


def _no_rows_in_public():
    """Endpoints that would serve held-out ETDataset rows are disabled on the public deployment."""
    if PUBLIC:
        raise HTTPException(404, "not available on the public deployment: this view replays ETDataset sensor rows "
                                 "(CC BY-ND 4.0, not redistributed). Use the Scenario Lab or upload your own data, "
                                 "or run the project locally.")


def _require_log():
    _no_rows_in_public()
    if not STORE.log:
        raise HTTPException(503, "engine log not built — run: python -m backend.precompute")


def _asset(a: str) -> str:
    if a not in STORE.assets:
        raise HTTPException(404, f"unknown asset {a!r}; known: {STORE.assets}")
    return a


def _test_range():
    a = next(iter(STORE.bounds))
    return STORE.bounds[a].period("test")


# --------------------------------------------------------------------------- service
@app.get("/api/health")
def health():
    """Liveness: the process is up (does not require the model)."""
    return {"ok": True, "status": "alive", "sessions": len(SESSIONS.sessions)}


@app.get("/api/ready")
def ready():
    """Readiness: model loaded, data available, engine log built."""
    checks = {"model_loaded": _STATE["dep"] is not None, "data_loaded": PUBLIC or bool(STORE.dfs),
              "engine_log": PUBLIC or bool(STORE.log)}
    body = {"ready": all(checks.values()), "checks": checks, "model_error": _STATE["error"], "loaded_at": _STATE["loaded_at"]}
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.get("/api/model")
def model_info():
    d = dep()
    m = d.art.meta
    f = PROJECT_ROOT / "models" / f"{m['active_model']}.pt"
    return _clean({
        "name": m["active_model"], "architecture": m["model_type"], "parameters": d.parameters,
        "weights_sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else None,
        "input": {"features": d.art.feature_names, "n_features": len(d.art.feature_names), "seq_len": d.art.seq_len,
                  "window_hours": d.art.seq_len * 15 / 60, "feature_set": m["feature_set"]},
        "output": {"target": f"P(failure within {d.horizon_min} min)", "horizon_min": d.horizon_min},
        "runtime": "onnxruntime" if d.onnx is not None else "pytorch",
        "calibration": {"method": d.calibrator.method, "fitted_on": "validation period only"},
        "confidence": {"method": "1 - 2*std of seed-ensemble member probabilities", "members": len(d.members)},
        "thresholds": {"warning_raw": d.warn_raw, "critical_raw": d.crit_raw, "warning_risk": d.warn_cal,
                       "critical_risk": d.crit_cal, "persistence_steps": STORE.cfg["threshold"]["persistence_k"],
                       "confidence_bins": CONFIDENCE_BINS,
                       "selection": STORE.cfg["threshold"]},
        "deployment": d.manifest, "loaded_at": _STATE["loaded_at"], "training_seed": m.get("seed"),
    })


@app.get("/api/metrics")
def metrics():
    return _clean(METRICS.snapshot() | {"sessions_active": len(SESSIONS.sessions)})


# --------------------------------------------------------------------------- system
MODEL_NAMES = {"gru_attention": "GRU + temporal attention", "gru": "GRU", "lstm": "LSTM", "lstm_attention": "LSTM + attention"}

@app.get("/api/system")
def system():
    d = dep()
    m = d.art.meta
    perf = _json(R / "final" / "performance.json", {})
    st = perf.get("stages_ms", {})
    p50 = lambda k: st.get(k, {}).get("p50")  # noqa: E731
    bench = {"source": "reports/final/performance.json (src/final/benchmark.py)",
             "machine": perf.get("machine", {}).get("processor"),
             "pytorch_ms_p50": p50("model · PyTorch eager"), "onnx_ms_p50": p50("model · ONNX Runtime (deployed)"),
             "engine_step_ms_p50": p50("engine step · total (ingest→alert)"),
             "engine_step_ms_p99": st.get("engine step · total (ingest→alert)", {}).get("p99")} if perf else {}
    lo, hi = _test_range()
    a0 = STORE.assets[0]
    return _clean({
        "status": "online", "time": time.time(), "disclosure": DISCLOSURE, "public": PUBLIC,
        "engine": {"runtime": type(d.scorer).__name__, "attribution": d.scorer.attribution_method,
                   "max_run_rows": MAX_RUN_ROWS},
        "model": {"name": m["active_model"], "type": m["model_type"], "parameters": d.parameters,
                  "seq_len": d.art.seq_len, "window_hours": d.art.seq_len * 15 / 60, "horizon_min": d.horizon_min,
                  "feature_set": m["feature_set"], "n_features": len(d.art.feature_names), "onnx_runtime": d.onnx is not None,
                  "ensemble_members": len(d.members), "calibration": d.calibrator.method,
                  "selected_in": d.manifest.get("selected_in", "Phase 1 (kept in Phase 2)")},
        "thresholds": {"warning_raw": d.warn_raw, "critical_raw": d.crit_raw, "warning_risk": d.warn_cal,
                       "critical_risk": d.crit_cal, "persistence_steps": STORE.cfg["threshold"]["persistence_k"],
                       "confidence_bins": CONFIDENCE_BINS},
        "dataset": {"name": STORE.cfg["dataset"]["name"], "assets": [{"id": a, "label": ASSET_LABEL.get(a, a)} for a in STORE.assets],
                    "sampling_min": STORE.cfg["dataset"]["freq_minutes"],
                    "test_period": STORE.split_meta.get(a0, {}).get("test", [None, None]),
                    "test_index": [lo, hi]},
        "benchmark": bench, "engine_timings": STORE.timings,
        "health_weights": HEALTH_WEIGHTS, "event_rules": EVENT_RULES,
    })


@app.get("/api/assets")
def assets():
    tf = STORE.test_failures()
    return [{"id": a, "label": ASSET_LABEL.get(a, a), "name": f"Transformer {ASSET_LABEL.get(a, a)}",
             "substation": f"Substation {'A' if a.endswith('1') else 'B'}", "source": f"ETDataset {a}",
             "test_failures": int((tf.asset == a).sum())} for a in STORE.assets]


# --------------------------------------------------------------------------- engine log (scrubber)
_TIMELINE: dict = {}


@app.get("/api/timeline")
def timeline():
    """Status/risk arrays for the global scrubber. The engine log is static, so the payload is built once."""
    _require_log()
    if "bytes" not in _TIMELINE:            # serialise once: skips per-request encoding of ~21k points
        _TIMELINE["bytes"] = json.dumps(_build_timeline(), separators=(",", ":")).encode()
    return Response(content=_TIMELINE["bytes"], media_type="application/json")


def _build_timeline():
    out = {}
    code = {"WARMUP": 0, "NORMAL": 1, "WARNING": 2, "CRITICAL": 3, "OFFLINE": 4}
    for a, lg in STORE.log.items():
        out[a] = {"index": lg["index"].tolist(), "risk": [None if v != v else round(v, 5) for v in lg.risk.astype(float)],
                  "status": [code.get(s, 0) for s in lg.status], "confidence": [None if v != v else round(v, 3) for v in lg.confidence.astype(float)],
                  "health": [None if v != v else round(v, 1) for v in lg.health.astype(float)]}
    a0 = next(iter(STORE.log))
    tf = STORE.test_failures()
    return _clean({"assets": out, "timestamps": STORE.log[a0].timestamp.tolist(), "status_codes": code,
                   "failures": records(tf[["asset", "event_id", "type", "failure_idx", "failure_time", "label"]])})


@app.get("/api/log")
def log(asset: str, start: int, end: int):
    _require_log()
    _asset(asset)
    if end <= start:
        raise HTTPException(422, "end must be greater than start")
    lg = STORE.log[asset]
    end = min(end, start + 3000)                              # bounded response size
    return records(lg.loc[(lg.index >= start) & (lg.index < end)])


@app.get("/api/snapshot")
def snapshot(index: int, history: int = 192):
    _require_log()
    history = max(1, min(history, 960))
    out = {}
    for a, lg in STORE.log.items():
        if index not in lg.index:
            raise HTTPException(404, "index outside engine log")
        row = lg.loc[index].to_dict()
        row["attribution"] = json.loads(row["attribution"]) if row.get("attribution") else {}
        out[a] = row
    ev = STORE.log_events
    feed = ev[(ev["index"] <= index) & (ev["index"] > index - history)].tail(80) if len(ev) else ev
    al = STORE.log_alerts
    ts_i = str(STORE.log[next(iter(STORE.log))].loc[index, "timestamp"])
    active = al[(al["index"] <= index) & (al["closed"].isna() | (al["closed"] >= ts_i))] if len(al) else al
    return _clean({"index": index, "assets": out, "feed": records(feed), "active_alerts": records(active.tail(10)) if len(al) else []})


# --------------------------------------------------------------------------- live sessions
def _session(**kw) -> Session:
    return SESSIONS.add(Session(dep(), STORE.dfs, persistence_k=STORE.cfg["threshold"]["persistence_k"],
                                merge_gap=STORE.cfg["threshold"]["merge_gap_steps"], **kw))


@app.post("/api/sessions")
def create_session(body: SessionCreate | None = None):
    _no_rows_in_public()
    body = body or SessionCreate()
    lo, hi = _test_range()
    start = lo if body.start_index is None else body.start_index
    stop = hi if body.stop_index is None else body.stop_index
    if not lo <= start < hi:
        raise HTTPException(400, f"start_index must be inside the held-out test period [{lo}, {hi})")
    if stop <= start:
        raise HTTPException(400, "stop_index must be greater than start_index")
    assets_ = [_asset(a) for a in (body.assets or STORE.assets)]
    return _session(assets=assets_, start=start, stop=min(stop, hi), mode="live").summary()


# ---- visitor-driven sessions: no ETDataset rows are served (allowed on the public deployment) ----
def _frame_session(df: pd.DataFrame, asset: str, mode: str, meta: dict) -> dict:
    d = dep()
    warm = d.art.fe.lookback + d.art.seq_len - 1
    s = SESSIONS.add(Session(d, {asset: df.reset_index(drop=True)}, [asset], start=warm, stop=len(df), mode=mode,
                             incident=meta, persistence_k=STORE.cfg["threshold"]["persistence_k"],
                             merge_gap=STORE.cfg["threshold"]["merge_gap_steps"]))
    return _clean(s.summary())


@app.post("/api/scenarios")
def create_scenario(body: ScenarioBody):
    """Scenario Lab: synthetic normal operation (fitted on aggregate training statistics, no dataset rows)
    plus a visitor-chosen fault injected with the benchmark's simulator; streamed through the real engine."""
    from src.scenario.scenario import build
    d = dep()
    _asset(body.asset)
    warm = d.art.fe.lookback + d.art.seq_len - 1
    df, meta = build(body.asset, body.month, body.fault, body.severity, body.precursor_hours, body.seed, warm)
    return _frame_session(df, body.asset, "scenario", meta | {"label": ASSET_LABEL.get(body.asset, body.asset),
                                                              "type": body.fault, "event_id": None})


@app.get("/api/scenarios/validation")
def scenario_validation():
    """Measured behaviour of the deployed model on synthetic baselines (src/scenario/validate.py)."""
    return _json(R / "final" / "scenario_validation.json", {})


@app.get("/api/samples/synthetic.csv")
def sample_csv(asset: str = "ETTm1", month: int = Query(4, ge=1, le=12), fault: str = "thermal_runaway",
               seed: int = Query(0, ge=0, le=1_000_000)):
    """Synthetic template for the upload feature (same generator as the Scenario Lab; no dataset rows)."""
    from src.scenario.scenario import FAULTS, build
    _asset(asset)
    if fault not in FAULTS:
        raise HTTPException(422, f"fault must be one of {list(FAULTS)}")
    d = dep()
    df, _ = build(asset, month, fault, 1.0, 8, seed, d.art.fe.lookback + d.art.seq_len - 1)
    out = df[["timestamp", *SENSORS]].copy()
    out[SENSORS] = out[SENSORS].round(4)
    csv = out.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="synthetic_{asset}_{fault}.csv"'})


@app.post("/api/uploads")
def create_upload(body: UploadBody):
    """Stream visitor-provided readings through the engine. ``asset`` picks which transformer's training
    statistics are used for scaling (the model was trained per transformer)."""
    d = dep()
    _asset(body.asset)
    need = d.art.fe.lookback + d.art.seq_len
    if len(body.rows) < need:
        raise HTTPException(422, f"need at least {need} consecutive 15-minute rows (history for the first prediction); "
                                 f"got {len(body.rows)}")
    df = pd.DataFrame([r.model_dump() for r in body.rows])
    df["timestamp"] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    df["offline"], df["failure"], df["sim_phase"], df["sim_event_type"] = False, 0, "normal", ""
    return _frame_session(df, body.asset, "upload", {"label": ASSET_LABEL.get(body.asset, body.asset), "type": "upload",
                                                     "rows": len(df), "event_id": None})


# ---- stateless runs (serverless-friendly): the whole stream is computed by the engine in one request; the
# browser then plays the returned records back at the chosen speed. Nothing is generated client-side.
MAX_RUN_ROWS = int(os.environ.get("SGPM_MAX_RUN_ROWS", "2000" if PUBLIC else "5000"))


def _run_to_end(summary: dict) -> dict:
    s = SESSIONS.get(summary["id"])
    t = time.perf_counter()
    out = s.step(s.stop - s.cursor)
    ms = (time.perf_counter() - t) * 1000
    SESSIONS.drop(s.id)
    recs = next(iter(out["records"].values()))
    for r in recs:                                    # trim what the UI does not use
        r.pop("member_scores", None)
        if r.get("timings_ms"):
            r["timings_ms"] = {"total": r["timings_ms"].get("total")}
    return _clean({"session": summary, "records": recs, "events": out["events"], "alerts": out["alerts"],
                   "compute_ms": round(ms, 1), "attribution_method": dep().scorer.attribution_method})


@app.post("/api/scenarios/run")
def run_scenario(body: ScenarioBody):
    """Scenario Lab, stateless: build the scenario and stream it through the engine in one request."""
    return _run_to_end(create_scenario(body))


@app.post("/api/uploads/run")
def run_upload(body: UploadBody):
    """Upload & Score, stateless: stream the visitor's rows through the engine in one request."""
    if len(body.rows) > MAX_RUN_ROWS:
        raise HTTPException(422, f"at most {MAX_RUN_ROWS} rows per run on this deployment; got {len(body.rows)}")
    return _run_to_end(create_upload(body))


@app.post("/api/sessions/{sid}/step")
def step_session(sid: str, body: StepBody | None = None):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "session expired — create a new one")
    out = s.step((body or StepBody()).n)
    for recs in out["records"].values():
        for r in recs:
            if r.get("timings_ms") and r.get("risk") is not None:
                METRICS.observe_engine(r["timings_ms"])
    return _clean(out)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "session expired")
    return _clean(s.summary() | {"feed": list(s.feed)[-200:], "alerts": list(s.alerts.values())})


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    SESSIONS.drop(sid)
    return {"ok": True}


# --------------------------------------------------------------------------- incidents
@app.get("/api/incidents")
def incidents():
    tf = STORE.test_failures()
    cols = [c for c in ["asset", "label", "event_id", "type", "failure_idx", "onset_idx", "failure_time", "onset_time",
                        "class", "signal_strength", "signal_metric", "detected", "lead_time_min", "engine_lead_time_min"] if c in tf]
    return records(tf[cols])


@app.post("/api/incidents/{asset}/{event_id}/replay")
def replay_incident(asset: str, event_id: int, body: ReplayBody | None = None):
    _no_rows_in_public()
    body = body or ReplayBody()
    _asset(asset)
    tf = STORE.test_failures()
    r = tf[(tf.asset == asset) & (tf.event_id == event_id)]
    if not len(r):
        raise HTTPException(404, "incident not found in held-out test period")
    r = r.iloc[0]
    lo, hi = _test_range()
    start = max(lo, int(r.failure_idx - body.hours_before * 4))
    stop = min(hi, int(r.failure_idx + body.hours_after * 4) + 1)
    return _session(assets=[asset], start=start, stop=stop, mode="replay", incident=_clean({k: r[k] for k in r.index})).summary()


# --------------------------------------------------------------------------- batch prediction
@app.post("/api/predict")
def predict(body: PredictBody):
    """Score posted sensor history with the deployed model (same features, calibration, thresholds)."""
    from src.streaming.alerts import AlertPolicy
    from src.streaming.inference import min_rows, predict_frame
    d = dep()
    _asset(body.asset)
    need = min_rows(d.art)
    if len(body.rows) < need:
        raise HTTPException(422, f"need at least {need} consecutive 15-min rows of history (lookback + window); got {len(body.rows)}")
    df = pd.DataFrame([r.model_dump() for r in body.rows])
    t = time.perf_counter()
    raw = predict_frame(d.art, df, body.asset)
    policy = AlertPolicy(d.warn_raw, d.crit_raw, STORE.cfg["threshold"]["persistence_k"])
    out = []
    for ts, p in zip(df.timestamp, raw):
        if np.isnan(p):
            continue
        out.append({"timestamp": str(ts), "raw_score": float(p), "risk": float(d.calibrator([p])[0]), "status": policy.update(float(p))})
    return {"asset": body.asset, "model": d.name, "n_scored": len(out), "first_scored_row": need - 1,
            "latency_ms": round((time.perf_counter() - t) * 1000, 2), "predictions": out}


# --------------------------------------------------------------------------- alerts
def _with_acks(r: dict, acks: dict) -> dict:
    if r["id"] in acks:
        r["status"] = acks[r["id"]]["status"]
        r["ack"] = acks[r["id"]]
    return r


@app.get("/api/alerts")
def alerts():
    acks = STORE.acks()
    out = []
    al = STORE.log_alerts
    tf = STORE.test_failures()
    for r in records(al) if len(al) else []:
        r["id"] = f"log-{r['id']}"
        r["signals"] = json.loads(r["signals"]) if isinstance(r.get("signals"), str) else r.get("signals")
        f = tf[(tf.asset == r["asset"]) & (tf.failure_time >= r["opened"]) &
               (tf.failure_time <= str(pd.Timestamp(r["opened"]) + pd.Timedelta(hours=24)))]
        r["outcome"] = (f"failure followed ({f.iloc[0]['type']}, simulated)" if len(f) else "no failure within 24 h")
        out.append(_with_acks(r, acks))
    for s in SESSIONS.sessions.values():
        for a in s.alerts.values():
            out.append(_with_acks(dict(a) | {"id": f"s-{s.id}-{a['id']}"}, acks))
    return _clean(sorted(out, key=lambda x: x["opened"], reverse=True))


def _check_alert_id(aid: str) -> None:
    if not ALERT_ID.match(aid):
        raise HTTPException(422, "malformed alert id")
    if aid.startswith("log-") and len(STORE.log_alerts) and int(aid[4:]) not in set(STORE.log_alerts["id"].astype(int)):
        raise HTTPException(404, "alert not found")


@app.post("/api/alerts/{aid}/ack")
def ack(aid: str, body: AckBody | None = None):
    _check_alert_id(aid)
    body = body or AckBody()
    state = {"status": "acknowledged", "by": body.by, "note": body.note, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    STORE.set_ack(aid, state)
    return {"id": aid} | state


@app.delete("/api/alerts/{aid}/ack")
def unack(aid: str):
    _check_alert_id(aid)
    STORE.clear_ack(aid)
    return {"id": aid, "status": "open"}


# --------------------------------------------------------------------------- analytics / models / methodology
@app.get("/api/analytics/failures")
def failure_analytics():
    fm = STORE.final_metrics
    det = STORE.detectability
    tf = STORE.test_failures()
    fa = STORE.false_alarms
    by_type_all = STORE.events[STORE.events.kind == "failure"].groupby("type").size().to_dict()
    lead = tf["engine_lead_time_min"].dropna().astype(float) if "engine_lead_time_min" in tf else pd.Series(dtype=float)
    return _clean({
        "test_failures": records(tf.drop(columns=["params"], errors="ignore")),
        "failure_types_all": by_type_all,
        "detectability": _json(R / "phase2" / "detectability_summary.json", {}),
        "detectability_table": records(det[det.split.isin(["val", "test"])]) if len(det) else [],
        "false_alarms": records(fa) if len(fa) else [],
        "false_alarm_episodes": _json(R / "final" / "false_alarms.json", []),
        "lead_time": {"values_min": lead.tolist(), "median": float(lead.median()) if len(lead) else None,
                      "mean": float(lead.mean()) if len(lead) else None},
        "monthly": fm.get("generalisation_per_month", []),
        "test_summary": fm.get("test", {}), "val_summary": fm.get("val", {}),
        "final": _json(R / "final" / "final_metrics.json", {}),
        "operating_point": {k: v for k, v in _json(R / "final" / "operating_point.json", {}).items()
                            if k in ("old", "new", "validation", "protocol")},
    })


@app.get("/api/models")
def models():
    out = {}
    eff = R / "efficiency_table.csv"
    out["efficiency"] = records(pd.read_csv(eff)) if eff.exists() else []
    fc = PROJECT_ROOT / "experiments" / "final_candidates.csv"
    out["phase1_candidates"] = records(pd.read_csv(fc)) if fc.exists() else []
    res = pd.read_csv(PROJECT_ROOT / "experiments" / "results.csv")
    b = res[(res.family == "baseline") & (res.prediction_horizon == 8)].drop_duplicates(["model", "features"], keep="last")
    out["baselines"] = records(b[["model", "features", "val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1",
                                  "event_recall_detectable", "lead_time_median_min", "false_alarms_per_day",
                                  "inference_latency_ms", "model_size_kb"]])
    out["phase2_summary"] = _json(R / "phase2" / "experiments_summary.json", [])
    out["ablation"] = _json(R / "phase2" / "ablation.json", [])
    out["calibration"] = _json(R / "phase2" / "calibration_uncertainty.json", {})
    out["protocol"] = _json(R / "final" / "protocol_summary.json", [])
    out["robustness"] = _json(R / "final" / "robustness.json", {})
    out["confidence_bins"] = _json(R / "final" / "confidence_bins.json", {})
    out["final_test"] = _json(R / "final" / "final_metrics.json", {}).get("test", {})   # deployed operating point
    return _clean(out)


@app.get("/api/patterns")
def patterns():
    """Historical pattern comparison: for each primary model-attributed signal, how often did engine-log
    alarms with that signal precede a (simulated) failure within 24 h?"""
    al = alerts()
    rows: dict[str, dict] = {}
    for a in al:
        if not a.get("id", "").startswith("log-") or not a.get("signals"):
            continue
        sig = a["signals"][0]["signal"]
        r = rows.setdefault(sig, {"signal": sig, "alarms": 0, "followed_by_failure": 0})
        r["alarms"] += 1
        r["followed_by_failure"] += int(str(a.get("outcome", "")).startswith("failure"))
    for r in rows.values():
        r["share_followed"] = r["followed_by_failure"] / r["alarms"]
    return sorted(rows.values(), key=lambda r: -r["alarms"])


@app.get("/api/methodology")
def methodology():
    d = dep()
    cfg = STORE.cfg
    return _clean({
        "disclosure": DISCLOSURE,
        "pipeline": ["Sensor ingestion", "Causal preprocessing", "Temporal features", f"{d.art.seq_len * 15 / 60:.0f}-hour sequence",
                     "GRU + attention", "Failure probability", "Calibration", "Confidence", "Risk engine", "Alert"],
        "facts": {"dataset": "ETTm1 + ETTm2 (ETDataset)", "sampling": "15 minutes",
                  "input_window": f"{d.art.seq_len * 15 / 60:.0f} hours", "horizon": f"{d.horizon_min // 60} hours",
                  "features": f"{len(d.art.feature_names)} causal features ({d.art.meta['feature_set']})",
                  "model": f"{MODEL_NAMES.get(d.art.meta['model_type'], d.art.meta['model_type'])} "
                           f"({d.parameters:,} parameters)",
                  "split": "chronological 70/15/15 per asset with embargo",
                  "threshold_rule": "max F1 subject to recall ≥ 0.70 on pooled fresh validation simulations; alarm after 2 consecutive steps",
                  "model_selection": "mean PR-AUC over 3 seeds × 5 fresh validation simulations (test never used)"},
        "simulation": cfg["simulation"],
        "leakage": _json(R / "leakage_report.json", {}),
        "reproduction": _json(R / "phase2" / "reproduction.json", {}).get("comparison", []),
        "shift": _json(R / "phase2" / "distribution_shift.json", {}),
        "calibration": {k: v for k, v in _json(R / "phase2" / "calibration_uncertainty.json", {}).items()
                        if k in ("chosen", "cv_selection", "uncertainty", "confidence_distribution")},
        "health_weights": HEALTH_WEIGHTS,
        "final": _json(R / "final" / "final_metrics.json", {}),
        "operating_point": {k: v for k, v in _json(R / "final" / "operating_point.json", {}).items()
                            if k in ("old", "new", "validation", "protocol")},
    })


# --------------------------------------------------------------------------- built frontend (production)
DIST = (PROJECT_ROOT / "frontend" / "dist").resolve()
if (DIST / "_app").exists():
    app.mount("/_app", StaticFiles(directory=DIST / "_app"), name="app-static")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "unknown API route")
    dist = DIST                                   # module attribute, read per request (tests point it elsewhere)
    if not (dist / "index.html").is_file():
        raise HTTPException(404, "frontend not built: run `cd frontend && npm run build` (or use `make demo`)")
    f = (dist / path).resolve()
    # only serve files that live inside dist/ (prevents ../ path traversal); else the SPA shell
    if path and f.is_file() and f.is_relative_to(dist):
        return FileResponse(f)
    # the shell must always be revalidated, otherwise browsers keep running an old bundle after a deploy
    return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
