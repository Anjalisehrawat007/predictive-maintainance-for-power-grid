"""Structured logging, request IDs and in-process latency metrics for the API."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict, deque

import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_log = logging.getLogger("sgpm.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        d.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            d["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(d)


def setup_logging(level: str = "INFO") -> logging.Logger:
    if not _log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        _log.addHandler(h)
        _log.setLevel(level)
        _log.propagate = False
    return _log


def log_event(msg: str, level: int = logging.INFO, **fields) -> None:
    _log.log(level, msg, extra={"extra_fields": fields})


class Metrics:
    """Rolling latency windows per route template (last 2,000 requests each)."""

    def __init__(self, window: int = 2000):
        self.lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.count: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.engine_steps: deque = deque(maxlen=5000)
        self.started = time.time()
        self.lock = threading.Lock()

    def observe(self, route: str, ms: float, status: int) -> None:
        with self.lock:
            self.lat[route].append(ms)
            self.count[route] += 1
            if status >= 500:
                self.errors[route] += 1

    def observe_engine(self, timings_ms: dict) -> None:
        with self.lock:
            self.engine_steps.append(timings_ms)

    @staticmethod
    def _pct(v) -> dict:
        a = np.asarray(v, dtype=float)
        if not len(a):
            return {}
        return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}

    def snapshot(self) -> dict:
        with self.lock:
            routes = {r: {"requests": self.count[r], "errors_5xx": self.errors[r], "latency_ms": self._pct(v)}
                      for r, v in self.lat.items()}
            eng = {}
            if self.engine_steps:
                keys = set().union(*self.engine_steps)
                eng = {k: self._pct([t[k] for t in self.engine_steps if k in t]) for k in sorted(keys)}
        return {"uptime_s": round(time.time() - self.started, 1), "routes": routes, "engine_step_ms": eng}


METRICS = Metrics()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Adds/propagates X-Request-ID, measures latency, logs one structured line per API request."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            ms = (time.perf_counter() - t0) * 1000
            route = request.scope.get("route")
            tmpl = getattr(route, "path", request.url.path)
            if request.url.path.startswith("/api"):
                METRICS.observe(f"{request.method} {tmpl}", ms, status)
                log_event("request", method=request.method, path=tmpl, status=status, ms=round(ms, 2), request_id=rid)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time-ms"] = f"{ms:.2f}"
        return response
