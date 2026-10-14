"""Operational intelligence engine: the real streaming model + calibration, confidence,
attribution, risk/alert logic and per-stage timing.

Every number this module emits is either (a) a sensor value from the dataset, (b) model
output, or (c) a *derived* indicator whose formula is documented here and in the UI
(health score, trend).  Simulation ground truth (failure flags, precursor phase) is passed
through separately and labelled as such; it never enters the model.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.explainability.groups import feature_group
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import Artifacts, deployment_manifest, load_artifacts, load_artifacts_lite
from src.streaming.calibration import ServingCalibrator
from src.utils.config import PROJECT_ROOT, load_config

# ONNX-only serving (e.g. a serverless deployment): no PyTorch; attribution by occlusion instead of
# integrated gradients. Selected with SGPM_ENGINE=onnx, or automatically when PyTorch is not installed.
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover - exercised in the serverless build
    HAVE_TORCH = False
ENGINE = os.environ.get("SGPM_ENGINE", "torch" if HAVE_TORCH else "onnx")

SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
GROUP_LABEL = {"OT": "Oil temperature", "thermal residual (OT vs load)": "Thermal residual (OT vs load)",
               "HUFL": "HV useful load", "HULL": "HV reactive load", "MUFL": "MV useful load",
               "MULL": "MV reactive load", "LUFL": "LV useful load", "LULL": "LV reactive load",
               "total load": "Total load ramp", "calendar (hour/day)": "Time-of-day context"}
HEALTH_WEIGHTS = {"failure_risk": 0.4, "thermal_stability": 0.2, "load_stability": 0.2, "sensor_quality": 0.1,
                  "model_confidence": 0.1}
EVENT_RULES = {"thermal_z": 2.5, "resid_sigma": 2.5, "volratio_q": 0.99, "spikes": 2, "risk_delta": 0.05}


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


class TorchScorer:
    """Reference scorer: ONNX (if verified identical) or PyTorch for the score, PyTorch seed members,
    integrated-gradients attribution."""
    attribution_method = "integrated gradients"

    def __init__(self, model, members, onnx):
        self.model, self.members, self.onnx = model, members, onnx

    def logit(self, x: np.ndarray) -> float:
        import torch
        if self.onnx is not None:
            return float(self.onnx.run(None, {"x": x})[0][0])
        with torch.no_grad():
            return float(self.model(torch.from_numpy(x))[0])

    def member_probs(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(x)
            return np.array([float(torch.sigmoid(m(t))[0]) for m in self.members])

    def attribution(self, x: np.ndarray, groups: dict, steps: int = 12) -> np.ndarray:
        """Signed (L, F) integrated-gradients attribution."""
        import torch
        xt = torch.from_numpy(x)
        alphas = torch.linspace(1 / steps, 1, steps)[:, None, None]
        xi = (alphas * xt).requires_grad_(True)
        g, = torch.autograd.grad(self.model(xi).sum(), xi)
        return (xt[0] * g.mean(0)).detach().numpy()


class OnnxScorer:
    """Dependency-light scorer (no PyTorch): ONNX Runtime for the model and the seed members.
    Attribution by occlusion: the drop in the model's logit when a sensor group (or one time step) is replaced by
    its training mean (0 after scaling). Computed in one batched ONNX call."""
    attribution_method = "occlusion"

    def __init__(self, sess, member_sessions):
        self.sess, self.member_sessions = sess, member_sessions

    def logit(self, x: np.ndarray) -> float:
        return float(self.sess.run(None, {"x": x})[0][0])

    def member_probs(self, x: np.ndarray) -> np.ndarray:
        return np.array([float(_sigmoid(m.run(None, {"x": x})[0][0])) for m in self.member_sessions])

    def attribution(self, x: np.ndarray, groups: dict) -> np.ndarray:
        L, F = x.shape[1], x.shape[2]
        names = list(groups)
        batch = [x[0]]
        for g in names:
            v = x[0].copy(); v[:, groups[g]] = 0.0; batch.append(v)
        for t in range(L):
            v = x[0].copy(); v[t, :] = 0.0; batch.append(v)
        out = self.sess.run(None, {"x": np.stack(batch).astype(np.float32)})[0].reshape(-1)
        dg, dt = out[0] - out[1:1 + len(names)], out[0] - out[1 + len(names):]     # logit drop per group / step
        # spread each group's contribution over its features, scaled by each time step's share (an (L, F) map
        # with the same group sums and time profile as the occlusion results)
        tw = np.abs(dt) / (np.abs(dt).sum() + 1e-12)
        attr = np.zeros((L, F))
        for g, d in zip(names, dg):
            idx = groups[g]
            attr[:, idx] = (d / len(idx)) * tw[:, None]
        return attr


@dataclass
class Deployment:
    name: str
    art: Artifacts
    members: list
    calibrator: ServingCalibrator
    warn_raw: float
    crit_raw: float
    onnx: object | None
    horizon_min: int
    group_index: dict[str, list[int]]
    asset_stats: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    scorer: object = None

    @property
    def parameters(self) -> int:
        if self.art.model is not None:
            return int(sum(p.numel() for p in self.art.model.parameters()))
        return int(json.loads((PROJECT_ROOT / "models" / "onnx_manifest.json").read_text())["parameters"])

    @property
    def warn_cal(self):
        return float(self.calibrator([self.warn_raw])[0])

    @property
    def crit_cal(self):
        return float(self.calibrator([self.crit_raw])[0])


def _onnx_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def load_deployment(cfg: dict | None = None, engine: str | None = None) -> Deployment:
    """Model served by the platform: models/phase2/deployment.json if present, else Phase-1 selection."""
    cfg = cfg or load_config()
    manifest = deployment_manifest(cfg)
    mdir = PROJECT_ROOT / "models"
    cal_path = mdir / "phase2" / manifest.get("calibrator", "calibrator_phase1_selected.json")
    cal = ServingCalibrator.from_dict(json.loads(cal_path.read_text())) if cal_path.exists() else ServingCalibrator()
    if (engine or ENGINE) == "onnx":
        art = load_artifacts_lite(cfg)
        meta = art.meta
        sess = _onnx_session(mdir / f"{meta['active_model']}.onnx")
        msess = [_onnx_session(mdir / f"ensemble_member{i}.onnx") for i in range(3) if (mdir / f"ensemble_member{i}.onnx").exists()]
        members, onnx, scorer = msess, sess, OnnxScorer(sess, msess)
    else:
        import torch

        from src.models.recurrent import build_model
        art = load_artifacts(cfg, manifest.get("primary_model_file"))
        meta = art.meta
        members = []
        for i in range(3):
            f = mdir / f"ensemble_member{i}.pt"
            if f.exists():
                m = build_model(meta["model_type"], len(art.feature_names), meta["hidden_size"], meta["num_layers"],
                                meta["dropout"])
                m.load_state_dict(torch.load(f, map_location="cpu"))
                members.append(m.eval())
        onnx = None
        onnx_path = mdir / f"{meta['active_model']}.onnx"
        if onnx_path.exists():
            try:
                sess = _onnx_session(onnx_path)
                x = np.random.randn(2, art.seq_len, len(art.feature_names)).astype(np.float32)
                with torch.no_grad():
                    ref = art.model(torch.from_numpy(x)).numpy()
                if np.allclose(sess.run(None, {"x": x})[0], ref, atol=1e-4):
                    onnx = sess                       # only used if numerically identical to PyTorch
            except Exception:
                onnx = None
        scorer = TorchScorer(art.model, members, onnx)
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(art.feature_names):
        groups.setdefault(feature_group(n), []).append(j)
    pol = AlertPolicy.from_config(cfg, meta, manifest)
    warn, crit = pol.warning_threshold, pol.critical_threshold
    dep = Deployment(meta["active_model"], art, members, cal, warn, crit, onnx,
                     meta["horizon"] * 15, groups, manifest=manifest, scorer=scorer)
    dep.asset_stats = _asset_stats(cfg, art)
    return dep


HEALTH_REF = PROJECT_ROOT / "configs" / "health_reference.json"


def _asset_stats(cfg, art) -> dict:
    """Training-period reference quantiles for the derived health indicator (train rows only).
    Computed from the processed data when available and cached as aggregate JSON, so a deployment without
    dataset rows (public mode) can still compute the health indicator."""
    from src.data.dataset import load_processed
    from src.preprocessing.splits import compute_bounds
    proc = PROJECT_ROOT / "data" / "processed"
    public = os.environ.get("SGPM_PUBLIC") == "1"          # public deployment never reads dataset rows
    if public or not all((proc / f"{a}.parquet").exists() for a in cfg["dataset"]["assets"]):
        return json.loads(HEALTH_REF.read_text())
    dfs, _ = load_processed(cfg)
    out = {}
    for a, df in dfs.items():
        b = compute_bounds(len(df), cfg, art.fe.lookback)
        F = art.fe.transform(df.iloc[: b.train_end], a)
        ok = ~df["offline"].iloc[: b.train_end].to_numpy()
        rs = float(F["OT_resid"][ok].std())
        th = np.maximum(F["OT_zbase"].abs(), F["OT_resid"].abs() / rs)[ok].dropna()
        lv = F[["HUFL_volratio", "MUFL_volratio", "HULL_volratio"]].max(axis=1)[ok].dropna()
        out[a] = {"resid_sigma": rs, "thermal_q50": float(th.quantile(.5)), "thermal_q99": float(th.quantile(.99)),
                  "load_q50": float(lv.quantile(.5)), "load_q99": float(lv.quantile(.99))}
    HEALTH_REF.write_text(json.dumps(out, indent=1))
    return out


class AssetStream:
    """One transformer's live state. ``update`` = one new observation."""

    def __init__(self, dep: Deployment, asset: str, persistence_k: int = 2, merge_gap: int = 4):
        self.dep, self.asset = dep, asset
        self.raw = deque(maxlen=dep.art.fe.lookback)
        self.feat = deque(maxlen=dep.art.seq_len)
        self.policy = AlertPolicy(dep.warn_raw, dep.crit_raw, persistence_k)
        self.merge_gap = merge_gap
        self.risk_hist = deque(maxlen=8)
        self.t = 0
        self.last_alarm_t = None
        self.warning_since = None
        self.prev = {"status": "WARMUP", "risk_reported": None, "thermal": False, "resid": False, "load": False,
                     "spikes": False, "offline": False}

    # ---------------------------------------------------------------- helpers
    def _attribution(self, x: np.ndarray) -> tuple[dict, list]:
        attr = self.dep.scorer.attribution(x, self.dep.group_index)          # signed (L, F)
        mag = np.abs(attr)
        tot = mag.sum() + 1e-12
        shares = {GROUP_LABEL.get(gname, gname): float(mag[:, idx].sum() / tot)
                  for gname, idx in self.dep.group_index.items()}
        signed = {GROUP_LABEL.get(gname, gname): float(attr[:, idx].sum()) for gname, idx in self.dep.group_index.items()}
        temporal = (mag.sum(1) / tot).tolist()
        sig = sorted(({"signal": k, "share": v, "direction": "raises risk" if signed[k] > 0 else "lowers risk"}
                      for k, v in shares.items()), key=lambda d: -d["share"])
        return {"signals": sig, "temporal": temporal, "method": self.dep.scorer.attribution_method}, sig[:2]

    def _health(self, F: pd.Series, risk: float, conf: float) -> dict:
        s = self.dep.asset_stats[self.asset]
        th = max(abs(F["OT_zbase"]), abs(F["OT_resid"]) / s["resid_sigma"])
        lv = max(F["HUFL_volratio"], F["MUFL_volratio"], F["HULL_volratio"])
        spikes = float(sum(F[f"{c}_spikes16"] for c in SENSORS if f"{c}_spikes16" in F))
        flat = any(len(set(list(r[c] for r in list(self.raw)[-8:]))) == 1 for c in SENSORS) if len(self.raw) >= 8 else False
        comp = {
            "thermal_stability": 100 * (1 - _clip01((th - s["thermal_q50"]) / (s["thermal_q99"] - s["thermal_q50"] + 1e-9))),
            "load_stability": 100 * (1 - _clip01((lv - s["load_q50"]) / (s["load_q99"] - s["load_q50"] + 1e-9))),
            "sensor_quality": 100 * (1 - _clip01(spikes / 10 + (0.5 if flat else 0))),
            "failure_risk": 100 * (1 - _clip01(risk)),
            "model_confidence": 100 * _clip01(conf),
        }
        total = sum(HEALTH_WEIGHTS[k] * v for k, v in comp.items())
        return {"score": round(total, 1), "components": {k: round(v, 1) for k, v in comp.items()},
                "inputs": {"thermal_deviation": round(float(th), 2), "load_volatility": round(float(lv), 2),
                           "spike_count_4h": spikes, "flatline": bool(flat)}}

    def _events(self, ts, F, risk, status, offline, idx) -> list[dict]:
        ev, p, s = [], self.prev, self.dep.asset_stats[self.asset]
        a = self.asset

        def add(kind, sev, msg, **kw):
            ev.append({"ts": ts, "asset": a, "kind": kind, "severity": sev, "message": msg, "index": idx} | kw)
        if offline != p["offline"]:
            add("offline" if offline else "online", "critical" if offline else "info",
                "Asset offline — breaker open (trip)" if offline else "Asset back online")
        if F is not None and not offline:
            th = abs(F["OT_zbase"]) >= EVENT_RULES["thermal_z"]
            rs = F["OT_resid"] / s["resid_sigma"] >= EVENT_RULES["resid_sigma"]
            ld = max(F["HUFL_volratio"], F["MUFL_volratio"]) >= s["load_q99"]
            sp = (F["HULL_spikes16"] + F["MULL_spikes16"]) >= EVENT_RULES["spikes"]
            if th and not p["thermal"]:
                add("thermal_deviation", "notice", f"Thermal deviation: OT {F['OT_zbase']:+.1f} σ vs 24 h baseline")
            if rs and not p["resid"]:
                add("thermal_residual", "notice", f"Oil temperature {F['OT_resid'] / s['resid_sigma']:.1f} σ above load-expected level")
            if ld and not p["load"]:
                add("load_instability", "notice", "Load instability: volatility above the asset's 99th percentile")
            if sp and not p["spikes"]:
                add("reactive_spikes", "notice", f"Reactive-load spike burst ({int(F['HULL_spikes16'] + F['MULL_spikes16'])} in 4 h)")
            p.update(thermal=th, resid=rs, load=ld, spikes=sp)
        if risk is not None and (p["risk_reported"] is None or abs(risk - p["risk_reported"]) >= EVENT_RULES["risk_delta"]):
            if p["risk_reported"] is not None:
                add("risk_update", "info", f"Risk updated {p['risk_reported'] * 100:.0f}% → {risk * 100:.0f}%",
                    old=p["risk_reported"], new=risk)
            p["risk_reported"] = risk
        if status != p["status"] and p["status"] != "WARMUP":
            if status == "WARNING":
                add("warning", "warning", "Early-warning threshold crossed")
            elif status == "CRITICAL":
                add("critical", "critical", "Critical threshold crossed")
            elif status == "NORMAL" and p["status"] in ("WARNING", "CRITICAL"):
                add("cleared", "info", "Risk back below warning threshold")
        p.update(status=status, offline=offline)
        return ev

    def prime(self, df: pd.DataFrame, n_tail: int = 8) -> None:
        """Fast warm-up from the rows preceding the start: one vectorised feature pass fills the
        buffers (exact, since features only need ``lookback`` rows), then the last ``n_tail`` rows
        go through ``update`` so alarm persistence, trend and event state are real."""
        art = self.dep.art
        head = df.iloc[: max(0, len(df) - n_tail)]
        if len(head):
            Fall = art.fe.transform(head, self.asset)
            Z = art.scaler.transform(Fall[art.scaler_columns].to_numpy(np.float64), self.asset)
            if art.feature_names != art.scaler_columns:
                Z = Z[:, [art.scaler_columns.index(c) for c in art.feature_names]]
            ok = Fall.notna().all(axis=1).to_numpy()
            for i in np.where(ok)[0][-(art.seq_len - 1):] if ok.any() else []:
                self.feat.append(Z[i])
            for r in head.tail(self.raw.maxlen).itertuples():
                self.raw.append({"timestamp": r.timestamp, **{c: float(getattr(r, c)) for c in SENSORS}})
        for r in df.iloc[len(head):].itertuples():
            self.update(r.timestamp, {c: getattr(r, c) for c in SENSORS}, bool(r.offline), -1, emit=False)

    # ---------------------------------------------------------------- main
    def update(self, ts, values: dict, offline: bool = False, idx: int = -1, emit: bool = True) -> tuple[dict, list]:
        T = {}
        t0 = time.perf_counter()
        self.raw.append({"timestamp": ts, **{c: float(values[c]) for c in SENSORS}})
        T["ingest"] = time.perf_counter() - t0
        rec = {"index": idx, "timestamp": str(ts), "asset": self.asset, "sensors": {c: float(values[c]) for c in SENSORS},
               "offline": bool(offline), "status": "WARMUP", "raw_score": None, "risk": None, "confidence": None}
        F = None
        if len(self.raw) == self.raw.maxlen:
            t = time.perf_counter()
            fe, cols = self.dep.art.fe, self.dep.art.scaler_columns
            if fe.robust or fe.adaptive_window:            # experimental feature sets: full pandas path
                Fall = fe.transform(pd.DataFrame(self.raw), self.asset)
                F = Fall.iloc[-1]
            else:                                          # fast exact path: newest row only (NumPy)
                F = fe.transform_last(np.array([[r[c] for c in SENSORS] for r in self.raw]), ts, self.asset)
            z = self.dep.art.scaler.transform(np.array([[F[c] for c in cols]], dtype=np.float64), self.asset)
            if self.dep.art.feature_names != cols:
                z = z[:, [cols.index(c) for c in self.dep.art.feature_names]]
            T["features"] = time.perf_counter() - t
            t = time.perf_counter()
            self.feat.append(z[0])
            T["sequence"] = time.perf_counter() - t
        if len(self.feat) == self.dep.art.seq_len:
            t = time.perf_counter()
            x = np.asarray(self.feat, dtype=np.float32)[None]
            raw = float(_sigmoid(self.dep.scorer.logit(x)))
            T["inference"] = time.perf_counter() - t
            t = time.perf_counter()
            mp = self.dep.scorer.member_probs(x) if self.dep.members else np.array([raw])
            conf = float(np.clip(1 - 2 * mp.std(), 0, 1))
            T["confidence"] = time.perf_counter() - t
            t = time.perf_counter()
            risk = float(self.dep.calibrator([raw])[0])
            T["calibration"] = time.perf_counter() - t
            t = time.perf_counter()
            expl, top = self._attribution(x)
            T["explain"] = time.perf_counter() - t
            t = time.perf_counter()
            status = self.policy.update(raw)
            if offline:
                status = "OFFLINE"
                self.policy.reset()
            self.risk_hist.append(risk)
            delta = risk - self.risk_hist[0] if len(self.risk_hist) >= 5 else 0.0
            trend = "rising" if delta > 0.03 else "falling" if delta < -0.03 else "stable"
            alarm = status in ("WARNING", "CRITICAL")
            if alarm:
                if self.last_alarm_t is None or self.t - self.last_alarm_t - 1 > self.merge_gap:
                    self.warning_since = str(ts)
                self.last_alarm_t = self.t
            active = self.last_alarm_t is not None and self.t - self.last_alarm_t <= self.merge_gap
            health = self._health(F, risk, conf)
            rec.update(raw_score=float(raw), risk=risk, confidence=conf, member_scores=mp.tolist(), status=status,
                       alarm=alarm, trend=trend, trend_delta_1h=float(delta), warning_since=self.warning_since if active else None,
                       warning_window_min=self.dep.horizon_min if alarm else None,
                       signals=top, attribution=expl["signals"], temporal_attribution=expl["temporal"], health=health)
            T["risk_engine"] = time.perf_counter() - t
        elif offline:
            rec["status"] = "OFFLINE"
        t = time.perf_counter()
        # monitor state is updated during warm-up too (so a pre-existing deviation is not reported
        # as "new" at the first live step); events are only emitted when ``emit`` is set
        events = self._events(str(ts), F, rec["risk"], rec["status"], bool(offline), idx)
        if not emit:
            events = []
        T["alert_engine"] = time.perf_counter() - t
        T["total"] = time.perf_counter() - t0
        rec["timings_ms"] = {k: round(v * 1000, 4) for k, v in T.items()}
        self.t += 1
        return rec, events
