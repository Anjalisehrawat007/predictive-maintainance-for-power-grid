"""Streaming inference: one observation in -> probability + status out.

    new observation -> ring buffer (lookback rows) -> causal features for the
    newest row -> scaled feature ring buffer (seq_len rows) -> model -> prob
    -> alert policy -> NORMAL / WARNING / CRITICAL
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pandas as pd
import torch

from src.streaming.alerts import OFFLINE, WARMUP, AlertPolicy
from src.streaming.artifacts import Artifacts

STEP_MIN = 15


class StreamingPredictor:
    def __init__(self, art: Artifacts, asset: str, policy: AlertPolicy, merge_gap: int = 4):
        self.art, self.asset, self.policy = art, asset, policy
        self.sensor_cols = art.fe.sensor_cols
        self.raw = deque(maxlen=art.fe.lookback)
        self.feat = deque(maxlen=art.seq_len)
        self.merge_gap = merge_gap
        self._episode_start = None
        self._last_alarm = None
        self._t = 0
        torch.set_num_threads(1)

    @property
    def ready(self) -> bool:
        return len(self.feat) == self.art.seq_len

    def update(self, timestamp, values: dict, offline: bool = False) -> dict:
        """Ingest one observation. ``offline`` = breaker-open flag from SCADA (metadata, not a model input)."""
        t0 = time.perf_counter()
        self.raw.append({"timestamp": timestamp, **{c: float(values[c]) for c in self.sensor_cols}})
        if len(self.raw) == self.raw.maxlen:
            df = pd.DataFrame(self.raw)
            self.feat.append(self.art.features(df, self.asset)[-1])
        out = {"timestamp": timestamp, "failure_probability": np.nan, "status": WARMUP, "alarm": False}
        if self.ready:
            x = torch.from_numpy(np.asarray(self.feat, dtype=np.float32))[None]
            with torch.no_grad():
                prob = float(torch.sigmoid(self.art.model(x))[0])
            status = self.policy.update(prob)
            out.update(failure_probability=prob, status=status, alarm=status in ("WARNING", "CRITICAL"))
        if offline:
            out["status"], out["alarm"] = OFFLINE, False
            self.policy.reset()
        # track the active warning episode (for online lead-time)
        if out["alarm"]:
            if self._last_alarm is None or self._t - self._last_alarm - 1 > self.merge_gap:
                self._episode_start = timestamp
            self._last_alarm = self._t
        out["warning_since"] = (self._episode_start if self._last_alarm is not None
                                and self._t - self._last_alarm <= self.merge_gap else None)
        out["latency_ms"] = (time.perf_counter() - t0) * 1000
        self._t += 1
        return out


def replay(art: Artifacts, df: pd.DataFrame, asset: str, policy: AlertPolicy, horizon: int,
           start: int, stop: int, merge_gap: int = 4, speed: float = 0.0, log_every: int = 0) -> pd.DataFrame:
    """Replay historical rows [start, stop) as if they arrived in real time."""
    from src.data.dataset import make_labels

    y = make_labels(df["failure"].to_numpy(), horizon)
    sp = StreamingPredictor(art, asset, policy, merge_gap)
    rows = []
    for i in range(start, stop):
        r = df.iloc[i]
        o = sp.update(r["timestamp"], r, offline=bool(r["offline"]))
        lead = np.nan
        if r["failure"] == 1 and o["warning_since"] is None and rows and rows[-1]["warning_since"] is not None:
            o["warning_since"] = rows[-1]["warning_since"]
        if r["failure"] == 1 and o["warning_since"] is not None:
            lead = (pd.Timestamp(r["timestamp"]) - pd.Timestamp(o["warning_since"])).total_seconds() / 60
        rows.append(o | {"asset": asset, "predicted_failure": int(o["alarm"]),
                         "actual_failure": y[i], "failure_event": int(r["failure"]),
                         "lead_time_min": lead, "offline": bool(r["offline"]),
                         **{c: r[c] for c in art.fe.sensor_cols}})
        if log_every and (i - start) % log_every == 0:
            print(f"{r['timestamp']}  p={o['failure_probability']:.3f}  {o['status']}  ({o['latency_ms']:.1f} ms)")
        if speed:
            time.sleep(speed)
    return pd.DataFrame(rows)
