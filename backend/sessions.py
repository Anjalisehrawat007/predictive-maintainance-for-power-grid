"""Streaming sessions: live replay of the held-out period through the real engine."""
from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import deque

import pandas as pd

from backend.intelligence import AssetStream, Deployment

_alert_ids = itertools.count(1)


class Session:
    def __init__(self, dep: Deployment, dfs: dict[str, pd.DataFrame], assets: list[str], start: int, stop: int,
                 mode: str = "live", incident: dict | None = None, persistence_k: int = 2, merge_gap: int = 4):
        self.id = uuid.uuid4().hex[:10]
        self.dep, self.dfs, self.assets, self.mode, self.incident = dep, dfs, assets, mode, incident
        self.start, self.stop, self.cursor = start, stop, start
        self.streams = {a: AssetStream(dep, a, persistence_k, merge_gap) for a in assets}
        self.feed: deque = deque(maxlen=1500)
        self.alerts: dict[int, dict] = {}
        self.open_alert: dict[str, int | None] = {a: None for a in assets}
        self.lock = threading.Lock()
        self.created = time.time()
        self.warmup_ms = self._warmup()

    def _warmup(self) -> float:
        """Fill the buffers with the rows *preceding* the start (no events emitted)."""
        t = time.perf_counter()
        warm = self.dep.art.fe.lookback + self.dep.art.seq_len - 1
        for a in self.assets:
            df = self.dfs[a]
            self.streams[a].prime(df.iloc[max(0, self.start - warm): self.start])
        return (time.perf_counter() - t) * 1000

    def _ground_truth(self, a: str, i: int) -> dict:
        r = self.dfs[a].iloc[i]
        return {"failure_event": int(r["failure"]), "sim_phase": str(r["sim_phase"]), "sim_event_type": str(r["sim_event_type"])}

    def _alerts(self, rec: dict) -> list[dict]:
        a, changed = rec["asset"], []
        oid = self.open_alert[a]
        if rec["status"] in ("WARNING", "CRITICAL"):
            if oid is None:
                aid = next(_alert_ids)
                self.alerts[aid] = {"id": aid, "asset": a, "severity": rec["status"], "opened": rec["timestamp"],
                                    "updated": rec["timestamp"], "risk": rec["risk"], "raw_score": rec["raw_score"],
                                    "confidence": rec["confidence"], "warning_window_min": rec["warning_window_min"],
                                    "signals": rec["signals"], "status": "open", "index": rec["index"],
                                    "source": "live session", "session": self.id}
                self.open_alert[a] = aid
                changed.append(self.alerts[aid])
            else:
                al = self.alerts[oid]
                if rec["status"] == "CRITICAL" and al["severity"] != "CRITICAL":
                    al["severity"] = "CRITICAL"
                if rec["risk"] >= al["risk"]:
                    al.update(risk=rec["risk"], raw_score=rec["raw_score"], confidence=rec["confidence"], signals=rec["signals"])
                al["updated"] = rec["timestamp"]
                changed.append(al)
        elif oid is not None and rec["status"] in ("NORMAL", "OFFLINE"):
            al = self.alerts[oid]
            if al["status"] == "open":
                al["status"] = "resolved"
            al["closed"] = rec["timestamp"]
            al["closed_by"] = "asset tripped (failure)" if rec["status"] == "OFFLINE" else "risk cleared"
            self.open_alert[a] = None
            changed.append(al)
        return changed

    def step(self, n: int = 1) -> dict:
        with self.lock:
            recs, events, alerts = {a: [] for a in self.assets}, [], []
            for _ in range(n):
                if self.cursor >= self.stop:
                    break
                i = self.cursor
                for a in self.assets:
                    r = self.dfs[a].iloc[i]
                    rec, ev = self.streams[a].update(r["timestamp"], r, bool(r["offline"]), i)
                    rec["ground_truth"] = self._ground_truth(a, i)
                    if rec["ground_truth"]["failure_event"]:
                        ev.append({"ts": rec["timestamp"], "asset": a, "kind": "failure", "severity": "critical",
                                   "message": "FAILURE (simulated ground-truth event)", "index": i})
                        if rec.get("warning_since") is None and recs[a] and recs[a][-1].get("warning_since"):
                            rec["warning_since"] = recs[a][-1]["warning_since"]
                        st = self.streams[a]
                        if rec.get("warning_since") is None and st.warning_since and st.last_alarm_t is not None \
                                and st.t - 1 - st.last_alarm_t <= st.merge_gap + 1:
                            rec["warning_since"] = st.warning_since
                        if rec.get("warning_since"):
                            rec["lead_time_min"] = (pd.Timestamp(rec["timestamp"]) - pd.Timestamp(rec["warning_since"])).total_seconds() / 60
                            ev.append({"ts": rec["timestamp"], "asset": a, "kind": "lead_time", "severity": "info",
                                       "message": f"First reliable warning was {rec['lead_time_min'] / 60:.0f}h {rec['lead_time_min'] % 60:02.0f}m before the failure",
                                       "index": i, "lead_time_min": rec["lead_time_min"]})
                        else:
                            rec["lead_time_min"] = None
                            ev.append({"ts": rec["timestamp"], "asset": a, "kind": "missed", "severity": "warning",
                                       "message": "No warning was active before this failure (missed)", "index": i})
                    recs[a].append(rec)
                    events.extend(ev)
                    alerts.extend(self._alerts(rec))
                self.cursor += 1
            self.feed.extend(events)
            return {"session": self.id, "cursor": self.cursor, "done": self.cursor >= self.stop, "records": recs,
                    "events": events, "alerts": alerts}

    def summary(self) -> dict:
        return {"id": self.id, "assets": self.assets, "mode": self.mode, "start": self.start, "stop": self.stop,
                "cursor": self.cursor, "incident": self.incident, "warmup_ms": round(self.warmup_ms, 1),
                "start_timestamp": str(self.dfs[self.assets[0]].timestamp.iloc[self.start]),
                "stop_timestamp": str(self.dfs[self.assets[0]].timestamp.iloc[min(self.stop, len(self.dfs[self.assets[0]])) - 1])}


class SessionManager:
    def __init__(self, max_sessions: int = 12):
        self.sessions: dict[str, Session] = {}
        self.max = max_sessions

    def add(self, s: Session) -> Session:
        if len(self.sessions) >= self.max:
            oldest = min(self.sessions.values(), key=lambda x: x.created)
            self.sessions.pop(oldest.id, None)
        self.sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    def drop(self, sid: str) -> None:
        self.sessions.pop(sid, None)
