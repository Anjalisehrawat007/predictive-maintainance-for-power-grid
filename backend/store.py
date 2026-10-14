"""Read-only data the API serves (loaded once, cached).  Every value comes from a file
produced by the pipeline — nothing is generated here."""
from __future__ import annotations

import json
import os
from functools import cached_property

import numpy as np
import pandas as pd

from src.data.dataset import load_processed
from src.preprocessing.splits import compute_bounds
from src.utils.config import PROJECT_ROOT, load_config

R = PROJECT_ROOT / "reports"
P2 = R / "phase2"
LOG = P2 / "engine_log"
ASSET_LABEL = {"ETTm1": "T1", "ETTm2": "T2"}
ACK_FILE = P2 / "alert_acks.json"
# Public deployment: no ETDataset rows on the server (CC BY-ND 4.0). Only committed metadata, reports and
# aggregate parameters are used; endpoints that would serve held-out rows are disabled (backend/app.py).
PUBLIC = os.environ.get("SGPM_PUBLIC") == "1"
SPLITS = PROJECT_ROOT / "data" / "splits" / "split_bounds.json"


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _clean(obj):
    """NaN/inf → None so the payload is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.astype(object).where(df.notna(), None).to_dict(orient="records"))


class Store:
    def __init__(self):
        self.cfg = load_config()

    @cached_property
    def data(self):
        if PUBLIC:
            ev = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "events.csv",
                             parse_dates=["onset_time", "failure_time", "end_time"])
            return {}, ev
        return load_processed(self.cfg)

    @property
    def assets(self) -> list[str]:
        return list(self.cfg["dataset"]["assets"])

    @cached_property
    def split_meta(self) -> dict:
        return _json(SPLITS, {})

    @property
    def dfs(self):
        return self.data[0]

    @property
    def events(self):
        return self.data[1]

    @cached_property
    def bounds(self):
        from src.features.engineering import FeatureEngineer
        lb = FeatureEngineer.from_config(self.cfg).lookback
        if PUBLIC:
            return {a: compute_bounds(int(self.split_meta[a]["n"]), self.cfg, lb) for a in self.assets}
        return {a: compute_bounds(len(df), self.cfg, lb) for a, df in self.dfs.items()}

    @cached_property
    def log(self) -> dict[str, pd.DataFrame]:
        out = {}
        for a in self.dfs:
            f = LOG / f"{a}.parquet"
            if f.exists():
                out[a] = pd.read_parquet(f).set_index("index", drop=False)
        return out

    @cached_property
    def log_events(self) -> pd.DataFrame:
        f = LOG / "events.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    @cached_property
    def log_alerts(self) -> pd.DataFrame:
        f = LOG / "alerts.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    @cached_property
    def timings(self) -> dict:
        return _json(LOG / "timings.json", {})

    @cached_property
    def final_metrics(self):
        return _json(R / "final_metrics.json", {})

    @cached_property
    def detectability(self) -> pd.DataFrame:
        f = P2 / "detectability_per_failure.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    @cached_property
    def false_alarms(self) -> pd.DataFrame:
        f = P2 / "false_alarms_categorised.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    def acks(self) -> dict:
        return _json(ACK_FILE, {})

    def set_ack(self, alert_id: str, state: dict) -> None:
        a = self.acks()
        a[alert_id] = state
        ACK_FILE.write_text(json.dumps(a, indent=2))

    def clear_ack(self, alert_id: str) -> None:
        a = self.acks()
        if a.pop(alert_id, None) is not None:
            ACK_FILE.write_text(json.dumps(a, indent=2))

    def test_failures(self) -> pd.DataFrame:
        ev = self.events
        rows = []
        for a, b in self.bounds.items():
            lo, hi = b.ranges()["test"]
            f = ev[(ev.asset == a) & (ev.kind == "failure") & (ev.failure_idx >= lo) & (ev.failure_idx < hi)]
            rows.append(f)
        f = pd.concat(rows)
        det = self.detectability
        if len(det):        # detectability class & signal strength (operating-point independent)
            f = f.merge(det[["asset", "event_id", "class", "signal_strength", "signal_metric"]],
                        on=["asset", "event_id"], how="left")
        final = R / "final" / "test_per_failure.csv"
        if final.exists():  # offline evaluation at the DEPLOYED operating point (src/final/final_eval.py)
            f = f.merge(pd.read_csv(final)[["asset", "event_id", "detected", "lead_time_min"]],
                        on=["asset", "event_id"], how="left")
        if self.log:
            lt = []
            for r in f.itertuples():
                lg = self.log.get(r.asset)
                v = lg.loc[r.failure_idx, "lead_time_min"] if lg is not None and r.failure_idx in lg.index else None
                lt.append(v)
            f["engine_lead_time_min"] = lt
        elif "lead_time_min" in f:
            # no engine log (public deployment): the offline evaluation is used; the two are equal for all
            # 40 test failures (tests/test_api_robustness.py::test_incident_list_is_consistent_with_final_evaluation)
            f["engine_lead_time_min"] = f["lead_time_min"]
        f["label"] = f.asset.map(ASSET_LABEL)
        return f.sort_values("failure_time")
