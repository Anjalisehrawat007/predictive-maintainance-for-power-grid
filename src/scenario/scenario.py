"""Scenario Lab: inject a visitor-chosen fault into a synthetic normal-operation stream.

The fault signatures are produced by the *same* injection code as the benchmark (src/data/simulation.py),
with magnitudes in the same units (multiples of each channel's local σ on the real training period).
``severity`` scales the benchmark's mid-range magnitude (1.0 = typical benchmark failure).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.simulation import FaultSimulator
from src.scenario.baseline import OUT as BASELINE_JSON
from src.scenario.baseline import SENSORS, generate
from src.utils.config import load_config

FAULTS = ("none", "thermal_runaway", "load_oscillation", "intermittent_arcing", "sudden")
STEPS_PER_HOUR = 4


def build(asset: str, month: int, fault: str, severity: float, precursor_hours: float, seed: int,
          warm_rows: int, lead_hours: float = 12.0, outage_hours: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Return a session-ready frame (sensors + ground-truth columns) and scenario metadata.

    Row layout: ``warm_rows`` (engine warm-up, not shown) → ``lead_hours`` of normal operation →
    precursor (``precursor_hours``) → trip → outage (``outage_hours``). With ``fault == "none"`` the
    stream is normal operation of the same total length (a false-alarm check)."""
    if fault not in FAULTS:
        raise ValueError(f"fault must be one of {FAULTS}")
    cfg = load_config()["simulation"]
    D = int(round(precursor_hours * STEPS_PER_HOUR)) if fault not in ("none", "sudden") else 0
    R = int(round(outage_hours * STEPS_PER_HOUR))
    t0 = warm_rows + int(lead_hours * STEPS_PER_HOUR)
    tf = t0 + D
    n = tf + R + 4
    df = generate(asset, f"2018-{month:02d}-10", n, seed)
    X = df[SENSORS].to_numpy(dtype=float)
    sig = np.asarray(json.loads(BASELINE_JSON.read_text())["assets"][asset]["local_sigma_real"])
    delta = np.zeros_like(X)
    sim = FaultSimulator(cfg, SENSORS, seed)
    sim._u = lambda lo_hi: float(lo_hi[0] + lo_hi[1]) / 2          # deterministic mid-range magnitudes
    params: dict = {}
    if fault in ("thermal_runaway", "load_oscillation", "intermittent_arcing"):
        params = sim._inject_failure(fault, X, delta, sig, t0, tf)
        delta[t0:tf] *= severity
    phase = np.array(["normal"] * n, dtype=object)
    etype = np.array([""] * n, dtype=object)
    failure = np.zeros(n, dtype=int)
    offline = np.zeros(n, dtype=bool)
    if fault != "none":
        sim._inject_outage(X, delta, sig, tf, R)
        phase[t0:tf] = "precursor"
        phase[tf:tf + R] = "outage"
        etype[t0:tf + R] = fault
        failure[tf] = 1
        offline[tf:tf + R] = True
    out = pd.DataFrame({"timestamp": df.timestamp, **{c: X[:, j] + delta[:, j] for j, c in enumerate(SENSORS)},
                        "offline": offline, "failure": failure, "sim_phase": phase, "sim_event_type": etype})
    meta = {"asset": asset, "month": month, "fault": fault, "severity": severity, "seed": seed,
            "precursor_hours": D / STEPS_PER_HOUR, "onset_idx": t0 if fault != "none" else None,
            "failure_idx": tf if fault != "none" else None, "onset_time": str(out.timestamp.iloc[t0]) if fault != "none" else None,
            "failure_time": str(out.timestamp.iloc[tf]) if fault != "none" else None, "start_idx": warm_rows, "stop_idx": n,
            "params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in params.items()}}
    return out, meta
