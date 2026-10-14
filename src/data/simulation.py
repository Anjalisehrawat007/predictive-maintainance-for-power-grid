"""Fault-injection simulation layer.

The ETT dataset contains *real* transformer load and oil-temperature
measurements but **no failure labels**.  This module injects
physically-motivated incipient-fault signatures on top of the real signal,
followed by a failure (trip) and an outage period.  It also injects *benign*
look-alike transients that do **not** lead to failure, so that a detector has
to separate progressive degradation from ordinary disturbances.

Everything injected is stored separately:

* ``sim_delta_<col>``  — the additive perturbation applied to each channel
  (observed = real + delta, so the real measurement is always recoverable);
* ``sim_phase``        — normal / precursor / nuisance / outage;
* ``sim_event_id``, ``sim_event_type``;
* ``failure``          — 1 exactly at the failure (trip) timestamp;
* ``offline``          — asset out of service (not scored).

None of the ``sim_*`` columns are ever used as model inputs (enforced by
``src.preprocessing.leakage``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96


@dataclass
class Event:
    asset: str
    event_id: int
    kind: str            # "failure" | "nuisance"
    type: str
    onset_idx: int       # first perturbed step
    failure_idx: int     # trip time (-1 for nuisance)
    end_idx: int         # last affected step (exclusive)
    params: dict


def local_sigma(x: np.ndarray, window: int = STEPS_PER_DAY) -> float:
    """Std of the deviation from a trailing rolling mean (robust local variability)."""
    s = pd.Series(x)
    dev = s - s.rolling(window, min_periods=window // 4).mean()
    v = float(np.nanstd(dev.to_numpy()))
    return v if v > 1e-6 else float(np.nanstd(x) + 1e-6)


def _ramp(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, endpoint=False) + 1.0 / max(n, 1)


def _lowpass(x: np.ndarray, tau: float) -> np.ndarray:
    """First-order thermal lag (causal exponential smoothing)."""
    a = 1.0 / tau
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


class FaultSimulator:
    """Injects failure and nuisance events into one asset's real time-series."""

    def __init__(self, sim_cfg: dict, sensor_cols: list[str], seed: int):
        self.cfg = sim_cfg
        self.cols = list(sensor_cols)
        self.ci = {c: i for i, c in enumerate(self.cols)}
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def _u(self, lo_hi) -> float:
        return float(self.rng.uniform(lo_hi[0], lo_hi[1]))

    def _choice(self, probs: dict) -> str:
        keys = list(probs)
        p = np.array([probs[k] for k in keys], dtype=float)
        return str(self.rng.choice(keys, p=p / p.sum()))

    # ------------------------------------------------------------- main entry
    def simulate(self, df: pd.DataFrame, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(df)
        X = df[self.cols].to_numpy(dtype=float)
        n_scale = int(n * self.cfg["scale_estimation_frac"])
        sig = np.array([local_sigma(X[:n_scale, j]) for j in range(len(self.cols))])
        delta = np.zeros_like(X)
        phase = np.array(["normal"] * n, dtype=object)
        ev_id = np.full(n, -1, dtype=int)
        ev_type = np.array([""] * n, dtype=object)
        failure = np.zeros(n, dtype=int)
        offline = np.zeros(n, dtype=bool)
        events: list[Event] = []
        busy = np.zeros(n, dtype=bool)  # steps reserved by failure episodes (+ margin)
        var = self.cfg.get("variability") or {}
        self.asset_factor = self._u(var["asset_amp"]) if "asset_amp" in var else 1.0

        # ---------------- failures ----------------
        t = int(self._u(self.cfg["failure_gap_days"]) * STEPS_PER_DAY)
        eid = 0
        while True:
            D = int(self._u(self.cfg["precursor_hours"]) * STEPS_PER_HOUR)
            R = int(self._u(self.cfg["outage_hours"]) * STEPS_PER_HOUR)
            tf = t
            t0 = tf - D
            if tf + R + 4 >= n:
                break
            ftype = self._choice(self.cfg["failure_types"])
            if self.cfg.get("variability"):
                tmp = np.zeros_like(delta)
                params = self._inject_failure(ftype, X, tmp, sig, t0, tf)
                params |= self._apply_variability(tmp, t0, tf)
                delta += tmp
            else:
                params = self._inject_failure(ftype, X, delta, sig, t0, tf)
            self._inject_outage(X, delta, sig, tf, R)
            if ftype != "sudden":
                phase[t0:tf] = "precursor"
                ev_id[t0:tf] = eid
                ev_type[t0:tf] = ftype
            phase[tf:tf + R] = "outage"
            ev_id[tf:tf + R] = eid
            ev_type[tf:tf + R] = ftype
            failure[tf] = 1
            offline[tf:tf + R] = True
            margin = 12 * STEPS_PER_HOUR
            busy[max(0, t0 - margin):min(n, tf + R + margin)] = True
            events.append(Event(asset, eid, "failure", ftype, t0 if ftype != "sudden" else tf,
                                tf, tf + R + 4, params | {"precursor_steps": D if ftype != "sudden" else 0,
                                                         "outage_steps": R}))
            eid += 1
            t = tf + int(self._u(self.cfg["failure_gap_days"]) * STEPS_PER_DAY)

        # ---------------- nuisance (benign) events ----------------
        n_nuis = int(self.cfg["nuisance_per_day"] * n / STEPS_PER_DAY)
        attempts = 0
        placed = 0
        while placed < n_nuis and attempts < n_nuis * 50:
            attempts += 1
            ntype = self._choice(self.cfg["nuisance_types"])
            dur = int(self._u({"benign_thermal_bump": (2, 8), "benign_load_surge": (1, 4),
                               "benign_oscillation": (1, 4), "sensor_glitch": (0.25, 0.75)}[ntype])
                      * STEPS_PER_HOUR)
            t0 = int(self.rng.integers(STEPS_PER_DAY, n - dur - STEPS_PER_DAY))
            span = slice(t0, t0 + dur + 8)
            overlap_ok = var.get("overlap_prob") and not offline[span].any() and self.rng.random() < var["overlap_prob"]
            if busy[span].any() and not overlap_ok:
                continue
            params = self._inject_nuisance(ntype, delta, sig, t0, dur)
            busy[span] = True
            phase[t0:t0 + dur] = "nuisance"
            ev_id[t0:t0 + dur] = eid
            ev_type[t0:t0 + dur] = ntype
            events.append(Event(asset, eid, "nuisance", ntype, t0, -1, t0 + dur, params))
            eid += 1
            placed += 1

        out = df[["timestamp"]].copy()
        out["asset"] = asset
        for j, c in enumerate(self.cols):
            out[c] = X[:, j] + delta[:, j]
        for j, c in enumerate(self.cols):
            out[f"sim_delta_{c}"] = delta[:, j]
        out["sim_phase"] = phase
        out["sim_event_id"] = ev_id
        out["sim_event_type"] = ev_type
        out["failure"] = failure
        out["offline"] = offline
        ev = pd.DataFrame([asdict(e) for e in events])
        if len(ev):
            for col in ("onset_idx", "failure_idx", "end_idx"):
                ev[col.replace("_idx", "_time")] = [
                    out["timestamp"].iloc[i] if 0 <= i < n else pd.NaT for i in ev[col]]
        return out, ev

    # ------------------------------------------------------------ injectors
    def _inject_failure(self, ftype, X, delta, sig, t0, tf) -> dict:
        D = tf - t0
        s = _ramp(D)
        m = self.cfg["magnitudes"]
        ot, hufl, mufl = self.ci["OT"], self.ci["HUFL"], self.ci["MUFL"]
        hull, mull = self.ci["HULL"], self.ci["MULL"]
        if ftype == "thermal_runaway":
            A, p = self._u(m["thermal_amp"]), self._u(m["thermal_power"])
            delta[t0:tf, ot] += A * sig[ot] * s ** p + self.rng.normal(0, 0.15 * sig[ot], D) * s
            return {"amp_sigma": A, "power": p}
        if ftype == "load_oscillation":
            A, P = self._u(m["osc_amp"]), self._u(m["osc_period_steps"])
            p = self._u((1.0, 2.0))
            phi = self._u((0, 2 * np.pi))
            k = np.arange(D)
            env = A * s ** p
            delta[t0:tf, hufl] += env * sig[hufl] * (np.sin(2 * np.pi * k / P + phi) + 0.3)
            delta[t0:tf, mufl] += 0.8 * env * sig[mufl] * (np.sin(2 * np.pi * k / P + phi + 0.3) + 0.3)
            delta[t0:tf, ot] += _lowpass(0.6 * sig[ot] * env, tau=6.0)
            return {"amp_sigma": A, "period_steps": P, "power": p}
        if ftype == "intermittent_arcing":
            lam, A = self._u(m["arc_rate_end"]), self._u(m["arc_amp"])
            spikes = self.rng.random(D) < lam * s ** 2
            mags = self.rng.uniform(0.5, 1.0, D) * A
            delta[t0:tf, hull] += spikes * mags * sig[hull]
            delta[t0:tf, mull] += spikes * mags * 0.7 * sig[mull] * self.rng.choice([0, 1], D)
            delta[t0:tf, ot] += 0.5 * sig[ot] * s ** 2
            return {"rate_end": lam, "amp_sigma": A, "n_spikes": int(spikes.sum())}
        return {}  # sudden: no precursor

    def _apply_variability(self, tmp: np.ndarray, t0: int, tf: int) -> dict:
        """Optional realism knobs (Phase-2 stress tests; OFF in the Phase-1 benchmark).

        * ``amp_scale``       — multiply the whole signature by U(lo, hi)
        * ``onset_delay_max`` — signature stays absent for a random fraction of the precursor
        * ``dropout_prob``    — signature intermittently absent (1–3 h segments)
        * ``shape_noise``     — multiplicative AR(1) irregularity of the growth envelope
        """
        v = self.cfg["variability"]
        D = tf - t0
        seg = tmp[t0:tf]
        info = {}
        if "amp_scale" in v:
            a = self._u(v["amp_scale"]); seg *= a; info["amp_scale"] = a
        if getattr(self, "asset_factor", 1.0) != 1.0:
            seg *= self.asset_factor; info["asset_factor"] = self.asset_factor
        if v.get("onset_delay_max"):
            d = int(self._u((0.0, v["onset_delay_max"])) * D); seg[:d] = 0.0; info["onset_delay_steps"] = d
        if v.get("dropout_prob"):
            mask = np.ones(D)
            i = 0
            while i < D:
                L = int(self.rng.integers(4, 13))
                if self.rng.random() < v["dropout_prob"]:
                    mask[i:i + L] = 0.0
                i += L
            mask[-2:] = 1.0                                  # the trip itself is always preceded by signal
            seg *= mask[:, None]; info["dropout_fraction"] = float(1 - mask.mean())
        if v.get("shape_noise"):
            e = np.zeros(D)
            for i in range(1, D):
                e[i] = 0.9 * e[i - 1] + self.rng.normal(0, v["shape_noise"])
            seg *= np.clip(1 + e, 0, None)[:, None]
        return info

    def _inject_outage(self, X, delta, sig, tf, R) -> None:
        """Asset trips: loads collapse to ~0, OT cools down; short blend back afterwards."""
        n = len(X)
        R = min(R, n - tf)
        ot = self.ci["OT"]
        tau = np.arange(R)
        for j in range(X.shape[1]):
            if j == ot:
                continue
            delta[tf:tf + R, j] = -X[tf:tf + R, j] + self.rng.normal(0, 0.05 * sig[j], R)
        ot_prev = X[tf - 1, ot] + delta[tf - 1, ot]
        cooled = ot_prev - 2.5 * sig[ot] * (1 - np.exp(-tau / 8.0))
        delta[tf:tf + R, ot] = cooled - X[tf:tf + R, ot]
        # linear 1 h blend back to the real signal
        end = tf + R
        b = min(4, n - end)
        if b > 0:
            w = np.linspace(1, 0, b + 2)[1:-1]
            delta[end:end + b] = delta[end - 1] * w[:, None]

    def _inject_nuisance(self, ntype, delta, sig, t0, dur) -> dict:
        ot, hufl, mufl = self.ci["OT"], self.ci["HUFL"], self.ci["MUFL"]
        k = np.arange(dur)
        if ntype == "benign_thermal_bump":
            A = self._u((1.0, 3.0))
            bump = A * sig[ot] * np.sin(np.pi * (k + 0.5) / dur) ** 2
            delta[t0:t0 + dur, ot] += bump
            return {"amp_sigma": A}
        if ntype == "benign_load_surge":
            A = self._u((1.5, 4.0))
            shape = np.minimum(1.0, np.minimum(k + 1, dur - k) / 2.0)
            delta[t0:t0 + dur, hufl] += A * sig[hufl] * shape
            delta[t0:t0 + dur, mufl] += 0.8 * A * sig[mufl] * shape
            delta[t0:t0 + dur, ot] += _lowpass(0.5 * sig[ot] * shape, tau=6.0)
            return {"amp_sigma": A}
        if ntype == "benign_oscillation":
            A, P = self._u((1.0, 2.5)), self._u((4, 12))
            delta[t0:t0 + dur, hufl] += A * sig[hufl] * np.sin(2 * np.pi * k / P)
            delta[t0:t0 + dur, mufl] += 0.8 * A * sig[mufl] * np.sin(2 * np.pi * k / P)
            return {"amp_sigma": A, "period_steps": P}
        # sensor_glitch
        j = int(self.rng.integers(0, len(self.cols)))
        A = self._u((3.0, 8.0))
        idx = self.rng.choice(dur, size=min(dur, int(self.rng.integers(1, 4))), replace=False)
        delta[t0 + idx, j] += A * sig[j] * self.rng.choice([-1, 1], len(idx))
        return {"channel": self.cols[j], "amp_sigma": A}
