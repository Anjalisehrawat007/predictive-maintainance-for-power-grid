"""Evaluate a *fixed* trained model on re-simulated benchmarks.

The real measurements stay the same; only the injected events change (new seed and/or
harder simulator settings).  The model and its fitted preprocessor (train-only fits from
Phase 1) are NOT refitted, so this measures (a) sampling variability of the benchmark
and (b) robustness to different / harder fault signatures.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.dataset import make_labels
from src.data.download import load_raw
from src.data.simulation import FaultSimulator
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.preprocessing.splits import compute_bounds, target_indices
from src.training.trainer import WindowBatcher, predict_proba

_RAW: dict[str, pd.DataFrame] = {}


def simulate(cfg: dict, seed_offset: int, overrides: dict | None = None):
    sim_cfg = copy.deepcopy(cfg["simulation"])
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            sim_cfg.setdefault(k, {}).update(v)
        else:
            sim_cfg[k] = v
    dfs, evs = {}, []
    for i, a in enumerate(cfg["dataset"]["assets"]):
        if a not in _RAW:
            _RAW[a] = load_raw(cfg, a)
        df, ev = FaultSimulator(sim_cfg, cfg["dataset"]["sensor_cols"], seed=seed_offset + i).simulate(_RAW[a], a)
        dfs[a] = df
        evs.append(ev)
    return dfs, pd.concat(evs, ignore_index=True)


def prepared_from(cfg: dict, art, dfs: dict, horizon: int):
    """Prepared-like object built with the *saved* (Phase-1) preprocessor — no refitting."""
    Xs, ys, ts, fails, offl = [], [], [], [], []
    idx = {"train": [], "val": [], "test": []}
    offsets, bounds, off = {}, {}, 0
    for a, df in dfs.items():
        Xs.append(art.features(df, a).astype(np.float32))
        y = make_labels(df["failure"].to_numpy(), horizon)
        o = df["offline"].to_numpy(bool)
        y[o] = np.nan
        b = compute_bounds(len(df), cfg, art.fe.lookback)
        for s in idx:
            t = target_indices(b, s, o)
            idx[s].append(t[~np.isnan(y[t])] + off)
        ys.append(y); ts.append(df["timestamp"].to_numpy()); fails.append(df["failure"].to_numpy()); offl.append(o)
        offsets[a], bounds[a] = off, b
        off += len(df)
    p = SimpleNamespace(X=np.concatenate(Xs), y=np.concatenate(ys), idx={k: np.concatenate(v) for k, v in idx.items()},
                        timestamps=np.concatenate(ts), failure=np.concatenate(fails), offline=np.concatenate(offl),
                        offsets=offsets, bounds=bounds, horizon=horizon)
    return p


def sudden_mask(p, events) -> np.ndarray:
    m = np.zeros(len(p.y), bool)
    for r in events[(events.kind == "failure") & (events.type == "sudden")].itertuples():
        tf = p.offsets[r.asset] + int(r.failure_idx)
        m[max(0, tf - p.horizon): tf] = True
    return m


def evaluate_fixed_model(cfg, art, dfs, events, thr: float, model=None) -> dict:
    p = prepared_from(cfg, art, dfs, art.meta["horizon"])
    b = WindowBatcher(p.X, p.y, art.seq_len, torch.device("cpu"))
    sud = sudden_mask(p, events)
    out = {}
    for s in ("val", "test"):
        pr = predict_proba(model or art.model, b, p.idx[s])
        y = p.y[p.idx[s]]
        m = classification_metrics(y, pr, thr)
        keep = ~sud[p.idx[s]]
        m["pr_auc_detectable"] = classification_metrics(y[keep], pr[keep], thr)["pr_auc"]
        ev = evaluate_events(p, s, pr, thr, events, cfg)["summary"]
        out[s] = m | ev
    return out
