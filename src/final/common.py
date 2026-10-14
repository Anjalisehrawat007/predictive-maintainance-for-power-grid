"""Shared helpers for the final evaluation (protocol, operating point, robustness, final metrics)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.data.pipeline import prepare
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.models.recurrent import build_model
from src.phase2 import resim
from src.phase2.common import P2_EXP, P2_MODELS, REPORTS, SEEDS, incumbent_state
from src.phase2.experiments import RS_SEED0, V
from src.streaming.artifacts import Artifacts
from src.training.trainer import WindowBatcher, predict_proba

FINAL = REPORTS / "final"
FINAL.mkdir(parents=True, exist_ok=True)
REAL_SEEDS = [RS_SEED0 + 10 * k for k in range(5)]      # fresh validation realisations (protocol)
L, H = 48, 8


def candidates() -> dict[str, tuple[str, str, dict]]:
    st = incumbent_state()
    return {
        "GRU + attention (incumbent)": ("features", "stationary", st),
        "GRU": ("architecture", "gru", V(st, model_type="gru")),
        "Compact GRU (h=32)": ("architecture", "gru_compact", V(st, model_type="gru_lite", hidden_size=32)),
        "LSTM + attention": ("architecture", "lstm_attention", V(st, model_type="lstm_attention")),
        "TCN": ("architecture", "tcn", V(st, model_type="tcn", hidden_size=32, num_layers=5, dropout=0.2)),
        "Lightweight transformer": ("architecture", "transformer", V(st, model_type="transformer", hidden_size=32, num_layers=2, dropout=0.2)),
    }


def load_phase2_model(group: str, cand: str, st: dict, seed: int, n_features: int):
    m = build_model(st["model_type"], n_features, st["hidden_size"], st["num_layers"], st["dropout"])
    m.load_state_dict(torch.load(P2_MODELS / group / f"{cand}_s{seed}.pt", map_location="cpu"))
    return m.eval()


def phase2_threshold(group: str, cand: str, seed: int) -> float:
    r = pd.read_csv(P2_EXP / "results.csv")
    return float(r[(r.group == group) & (r.candidate == cand) & (r.seed == seed)].threshold.iloc[0])


def featurizer(p) -> Artifacts:
    return Artifacts(None, {"seq_len": L, "horizon": H}, p.fe, p.scaler, p.extras["scaler_columns"], p.feature_names)


def reference_prepared(cfg):
    return prepare(cfg, "stationary", H)


def realisation(cfg, art, seed: int, overrides: dict | None = None):
    """Re-simulated benchmark (same real data, new events) featurised with ``art``'s train-fitted preprocessor."""
    dfs, ev = resim.simulate(cfg, seed, overrides)
    q = resim.prepared_from(cfg, art, dfs, H)
    q.events = ev
    q.asset = np.concatenate([np.full(q.bounds[a].n, a, dtype=object) for a in q.offsets])
    q.sudden = resim.sudden_mask(q, ev)
    return q


def predict(model, q, rows) -> np.ndarray:
    return predict_proba(model, WindowBatcher(q.X, q.y, L, torch.device("cpu")), rows)


def split_metrics(cfg, q, split: str, probs: np.ndarray, thr: float) -> dict:
    rows = q.idx[split]
    y = q.y[rows]
    m = classification_metrics(y, probs, thr)
    keep = ~q.sudden[rows]
    m["pr_auc_detectable"] = classification_metrics(y[keep], probs[keep], thr)["pr_auc"]
    return m | evaluate_events(q, split, probs, thr, q.events, cfg)["summary"]


__all__ = ["flat_records", "FINAL", "REAL_SEEDS", "SEEDS", "L", "H", "candidates", "load_phase2_model", "phase2_threshold", "featurizer",
           "reference_prepared", "realisation", "predict", "split_metrics"]


def flat_records(agg) -> list[dict]:
    """MultiIndex (metric, stat) columns → flat keys like ``pr_auc_mean`` (stable JSON for the UI)."""
    a = agg.reset_index()
    a.columns = ["_".join(str(x) for x in c if x != "").strip("_") if isinstance(c, tuple) else c for c in a.columns]
    return json.loads(a.to_json(orient="records"))
