"""Shared helpers for the Phase-2 research iteration.

Phase-2 outputs are written to separate folders (experiments/phase2, models/phase2,
reports/phase2) so the original Phase-1 artefacts are never overwritten.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config, resolve

P2_EXP = PROJECT_ROOT / "experiments" / "phase2"
P2_MODELS = PROJECT_ROOT / "models" / "phase2"
P2_REPORTS = PROJECT_ROOT / "reports" / "phase2"
REPORTS = PROJECT_ROOT / "reports"
SEEDS = [42, 7, 123]


def phase2_config(base: dict | None = None) -> dict:
    """Same config as Phase 1, but experiment/model outputs redirected to phase2 folders."""
    cfg = copy.deepcopy(base or load_config())
    cfg["paths"]["experiments_dir"] = "experiments/phase2"
    cfg["paths"]["models_dir"] = "models/phase2"
    for p in (P2_EXP, P2_MODELS, P2_REPORTS):
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def phase1_meta() -> dict:
    return json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text())


def incumbent_state() -> dict:
    return json.loads((PROJECT_ROOT / "experiments" / "best_config.json").read_text())


def asset_view(p, asset: str):
    """Shallow copy of a Prepared restricted (for event evaluation) to one asset."""
    v = copy.copy(p)
    v.offsets = {asset: p.offsets[asset]}
    return v


def mean_std(df: pd.DataFrame, by: str | list, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = m[c]
        out[c + "_std"] = s[c]
    out["n"] = g.size()
    return out


def fmt_pm(m: float, s: float, nd: int = 3) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if s == s else f"{m:.{nd}f}"


def write_md(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


__all__ = ["P2_EXP", "P2_MODELS", "P2_REPORTS", "REPORTS", "SEEDS", "phase2_config", "phase1_meta",
           "incumbent_state", "asset_view", "mean_std", "fmt_pm", "write_md", "resolve"]
