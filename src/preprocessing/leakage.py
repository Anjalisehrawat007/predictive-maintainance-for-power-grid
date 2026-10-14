"""Explicit data-leakage checks.

Each check returns ``(passed, detail)``.  :func:`run_all_checks` runs every
check, writes ``reports/leakage_report.json`` and raises if any check fails.
"""
from __future__ import annotations

import json
import re

import numpy as np

from src.data.dataset import load_processed, make_labels
from src.data.pipeline import Prepared
from src.features.engineering import FeatureEngineer
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("leakage")

FORBIDDEN = re.compile(r"^(sim_|failure|offline|label|y$|target)")


def check_forbidden_columns(feature_names: list[str]) -> tuple[bool, str]:
    bad = [c for c in feature_names if FORBIDDEN.match(c)]
    return not bad, f"forbidden feature columns: {bad}" if bad else "no simulation/label columns among features"


def check_chronology(p: Prepared) -> tuple[bool, str]:
    """Targets are disjoint, strictly ordered train < val < test within each asset."""
    msgs = []
    for a, off in p.offsets.items():
        n = p.bounds[a].n
        sel = {s: p.idx[s][(p.idx[s] >= off) & (p.idx[s] < off + n)] for s in p.idx}
        if not (sel["train"].max() < sel["val"].min() and sel["val"].max() < sel["test"].min()):
            return False, f"{a}: splits not chronological"
        msgs.append(f"{a}: train<{p.timestamps[sel['val'].min()]}<=val<{p.timestamps[sel['test'].min()]}<=test")
    inter = set(p.idx["train"]) & set(p.idx["val"]) | set(p.idx["val"]) & set(p.idx["test"])
    return not inter, "; ".join(msgs)


def check_embargo(p: Prepared, seq_len: int) -> tuple[bool, str]:
    """Train label windows stay in train; val/test input windows stay in their own period."""
    for a, off in p.offsets.items():
        b = p.bounds[a]
        if b.emb_out < p.horizon or b.emb_in < seq_len:
            return False, f"{a}: embargo too small (out={b.emb_out}<H={p.horizon} or in={b.emb_in}<L={seq_len})"
        for s in ("train", "val", "test"):
            lo, hi = b.period(s)
            t = p.idx[s][(p.idx[s] >= off) & (p.idx[s] < off + b.n)] - off
            if (t - seq_len + 1 < lo).any() and s != "train":
                return False, f"{a}/{s}: input window leaves its period"
            if (t + p.horizon >= hi).any() and s != "test":
                return False, f"{a}/{s}: label window reaches into the next period"
    return True, f"embargo ok for L={seq_len}, H={p.horizon}"


def check_scaler_train_only(p: Prepared, cfg: dict) -> tuple[bool, str]:
    """Recompute train-only statistics and compare with the fitted scaler."""
    dfs, _ = load_processed(cfg)
    fe = p.fe
    for a, df in dfs.items():
        b = p.bounds[a]
        F = fe.transform(df, a)[[c for c in p.feature_names]].to_numpy(float)
        off = df["offline"].to_numpy(bool)
        tr = F[: b.train_end][~off[: b.train_end]]
        names = p.feature_names
        full_names = p.fe.feature_sets()["raw_engineered"]
        mean = p.scaler.mean_[a]
        if len(mean) != len(names):          # 'selected' subset: map back
            mean = mean[[full_names.index(c) for c in names]]
        if not np.allclose(np.nanmean(tr, axis=0), mean, rtol=1e-6, atol=1e-8):
            return False, f"{a}: scaler mean differs from train-only mean"
        if np.allclose(np.nanmean(F, axis=0), mean, rtol=1e-6, atol=1e-8):
            return False, f"{a}: scaler mean equals full-series mean (suspicious)"
    return True, "scaler statistics equal training-period statistics"


def check_feature_causality(cfg: dict, n_probe: int = 5, seed: int = 0) -> tuple[bool, str]:
    """Perturb the future and verify past/present features do not change."""
    dfs, _ = load_processed(cfg)
    rng = np.random.default_rng(seed)
    for a, df in dfs.items():
        df = df.iloc[:6000].reset_index(drop=True)
        fe = FeatureEngineer.from_config(cfg, adaptive=True, robust=True)   # superset of all feature sets
        fe.fit(df, a, np.arange(len(df)) < 3000)
        base = fe.transform(df, a)
        for t in rng.integers(fe.lookback + 10, len(df) - 50, n_probe):  # noqa: B007
            pert = df.copy()
            for c in cfg["dataset"]["sensor_cols"]:
                pert.loc[t + 1:, c] = pert.loc[t + 1:, c] * 3.7 + 11.0
            new = fe.transform(pert, a)
            a_, b_ = base.iloc[: t + 1].to_numpy(), new.iloc[: t + 1].to_numpy()
            if not np.allclose(np.nan_to_num(a_, nan=-999), np.nan_to_num(b_, nan=-999)):
                return False, f"{a}: features at t<={t} changed after perturbing t>{t}"
    return True, "features invariant to future perturbations"


def check_labels(p: Prepared) -> tuple[bool, str]:
    """Labels equal an independent recomputation and use only (t, t+H]."""
    for a, off in p.offsets.items():
        n = p.bounds[a].n
        f = p.failure[off: off + n]
        y = make_labels(f, p.horizon)
        ref = np.array([float(f[t + 1: t + p.horizon + 1].any()) if t + p.horizon < n else np.nan
                        for t in range(n)])
        got = p.y[off: off + n]
        mask = ~np.isnan(got)
        if not np.array_equal(got[mask], ref[mask]) or not np.array_equal(np.isnan(y), np.isnan(ref)):
            return False, f"{a}: label mismatch"
    return True, "labels = 1[failure in (t, t+H]], offline rows unscored"


def run_all_checks(cfg: dict, p: Prepared, seq_len: int) -> dict:
    checks = {
        "forbidden_columns": check_forbidden_columns(p.feature_names),
        "chronological_splits": check_chronology(p),
        "embargo": check_embargo(p, seq_len),
        "scaler_train_only": check_scaler_train_only(p, cfg),
        "feature_causality": check_feature_causality(cfg),
        "labels": check_labels(p),
    }
    report = {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()}
    (resolve(cfg, "reports_dir") / "leakage_report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        log.info("[%s] %s — %s", "PASS" if v["passed"] else "FAIL", k, v["detail"])
    failed = [k for k, v in report.items() if not v["passed"]]
    if failed:
        raise AssertionError(f"Leakage checks failed: {failed}")
    return report
