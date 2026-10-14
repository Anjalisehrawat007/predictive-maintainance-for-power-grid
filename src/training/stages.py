"""Iterative experiment stages.

Each stage evaluates a small set of candidates (incumbent included) with
several seeds and keeps the candidate with the best *validation* score.  The
incumbent configuration lives in ``experiments/best_config.json`` so stages
can be re-run independently; every decision is appended to
``experiments/stage_log.json``.
"""
from __future__ import annotations

import copy
import json

import pandas as pd

from src.training.experiment import run_baseline, run_dl
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("stages")

DEFAULT_STATE = {
    "model_type": "gru_attention", "feature_set": "raw_engineered", "seq_len": 48, "horizon": 8,
    "loss": "weighted_bce", "hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 0.002,
    "negative_ratio": None, "train_overrides": {"weight_decay": 1e-4, "input_noise": 0.0},
}
SEEDS_ABLATION = [42, 7]
SEEDS_FULL = [42, 7, 123]


def _state_path(cfg):
    return resolve(cfg, "experiments_dir") / "best_config.json"


def load_state(cfg) -> dict:
    p = _state_path(cfg)
    return json.loads(p.read_text()) if p.exists() else copy.deepcopy(DEFAULT_STATE)


def save_state(cfg, state: dict) -> None:
    _state_path(cfg).write_text(json.dumps(state, indent=2))


def _log_stage(cfg, entry: dict) -> None:
    p = resolve(cfg, "experiments_dir") / "stage_log.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist = [h for h in hist if h["stage"] != entry["stage"]] + [entry]
    p.write_text(json.dumps(hist, indent=2, default=float))


def _run(cfg, version, st, seed, **kw):
    return run_dl(cfg, version, model_type=st["model_type"], feature_set=st["feature_set"],
                  horizon=st["horizon"], seq_len=st["seq_len"], loss=st["loss"],
                  hidden_size=st["hidden_size"], num_layers=st["num_layers"], dropout=st["dropout"],
                  lr=st["lr"], negative_ratio=st["negative_ratio"], seed=seed,
                  train_overrides=st["train_overrides"], **kw)


def operational_utility(row: dict) -> float:
    """Horizon-independent score: detectable-event recall minus 0.2 x false alarms/day."""
    return row["val_event_recall"] - 0.2 * row["val_false_alarms_per_day"]


def compare(cfg, stage: str, candidates: dict[str, dict], seeds, metric: str = "val_pr_auc",
            incumbent: str | None = None, min_gain: float = 0.0) -> dict:
    """Run every candidate x seed; keep the best mean validation score.

    If ``incumbent`` is given, a challenger only replaces it when it wins by more
    than ``min_gain`` (used for changes that add model complexity, so that
    seed-noise-sized "wins" cannot make the model bigger and slower).
    """
    rows = []
    for label, st in candidates.items():
        for s in seeds:
            r = _run(cfg, f"{stage}:{label}", st, s)
            r["candidate"] = label
            r["score"] = operational_utility(r) if metric == "utility" else r[metric]
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("candidate").agg(
        score=("score", "mean"), score_std=("score", "std"), val_pr_auc=("val_pr_auc", "mean"),
        test_pr_auc=("pr_auc", "mean"), val_event_recall=("val_event_recall", "mean"),
        val_fa_day=("val_false_alarms_per_day", "mean"), params=("parameters", "first"),
        latency_ms=("inference_latency_ms", "mean")).sort_values("score", ascending=False)
    best = agg.index[0]
    reason = "best mean validation score"
    if incumbent and best != incumbent and agg.loc[best, "score"] - agg.loc[incumbent, "score"] <= min_gain:
        reason = (f"gain {agg.loc[best, 'score'] - agg.loc[incumbent, 'score']:+.4f} <= min_gain {min_gain} "
                  f"-> keep incumbent")
        best = incumbent
    log.info("Stage %s results (selection metric=%s, mean over %d seeds):\n%s", stage, metric, len(seeds),
             agg.round(4).to_string())
    log.info("Stage %s -> keep '%s' (%s)", stage, best, reason)
    _log_stage(cfg, {"stage": stage, "metric": metric, "selected": best, "reason": reason,
                     "table": agg.reset_index().to_dict(orient="records")})
    return candidates[best]


def _variant(st: dict, **kw) -> dict:
    v = copy.deepcopy(st)
    for k, val in kw.items():
        if k in ("weight_decay", "input_noise"):
            v["train_overrides"][k] = val
        else:
            v[k] = val
    return v


# ------------------------------------------------------------------ stages
def stage_baselines(cfg):
    for fs in ("raw", "engineered", "raw_engineered", "stationary"):
        run_baseline(cfg, "gbdt", fs, version="baselines")
    for name in ("rule_based", "logistic_regression", "isolation_forest", "random_forest"):
        for fs in ("raw_engineered", "stationary"):
            run_baseline(cfg, name, fs, version="baselines")


def stage_features(cfg):
    st = load_state(cfg)
    cands = {fs: _variant(st, feature_set=fs) for fs in ("raw", "engineered", "raw_engineered", "stationary")}
    save_state(cfg, compare(cfg, "features", cands, SEEDS_ABLATION))


def stage_regularization(cfg):
    st = load_state(cfg)
    cands = {
        "incumbent": st,
        "dropout0.4_wd1e-3": _variant(st, dropout=0.4, weight_decay=1e-3),
        "dropout0.4_wd1e-3_noise0.1": _variant(st, dropout=0.4, weight_decay=1e-3, input_noise=0.1),
        "dropout0.4_wd1e-3_noise0.1_lr5e-4": _variant(st, dropout=0.4, weight_decay=1e-3, input_noise=0.1, lr=5e-4),
    }
    save_state(cfg, compare(cfg, "regularization", cands, SEEDS_ABLATION))


def stage_architectures(cfg):
    st = load_state(cfg)
    cands = {m: _variant(st, model_type=m) for m in
             ("lstm", "gru", "bilstm", "lstm_attention", "gru_attention")}
    cands["gru_lite_h32"] = _variant(st, model_type="gru_lite", hidden_size=32)
    save_state(cfg, compare(cfg, "architectures", cands, SEEDS_FULL))


def stage_imbalance(cfg):
    st = load_state(cfg)
    cands = {
        "bce": _variant(st, loss="bce", negative_ratio=None),
        "weighted_bce": _variant(st, loss="weighted_bce", negative_ratio=None),
        "focal": _variant(st, loss="focal", negative_ratio=None),
        "bce_negsample10": _variant(st, loss="bce", negative_ratio=10),
    }
    save_state(cfg, compare(cfg, "imbalance", cands, SEEDS_ABLATION))


def stage_seq_len(cfg):
    st = load_state(cfg)
    cands = {f"L{L}": _variant(st, seq_len=L) for L in (24, 48, 96)}
    save_state(cfg, compare(cfg, "seq_len", cands, SEEDS_ABLATION))


def stage_horizon(cfg):
    """Labels differ per horizon, so PR-AUC is not comparable: use event-level utility."""
    st = load_state(cfg)
    cands = {f"H{H}": _variant(st, horizon=H) for H in cfg["labels"]["candidate_horizons"]}
    save_state(cfg, compare(cfg, "horizon", cands, SEEDS_ABLATION, metric="utility"))


def stage_hpo(cfg):
    from src.training.hpo import run_hpo

    st = load_state(cfg)
    best = run_hpo(cfg, st)
    cands = {"incumbent": st, "optuna_best": best}
    save_state(cfg, compare(cfg, "hpo_confirm", cands, SEEDS_FULL, incumbent="incumbent", min_gain=0.01))


def stage_adaptive(cfg):
    """Post-diagnosis improvement: regime-relative (robust, causal) feature normalisation."""
    st = load_state(cfg)
    run_baseline(cfg, "gbdt", "adaptive", version="adaptive:baseline")
    cands = {"incumbent": st, "adaptive_features": _variant(st, feature_set="adaptive")}
    save_state(cfg, compare(cfg, "adaptive", cands, SEEDS_FULL, incumbent="incumbent", min_gain=0.0))


STAGES = {
    "baselines": stage_baselines, "features": stage_features, "regularization": stage_regularization,
    "architectures": stage_architectures, "imbalance": stage_imbalance, "seq_len": stage_seq_len,
    "horizon": stage_horizon, "hpo": stage_hpo, "adaptive": stage_adaptive,
}
ORDER = ["baselines", "features", "regularization", "architectures", "imbalance", "seq_len", "horizon", "hpo",
         "adaptive"]
