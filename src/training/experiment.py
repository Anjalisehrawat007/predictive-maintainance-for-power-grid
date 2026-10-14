"""Experiment runner: train -> threshold on validation -> evaluate val & test -> log.

Model *selection* only ever looks at ``val_*`` columns; ``test`` metrics are
recorded for the final report but never used to choose anything.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_processed
from src.data.pipeline import Prepared, prepare
from src.evaluation.efficiency import model_size_kb, sklearn_latency, torch_efficiency
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics, select_threshold
from src.models.baselines import build_baseline
from src.models.recurrent import build_model
from src.training.trainer import WindowBatcher, predict_proba, train_model
from src.utils.config import resolve
from src.utils.logging import get_logger
from src.utils.seed import get_device, set_seed

log = get_logger("experiment")

_CACHE: dict[tuple, Prepared] = {}


def get_prepared(cfg: dict, feature_set: str, horizon: int) -> Prepared:
    key = (feature_set, horizon)
    if key not in _CACHE:
        _CACHE[key] = prepare(cfg, feature_set, horizon)
    return _CACHE[key]


def _evaluate(cfg: dict, p: Prepared, pv: np.ndarray, pt: np.ndarray) -> tuple[dict, dict]:
    t = cfg["threshold"]
    yv, yt = p.y[p.idx["val"]], p.y[p.idx["test"]]
    thr = select_threshold(yv, pv, t["objective"], t["min_recall"], t["max_fpr"])
    _, events = load_processed(cfg)
    res = {}
    detail = {}
    sudden = p.sudden_positive_mask(events)
    for split, y, pr in (("val", yv, pv), ("test", yt, pt)):
        m = classification_metrics(y, pr, thr)
        keep = ~sudden[p.idx[split]]
        m["pr_auc_detectable"] = classification_metrics(y[keep], pr[keep], thr)["pr_auc"]
        ev = evaluate_events(p, split, pr, thr, events, cfg)
        res[split] = m | ev["summary"]
        detail[split] = ev
    return res, detail | {"threshold": thr}


def _log_row(cfg: dict, row: dict, preds: dict) -> None:
    exp = resolve(cfg, "experiments_dir")
    (exp / "preds").mkdir(exist_ok=True)
    np.savez_compressed(exp / "preds" / f"{row['run_id']}.npz", **preds)
    path = exp / "results.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def _row(cfg, run_id, version, model, family, feature_set, horizon, seq_len, loss, res, eff, extra) -> dict:
    v, t = res["val"], res["test"]
    row = {
        "run_id": run_id, "version": version, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model, "family": family, "features": feature_set, "sequence_length": seq_len,
        "prediction_horizon": horizon, "loss": loss, "threshold": v["threshold"],
        # validation (used for selection)
        "val_pr_auc": v["pr_auc"], "val_roc_auc": v["roc_auc"], "val_precision": v["precision"],
        "val_recall": v["recall"], "val_f1": v["f1"], "val_event_recall": v["event_recall_detectable"],
        "val_lead_time_median_min": v["lead_time_median_min"], "val_false_alarms_per_day": v["false_alarms_per_day"],
        # test (reported only)
        "precision": t["precision"], "recall": t["recall"], "f1": t["f1"], "roc_auc": t["roc_auc"],
        "pr_auc": t["pr_auc"], "pr_auc_detectable": t["pr_auc_detectable"],
        "val_pr_auc_detectable": v["pr_auc_detectable"], "accuracy": t["accuracy"], "specificity": t["specificity"],
        "false_positive_rate": t["fpr"], "false_negative_rate": t["fnr"],
        "event_recall": t["event_recall"], "event_recall_detectable": t["event_recall_detectable"],
        "lead_time": t["lead_time_mean_min"], "lead_time_median_min": t["lead_time_median_min"],
        "false_alarms_per_day": t["false_alarms_per_day"], "false_alarms_per_week": t["false_alarms_per_week"],
        "episode_precision": t["episode_precision"], "alarm_time_fraction": t.get("alarm_time_fraction"),
        "val_alarm_time_fraction": v.get("alarm_time_fraction"),
    } | eff | extra
    return row


def run_baseline(cfg: dict, name: str, feature_set: str = "raw_engineered", horizon: int | None = None,
                 version: str = "v0-baseline", seed: int | None = None) -> dict:
    seed = cfg["seed"] if seed is None else seed
    horizon = horizon or cfg["labels"]["horizon_steps"]
    set_seed(seed)
    p = get_prepared(cfg, feature_set, horizon)
    tr, va, te = p.idx["train"], p.idx["val"], p.idx["test"]
    model = build_baseline(name, seed, p.feature_names)
    t0 = time.time()
    model.fit(p.X[tr], p.y[tr].astype(int))
    train_time = time.time() - t0
    pv, pt = model.predict_proba(p.X[va])[:, 1], model.predict_proba(p.X[te])[:, 1]
    res, _ = _evaluate(cfg, p, pv, pt)
    eff = {"parameters": np.nan, "model_size_kb": model_size_kb(model),
           "inference_latency_ms": sklearn_latency(model, p.X.shape[1])["latency_ms_median"],
           "train_time_s": train_time}
    run_id = f"{name}-{uuid.uuid4().hex[:6]}"
    row = _row(cfg, run_id, version, name, "baseline", feature_set, horizon, 1, "n/a", res, eff,
               {"seed": seed, "n_features": len(p.feature_names)})
    _log_row(cfg, row, {"val": pv, "test": pt})
    log.info("%s | val PR-AUC %.3f | test PR-AUC %.3f F1 %.3f evRecall %.2f FA/day %.2f",
             name, row["val_pr_auc"], row["pr_auc"], row["f1"], row["event_recall_detectable"],
             row["false_alarms_per_day"])
    return row


def run_dl(cfg: dict, version: str, model_type: str | None = None, feature_set: str | None = None,
           horizon: int | None = None, seq_len: int | None = None, loss: str | None = None,
           hidden_size: int | None = None, num_layers: int | None = None, dropout: float | None = None,
           lr: float | None = None, negative_ratio: float | None = -1, seed: int | None = None,
           max_epochs: int | None = None, save_as: str | None = None, notes: str = "",
           return_model: bool = False, train_overrides: dict | None = None, trial=None):
    mcfg, tcfg = dict(cfg["model"]), dict(cfg["training"])
    model_type = model_type or mcfg["type"]
    feature_set = feature_set or cfg["features"]["set"]
    horizon = int(horizon or cfg["labels"]["horizon_steps"])
    seq_len = int(seq_len or mcfg["seq_len"])
    hidden_size = int(hidden_size or mcfg["hidden_size"])
    num_layers = int(num_layers or mcfg["num_layers"])
    dropout = float(mcfg["dropout"] if dropout is None else dropout)
    tcfg["loss"] = loss or tcfg["loss"]
    tcfg["lr"] = lr or tcfg["lr"]
    if negative_ratio != -1:
        tcfg["negative_ratio"] = negative_ratio
    if max_epochs:
        tcfg["max_epochs"] = max_epochs
    tcfg.update(train_overrides or {})
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    device = get_device(tcfg["device"])
    p = get_prepared(cfg, feature_set, horizon)
    model = build_model(model_type, p.X.shape[1], hidden_size, num_layers, dropout)
    run_id = f"{model_type}-{uuid.uuid4().hex[:6]}"
    log.info("=== %s | %s | feats=%s L=%d H=%d loss=%s hid=%d layers=%d seed=%d ===", run_id, version,
             feature_set, seq_len, horizon, tcfg["loss"], hidden_size, num_layers, seed)
    tr = train_model(model, p.X, p.y, p.idx["train"], p.idx["val"], seq_len, tcfg, device, seed, trial=trial)
    model = tr.model.to("cpu")
    batcher = WindowBatcher(p.X, p.y, seq_len, torch.device("cpu"))
    pv = predict_proba(model, batcher, p.idx["val"])
    pt = predict_proba(model, batcher, p.idx["test"])
    res, detail = _evaluate(cfg, p, pv, pt)
    eff = torch_efficiency(model, seq_len, p.X.shape[1]) | {"train_time_s": tr.train_time_s}
    row = _row(cfg, run_id, version, model_type, "deep", feature_set, horizon, seq_len, tcfg["loss"], res, eff,
               {"seed": seed, "n_features": len(p.feature_names), "hidden_size": hidden_size,
                "num_layers": num_layers, "dropout": dropout, "lr": tcfg["lr"],
                "negative_ratio": tcfg.get("negative_ratio"), "weight_decay": tcfg["weight_decay"],
                "input_noise": tcfg.get("input_noise", 0.0), "best_epoch": tr.best_epoch, "notes": notes})
    _log_row(cfg, row, {"val": pv, "test": pt})
    if save_as:
        mdir = resolve(cfg, "models_dir")
        torch.save(model.state_dict(), mdir / f"{save_as}.pt")
        meta = {"run_id": run_id, "model_type": model_type, "feature_set": feature_set,
                "feature_names": p.feature_names, "seq_len": seq_len, "horizon": horizon,
                "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
                "threshold": detail["threshold"], "history": tr.history, "metrics": res}
        (mdir / f"{save_as}.json").write_text(json.dumps(meta, indent=2, default=float))
    log.info("%s | val PR-AUC %.3f | test PR-AUC %.3f F1 %.3f evRecall %.2f lead(med) %.0f min FA/day %.2f | %d params %.2f ms",
             run_id, row["val_pr_auc"], row["pr_auc"], row["f1"], row["event_recall_detectable"],
             row["lead_time_median_min"], row["false_alarms_per_day"], row["parameters"], row["inference_latency_ms"])
    return (row, model, p) if return_model else row
