"""Phase-2 experiments with *multi-realisation validation*.

Phase-1 diagnostics showed the original validation events were 'used up' by Phase-1
selection (the selected model scores 0.55 on them but ≈0.38 on fresh realisations).
Every Phase-2 run is therefore evaluated on:

* ``val``      — the original validation period/events (threshold is chosen here);
* ``val_rs``   — K fresh re-simulations of the validation period (same real data, new
                 events, same simulator settings).  **This is the selection metric.**
* ``test``/``test_rs`` — reported only; never used for any decision.

Results → experiments/phase2/results.csv; weights → models/phase2/<group>/.
"""
from __future__ import annotations

import argparse
import copy
import time
import uuid

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_processed, make_labels
from src.evaluation.efficiency import model_size_kb, torch_latency
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics, select_threshold
from src.models.recurrent import build_model, count_parameters
from src.phase2 import resim
from src.phase2.common import P2_EXP, P2_MODELS, SEEDS, incumbent_state, phase2_config
from src.streaming.artifacts import Artifacts
from src.training.experiment import get_prepared
from src.training.trainer import WindowBatcher, predict_proba, train_model
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("phase2")
K_REAL = 5
RS_SEED0 = 7000
MH_HORIZONS = [2, 4, 8, 16]            # 30 min, 1 h, 2 h, 4 h
_RS_DATA: dict[int, tuple] = {}
_RS_PREP: dict[tuple, object] = {}


def _rs_data(cfg, k):
    if k not in _RS_DATA:
        _RS_DATA[k] = resim.simulate(cfg, RS_SEED0 + 10 * k)
    return _RS_DATA[k]


def _featurizer(p) -> Artifacts:
    return Artifacts(None, {}, p.fe, p.scaler, p.extras["scaler_columns"], p.feature_names)


def rs_prepared(cfg, p, k, horizon):
    key = (p.extras["feature_set"], k, horizon)
    if key not in _RS_PREP:
        if len(_RS_PREP) > 12:                     # bound memory: keep only the current feature set
            for kk in [kk for kk in _RS_PREP if kk[0] != key[0]]:
                del _RS_PREP[kk]
        dfs, ev = _rs_data(cfg, k)
        q = resim.prepared_from(cfg, _featurizer(p), dfs, horizon)
        q.events = ev
        _RS_PREP[key] = q
    return _RS_PREP[key]


def multi_labels(p, horizons) -> np.ndarray:
    Y = np.full((len(p.y), len(horizons)), np.nan, dtype=np.float32)
    for a, off in p.offsets.items():
        n = p.bounds[a].n
        f = p.failure[off: off + n]
        for j, h in enumerate(horizons):
            Y[off: off + n, j] = make_labels(f, h)
    Y[p.offline] = np.nan
    return Y


def _rows(p, split, assets):
    r = p.idx[split]
    if assets is None:
        return r
    a = np.array([_asset_of(p, i) for i in r]) if not hasattr(p, "asset") else p.asset[r]
    return r[np.isin(a, assets)]


def _asset_of(p, i):
    for a, off in p.offsets.items():
        if off <= i < off + p.bounds[a].n:
            return a


def _view(p, assets):
    if assets is None:
        return p
    v = copy.copy(p)
    v.offsets = {a: o for a, o in p.offsets.items() if a in assets}
    return v


def _metrics(cfg, p, events, split, rows, probs, thr, assets, sudden=None) -> dict:
    """Sample + event metrics for ``rows`` (subset of p.idx[split]) restricted to ``assets``."""
    y = p.y[rows]
    m = classification_metrics(y, probs, thr)
    if sudden is not None:
        keep = ~sudden[rows]
        m["pr_auc_detectable"] = classification_metrics(y[keep], probs[keep], thr)["pr_auc"]
    full = np.full(len(p.idx[split]), np.nan)
    pos = np.searchsorted(p.idx[split], rows)
    full[pos] = probs
    ev = evaluate_events(_view(p, assets), split, full, thr, events, cfg)["summary"]
    return m | ev


def run_one(cfg, group: str, cand: str, st: dict, seed: int, horizons: list[int] | None = None,
            train_assets=None, eval_assets=None, save: bool = True) -> dict:
    set_seed(seed)
    p = get_prepared(cfg, st["feature_set"], st["horizon"])
    _, events = load_processed(cfg)
    L = int(st["seq_len"])
    Y = p.y if not horizons else multi_labels(p, horizons)
    prim = horizons.index(st["horizon"]) if horizons else 0
    tcfg = copy.deepcopy(cfg["training"]) | {"loss": st["loss"], "lr": st["lr"], "negative_ratio": st.get("negative_ratio"),
                                             "primary_col": prim}
    tcfg.update(st.get("train_overrides", {}))
    model = build_model(st["model_type"], p.X.shape[1], st["hidden_size"], st["num_layers"], st["dropout"],
                        n_outputs=len(horizons) if horizons else 1)
    tr_rows, va_rows = _rows(p, "train", train_assets), _rows(p, "val", train_assets)
    t0 = time.time()
    res = train_model(model, p.X, Y, tr_rows, va_rows, L, tcfg, torch.device("cpu"), seed)
    train_time = time.time() - t0
    model = res.model.eval()

    def pred(q, rows):
        b = WindowBatcher(q.X, q.y if not horizons else np.nan_to_num(q.y), L, torch.device("cpu"))
        out = predict_proba(model, b, rows)
        return out

    # threshold: original validation period of the TRAINING assets
    pv_all = pred(p, va_rows)
    pv = pv_all[:, prim] if horizons else pv_all
    thr = select_threshold(p.y[va_rows], pv, cfg["threshold"]["objective"], cfg["threshold"]["min_recall"],
                           cfg["threshold"]["max_fpr"])
    ev_assets = eval_assets
    sud = p.sudden_positive_mask(events)
    row = {"run_id": f"{group}-{cand}-{uuid.uuid4().hex[:5]}", "group": group, "candidate": cand, "seed": seed,
           "model": st["model_type"], "features": st["feature_set"], "seq_len": L, "horizon": st["horizon"],
           "multi_horizon": bool(horizons), "train_assets": ",".join(train_assets or p.offsets),
           "eval_assets": ",".join(ev_assets or p.offsets), "threshold": thr, "best_epoch": res.best_epoch,
           "train_time_s": train_time}
    preds = {}
    for split in ("val", "test"):
        rows = _rows(p, split, ev_assets if split == "test" else train_assets)
        pr_all = pred(p, rows)
        pr = pr_all[:, prim] if horizons else pr_all
        preds[split] = pr
        m = _metrics(cfg, p, events, split, rows, pr, thr, ev_assets if split == "test" else train_assets, sud)
        for k in ("pr_auc", "pr_auc_detectable", "roc_auc", "precision", "recall", "f1", "event_recall_detectable",
                  "lead_time_median_min", "lead_time_mean_min", "false_alarms_per_day", "alarm_time_fraction"):
            row[f"{split}_{k}"] = m.get(k, np.nan)
        if horizons:
            for j, h in enumerate(horizons):
                yh = multi_labels(p, [h])[rows, 0]
                row[f"{split}_pr_auc_h{h}"] = classification_metrics(yh, pr_all[:, j], 0.5)["pr_auc"]
    # multi-realisation evaluation
    acc = {"val": [], "test": []}
    for k in range(K_REAL):
        q = rs_prepared(cfg, p, k, st["horizon"])
        q.asset = np.concatenate([np.full(q.bounds[a].n, a, dtype=object) for a in q.offsets])
        sq = resim.sudden_mask(q, q.events)
        for split in ("val", "test"):
            rows = _rows(q, split, ev_assets if split == "test" else train_assets)
            pr_all = pred(q, rows)
            pr = pr_all[:, prim] if horizons else pr_all
            m = _metrics(cfg, q, q.events, split, rows, pr, thr, ev_assets if split == "test" else train_assets, sq)
            if horizons:
                for j, h in enumerate(horizons):
                    yh = multi_labels(q, [h])[rows, 0]
                    m[f"pr_auc_h{h}"] = classification_metrics(yh, pr_all[:, j], 0.5)["pr_auc"]
            acc[split].append(m)
    for split, ms in acc.items():
        keys = ["pr_auc", "pr_auc_detectable", "precision", "recall", "f1", "event_recall_detectable",
                "lead_time_median_min", "false_alarms_per_day"] + ([f"pr_auc_h{h}" for h in horizons] if horizons else [])
        for k in keys:
            vals = [m.get(k, np.nan) for m in ms]
            row[f"{split}_rs_{k}"] = float(np.nanmean(vals))
        row[f"{split}_rs_pr_auc_sd_realisations"] = float(np.std([m["pr_auc"] for m in ms]))
    row["parameters"] = count_parameters(model)
    row["model_size_kb"] = model_size_kb(model)
    row["latency_ms"] = torch_latency(model, L, p.X.shape[1], 1, threads=1, n_iter=150)["latency_ms_median"]
    if save:
        d = P2_MODELS / group
        d.mkdir(parents=True, exist_ok=True)
        # file-system-safe name ("stationary:ETTm1->ETTm2" -> "stationary__ETTm1_to_ETTm2"), valid on every OS
        fname = cand.replace("->", "_to_").replace(":", "__")
        torch.save(model.state_dict(), d / f"{fname}_s{seed}.pt")
        (P2_EXP / "preds").mkdir(exist_ok=True)
        np.savez_compressed(P2_EXP / "preds" / f"{row['run_id']}.npz", **preds)
    path = P2_EXP / "results.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    log.info("%s/%s s%d | val %.3f | val_rs %.3f (det %.3f) | test %.3f | test_rs %.3f | evrec_rs %.2f FA_rs %.2f | %d params %.2f ms",
             group, cand, seed, row["val_pr_auc"], row["val_rs_pr_auc"], row["val_rs_pr_auc_detectable"],
             row["test_pr_auc"], row["test_rs_pr_auc"], row["val_rs_event_recall_detectable"],
             row["val_rs_false_alarms_per_day"], row["parameters"], row["latency_ms"])
    return row


def V(st, **kw):
    v = copy.deepcopy(st)
    for k, val in kw.items():
        if k in ("weight_decay", "input_noise"):
            v["train_overrides"][k] = val
        else:
            v[k] = val
    return v


def plan(st: dict) -> list[tuple]:
    """(group, candidate, state, kwargs) — a deliberately small, meaningful set."""
    P = []
    for fs in ("stationary", "raw", "engineered", "stationary_nocal", "robust"):
        P.append(("features", fs, V(st, feature_set=fs), {}))
    arch = {"gru": V(st, model_type="gru"), "gru_compact": V(st, model_type="gru_lite", hidden_size=32),
            "lstm_attention": V(st, model_type="lstm_attention"),
            "tcn": V(st, model_type="tcn", hidden_size=32, num_layers=5, dropout=0.2),
            "transformer": V(st, model_type="transformer", hidden_size=32, num_layers=2, dropout=0.2)}
    for k, v in arch.items():
        P.append(("architecture", k, v, {}))
    for h in (2, 4, 16):
        P.append(("horizon", f"single_H{h}", V(st, horizon=h), {}))
    P.append(("horizon", "multi_hazard", st, {"horizons": MH_HORIZONS}))
    for fs in ("stationary", "robust"):
        P.append(("cross_asset", f"{fs}:ETTm1->ETTm2", V(st, feature_set=fs), {"train_assets": ["ETTm1"], "eval_assets": ["ETTm2"]}))
        P.append(("cross_asset", f"{fs}:ETTm2->ETTm1", V(st, feature_set=fs), {"train_assets": ["ETTm2"], "eval_assets": ["ETTm1"]}))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="features,architecture,horizon,cross_asset")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    cfg = phase2_config()
    st = incumbent_state()
    groups = args.groups.split(",")
    done = pd.read_csv(P2_EXP / "results.csv") if (P2_EXP / "results.csv").exists() else pd.DataFrame()
    for g, cand, s, kw in plan(st):
        if g not in groups:
            continue
        for seed in map(int, args.seeds.split(",")):
            if len(done) and ((done.group == g) & (done.candidate == cand) & (done.seed == seed)).any():
                continue                      # resumable
            run_one(cfg, g, cand, s, seed, **kw)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
