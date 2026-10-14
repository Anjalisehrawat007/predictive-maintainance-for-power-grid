"""Cross-asset generalisation with a domain-invariant feature subset.

Strategy (causal, label-free, train-only): keep features whose distribution is similar across the two
transformers' TRAINING periods (cross-asset PSI < 0.10, after each asset's own scaling).  Then train on
one asset and test on the other, in both directions, 3 seeds.  Compared with the stationary set
(Phase 2) and with in-domain training.  Analysis only — the deployed model is not changed.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_processed
from src.data.pipeline import prepare
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics, select_threshold
from src.final.common import FINAL, SEEDS, flat_records
from src.models.recurrent import build_model
from src.phase2.analysis import psi
from src.phase2.common import P2_REPORTS, incumbent_state
from src.training.trainer import WindowBatcher, predict_proba, train_model
from src.utils.config import load_config

PSI_MAX = 0.10


def run() -> pd.DataFrame:
    cfg = load_config()
    p = prepare(cfg, "stationary", 8)
    _, events = load_processed(cfg)
    a1, a2 = list(p.offsets)
    tr = p.idx["train"]
    x1, x2 = tr[p.asset[tr] == a1], tr[p.asset[tr] == a2]
    scores = np.array([psi(p.X[x1, j], p.X[x2, j]) for j in range(p.X.shape[1])])
    keep = np.where(scores < PSI_MAX)[0]
    feats = [p.feature_names[j] for j in keep]
    st = incumbent_state()
    tcfg = copy.deepcopy(cfg["training"]) | {"loss": st["loss"], "lr": st["lr"], "negative_ratio": st.get("negative_ratio")}
    tcfg.update(st["train_overrides"])
    rows = []
    for fs_name, cols in (("stationary (65)", np.arange(p.X.shape[1])), (f"invariant (PSI<{PSI_MAX}, {len(keep)})", keep)):
        X = np.ascontiguousarray(p.X[:, cols])
        for src, tgt in ((a1, a2), (a2, a1), ("both", a1), ("both", a2)):
            for s in SEEDS:
                torch.manual_seed(s); np.random.seed(s)
                sel = lambda split, a: p.idx[split] if a == "both" else p.idx[split][p.asset[p.idx[split]] == a]  # noqa: E731
                m = build_model(st["model_type"], X.shape[1], st["hidden_size"], st["num_layers"], st["dropout"])
                res = train_model(m, X, p.y, sel("train", src), sel("val", src), 48, tcfg, torch.device("cpu"), s)
                b = WindowBatcher(X, p.y, 48, torch.device("cpu"))
                vr = sel("val", src)
                thr = select_threshold(p.y[vr], predict_proba(res.model, b, vr), cfg["threshold"]["objective"],
                                       cfg["threshold"]["min_recall"], cfg["threshold"]["max_fpr"])
                trow = p.idx["test"][p.asset[p.idx["test"]] == tgt]
                pr = predict_proba(res.model, b, trow)
                cm = classification_metrics(p.y[trow], pr, thr)
                full = np.full(len(p.idx["test"]), np.nan)
                full[np.searchsorted(p.idx["test"], trow)] = pr
                v = __import__("copy").copy(p); v.offsets = {tgt: p.offsets[tgt]}
                ev = evaluate_events(v, "test", full, thr, events, cfg)["summary"]
                rows.append({"features": fs_name, "train": src, "test": tgt, "seed": s, "pr_auc": cm["pr_auc"], "recall": cm["recall"],
                             "precision": cm["precision"], "f1": cm["f1"], "event_recall_detectable": ev["event_recall_detectable"],
                             "false_alarms_per_day": ev["false_alarms_per_day"], "lead_time_median_min": ev["lead_time_median_min"]})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "cross_asset_runs.csv", index=False)
    agg = df.groupby(["features", "train", "test"]).agg(["mean", "std"]).drop(columns="seed").round(3)
    out = {"psi_threshold": PSI_MAX, "invariant_features": feats, "n_invariant": len(feats),
           "most_asset_specific": pd.Series(scores, index=p.feature_names).sort_values(ascending=False).head(10).round(3).to_dict(),
           "summary": flat_records(agg)}
    (FINAL / "cross_asset.json").write_text(json.dumps(out, indent=1))
    pd.Series(scores, index=p.feature_names).sort_values().to_csv(P2_REPORTS / "feature_psi_between_assets.csv")
    return agg


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(run().to_string())
