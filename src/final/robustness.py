"""Formal robustness evaluation across stress profiles (configs/failure_simulation.yaml).

A. Deployed model (reference training) vs B. the same configuration trained on a MODERATE-profile
benchmark (more varied signatures).  Each: 3 seeds; thresholds per protocol (pooled fresh NORMAL
validation); evaluated on 5 realisations per profile.  Decision uses validation periods only.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch

from src.data.pipeline import prepare
from src.evaluation.metrics import select_threshold
from src.final.common import (
    FINAL,
    REAL_SEEDS,
    SEEDS,
    featurizer,
    flat_records,
    load_phase2_model,
    predict,
    realisation,
    split_metrics,
)
from src.models.recurrent import build_model
from src.phase2 import resim
from src.phase2.common import incumbent_state
from src.training.trainer import train_model
from src.utils.config import load_config
from src.utils.plotting import INK2, SERIES, apply_style, plt, save

PROFILES = ["NORMAL", "MILD", "MODERATE", "SEVERE"]
EVAL_SEEDS = [21000 + 10 * k for k in range(5)]
TRAIN_SEED = 31000
KEYS = ["pr_auc", "pr_auc_detectable", "recall", "precision", "f1", "event_recall_detectable", "lead_time_median_min",
        "false_alarms_per_day"]


def train_variant(cfg, profile: str) -> tuple[list, object]:
    dfs, _ = resim.simulate(cfg, TRAIN_SEED, cfg["stress_profiles"][profile])
    p = prepare(cfg, "stationary", 8, dfs=dfs)
    st = incumbent_state()
    tcfg = copy.deepcopy(cfg["training"]) | {"loss": st["loss"], "lr": st["lr"], "negative_ratio": st.get("negative_ratio")}
    tcfg.update(st["train_overrides"])
    models = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        m = build_model(st["model_type"], p.X.shape[1], st["hidden_size"], st["num_layers"], st["dropout"])
        models.append(train_model(m, p.X, p.y, p.idx["train"], p.idx["val"], 48, tcfg, torch.device("cpu"), s).model.eval())
    return models, featurizer(p)


def protocol_threshold(cfg, models, art) -> list[float]:
    """Per model: max F1 s.t. recall ≥ 0.70 on pooled fresh NORMAL validation realisations (protocol)."""
    qs = [realisation(cfg, art, s) for s in REAL_SEEDS]
    t = cfg["threshold"]
    out = []
    for m in models:
        y = np.concatenate([q.y[q.idx["val"]] for q in qs])
        pr = np.concatenate([predict(m, q, q.idx["val"]) for q in qs])
        out.append(select_threshold(y, pr, t["objective"], t["min_recall"], t["max_fpr"]))
    return out


def run() -> dict:
    cfg = load_config()
    ref_p = prepare(cfg, "stationary", 8)
    ref_art = featurizer(ref_p)
    st = incumbent_state()
    variants = {"reference training (deployed config)": ([load_phase2_model("features", "stationary", st, s, ref_p.X.shape[1]) for s in SEEDS], ref_art)}
    variants["trained on MODERATE-profile signatures"] = train_variant(cfg, "MODERATE")
    thr = {k: protocol_threshold(cfg, ms, art) for k, (ms, art) in variants.items()}
    rows = []
    for prof in PROFILES:
        for k, es in enumerate(EVAL_SEEDS):
            for vname, (ms, art) in variants.items():
                q = realisation(cfg, art, es, cfg["stress_profiles"][prof])
                for si, m in enumerate(ms):
                    for split in ("val", "test"):
                        pr = predict(m, q, q.idx[split])
                        met = split_metrics(cfg, q, split, pr, thr[vname][si])
                        rows.append({"variant": vname, "profile": prof, "realisation": k, "seed": SEEDS[si], "split": split}
                                    | {kk: met.get(kk) for kk in KEYS})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "robustness_runs.csv", index=False)
    agg = df.groupby(["variant", "split", "profile"], sort=False)[KEYS].agg(["mean", "std"])
    # decision on VALIDATION: adopt B only if mean val PR-AUC over MILD/MODERATE/SEVERE improves by > 0.01
    # and NORMAL val PR-AUC does not drop by more than 0.01
    v = df[df.split == "val"].groupby(["variant", "profile"]).pr_auc.mean().unstack()
    a, b = v.index
    stressed = ["MILD", "MODERATE", "SEVERE"]
    gain = float(v.loc[b, stressed].mean() - v.loc[a, stressed].mean())
    normal_change = float(v.loc[b, "NORMAL"] - v.loc[a, "NORMAL"])
    adopt = gain > 0.01 and normal_change >= -0.01
    out = {"profiles": PROFILES, "thresholds": thr, "val_pr_auc": v.round(4).to_dict(), "stressed_gain": gain,
           "normal_change": normal_change, "adopt_robust_training": adopt,
           "rule": "adopt MODERATE-trained only if mean val PR-AUC over MILD/MODERATE/SEVERE improves by > 0.01 and NORMAL drops ≤ 0.01",
           "summary": flat_records(agg)}
    (FINAL / "robustness.json").write_text(json.dumps(out, indent=1, default=float))
    # robustness curve (test periods, reporting)
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    t = df[df.split == "test"]
    for ax, (key, lab) in zip(axes, [("pr_auc", "PR-AUC"), ("event_recall_detectable", "predictable failures warned"),
                                     ("lead_time_median_min", "median lead time (min)"), ("false_alarms_per_day", "false alarms / day")]):
        for i, (vname, g) in enumerate(t.groupby("variant", sort=False)):
            mm = g.groupby("profile")[key].agg(["mean", "std"]).reindex(PROFILES)
            ax.errorbar(range(4), mm["mean"], yerr=mm["std"], marker="o", capsize=3, color=SERIES[i], label=vname)
        ax.set_xticks(range(4), PROFILES); ax.set_title(lab); ax.grid(True, color="#e4e3df")
    axes[0].legend(fontsize=7)
    fig.suptitle("Robustness curve — held-out test period, 5 re-simulations × 3 seeds per stress level", color=INK2)
    save(fig, FINAL / "robustness_curve.png")
    return out | {"agg": agg}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    o = run()
    print(pd.DataFrame(o["val_pr_auc"]).round(3).to_string())
    print("gain", round(o["stressed_gain"], 4), "normal change", round(o["normal_change"], 4), "adopt", o["adopt_robust_training"])
    print(o["agg"].round(3).to_string())
