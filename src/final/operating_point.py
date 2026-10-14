"""Operating point under the final protocol: thresholds + calibration from POOLED fresh validation
realisations; confidence-bin reliability.  Test is evaluated only after everything is fixed."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics, select_threshold, threshold_for_precision
from src.final.common import FINAL, REAL_SEEDS, predict, realisation, split_metrics
from src.phase2.analysis import ece
from src.phase2.calibration import CONFIDENCE_BINS, Calibrator, confidence_from_members, load_members
from src.phase2.common import P2_MODELS
from src.streaming.artifacts import load_artifacts
from src.utils.config import load_config

BINS = {"HIGH": (CONFIDENCE_BINS["HIGH"], 1.01), "MEDIUM": (CONFIDENCE_BINS["MEDIUM"], CONFIDENCE_BINS["HIGH"]),
        "LOW": (-0.01, CONFIDENCE_BINS["MEDIUM"])}
TEST_SEEDS = [12000 + 10 * k for k in range(5)]     # independent re-simulations used ONLY for reporting


def conf_bins(y, p, conf, thr) -> dict:
    alarm = p >= thr
    out = {}
    for b, (lo, hi) in BINS.items():
        m = alarm & (conf >= lo) & (conf < hi)
        out[b] = {"alarms": int(m.sum()), "precision": float((y[m] == 1).mean()) if m.any() else None,
                  "share_of_alarms": float(m.sum() / max(alarm.sum(), 1))}
    return out


def run() -> dict:
    cfg = load_config()
    art = load_artifacts(cfg)
    old_w, old_c = art.meta["active_threshold"], art.meta["critical_threshold"]
    from src.data.pipeline import prepare
    p = prepare(cfg, art.meta["feature_set"], art.meta["horizon"])
    members = load_members(cfg, p, art.meta)
    # ---- pooled fresh validation (decision data)
    ys, ps, cs = [], [], []
    vals = []
    for s in REAL_SEEDS:
        q = realisation(cfg, art, s)
        rows = q.idx["val"]
        pr = predict(art.model, q, rows)
        mp = np.stack([predict(m, q, rows) for m in members])
        ys.append(q.y[rows]); ps.append(pr); cs.append(confidence_from_members(mp))
        vals.append((q, pr))
    y, pr, conf = np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)
    t = cfg["threshold"]
    new_w = select_threshold(y, pr, t["objective"], t["min_recall"], t["max_fpr"])
    new_c = max(new_w, threshold_for_precision(y, pr, cfg["alerts"]["critical_precision"]))
    cal_new = Calibrator("isotonic").fit(pr, y)
    old_cal = Calibrator.from_dict(json.loads((P2_MODELS / "calibrator_phase1_selected.json").read_text()))
    # ---- validation summary at both operating points
    def pooled(thr):
        evs = [split_metrics(cfg, q, "val", prq, thr) for q, prq in vals]
        cm = classification_metrics(y, pr, thr)
        return cm | {k: float(np.nanmean([e[k] for e in evs])) for k in ("event_recall_detectable", "false_alarms_per_day", "lead_time_median_min")}
    val_old, val_new = pooled(old_w), pooled(new_w)
    # ---- test (reporting only): original benchmark + independent re-simulations
    from src.final.common import featurizer  # noqa: F401
    from src.phase2.resim import sudden_mask
    p.sudden = sudden_mask(p, __import__("src.data.dataset", fromlist=["load_processed"]).load_processed(cfg)[1])
    p.events = __import__("src.data.dataset", fromlist=["load_processed"]).load_processed(cfg)[1]
    test_sets = [("original", p)] + [(f"re-sim {k}", realisation(cfg, art, s)) for k, s in enumerate(TEST_SEEDS)]
    rows_out, cal_rows, conf_rows = [], [], []
    for name, q in test_sets:
        rows = q.idx["test"]
        prt = predict(art.model, q, rows)
        yt = q.y[rows]
        for op, (w, _c) in {"phase-1 operating point": (old_w, old_c), "final operating point": (new_w, new_c)}.items():
            m = split_metrics(cfg, q, "test", prt, w)
            rows_out.append({"set": name, "operating_point": op, "threshold": w} | {k: m[k] for k in (
                "pr_auc", "pr_auc_detectable", "precision", "recall", "f1", "event_recall", "event_recall_detectable",
                "lead_time_median_min", "false_alarms_per_day")})
        for cname, c in (("raw score", Calibrator("identity")), ("isotonic (original validation)", old_cal),
                         ("isotonic (pooled fresh validation)", cal_new)):
            q2 = c(prt)
            cal_rows.append({"set": name, "calibration": cname, "brier": float(np.mean((q2 - yt) ** 2)), "ece": ece(yt, q2)})
        mp = np.stack([predict(m_, q, rows) for m_ in members])
        conf_rows.append({"set": name} | {f"{b}_{k}": v[k] for b, v in conf_bins(yt, prt, confidence_from_members(mp), new_w).items()
                                          for k in ("alarms", "precision")})
    ops = pd.DataFrame(rows_out)
    cal = pd.DataFrame(cal_rows)
    cb = pd.DataFrame(conf_rows)
    out = {"protocol": "thresholds & calibration from pooled fresh validation (5 realisations); test reported only",
           "old": {"warning": old_w, "critical": old_c}, "new": {"warning": new_w, "critical": new_c},
           "validation": {"phase-1 operating point": val_old, "final operating point": val_new},
           "test": ops.to_dict(orient="records"), "calibration": cal.to_dict(orient="records"),
           "confidence_bins_val": conf_bins(y, pr, conf, new_w), "confidence_bins_test": cb.to_dict(orient="records"),
           "calibrator": cal_new.to_dict()}
    (FINAL / "operating_point.json").write_text(json.dumps(out, indent=1, default=float))
    (P2_MODELS / "calibrator_final.json").write_text(json.dumps(cal_new.to_dict()))
    (FINAL / "confidence_bins.json").write_text(json.dumps({"val": out["confidence_bins_val"], "test": out["confidence_bins_test"]}, indent=1, default=float))
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    o = run()
    print("old", o["old"], "new", o["new"])
    print(pd.DataFrame(o["validation"]).T[["precision", "recall", "f1", "event_recall_detectable", "false_alarms_per_day", "lead_time_median_min"]].round(3))
    t = pd.DataFrame(o["test"])
    print(t.groupby("operating_point")[["precision", "recall", "f1", "event_recall_detectable", "false_alarms_per_day", "lead_time_median_min"]].mean().round(3))
    print(t[t.set == "original"].round(3).to_string())
    print(pd.DataFrame(o["calibration"]).groupby("calibration")[["brier", "ece"]].mean().round(4))
    print(json.dumps(o["confidence_bins_val"], indent=1))
    print(pd.DataFrame(o["confidence_bins_test"]).round(3).to_string())
