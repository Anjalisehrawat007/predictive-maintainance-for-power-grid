"""Full evaluation of the selected model + comparison tables + figures.

python evaluate.py                 # everything
python evaluate.py --skip-cross    # skip the cross-asset generalisation run
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

from src.data.dataset import load_processed
from src.data.pipeline import prepare
from src.evaluation import report as R
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.models.recurrent import build_model
from src.streaming.artifacts import load_artifacts
from src.training.trainer import WindowBatcher, predict_proba, train_model
from src.utils.config import load_config, resolve
from src.utils.logging import get_logger
from src.utils.plotting import apply_style
from src.utils.seed import set_seed

warnings.filterwarnings("ignore")
log = get_logger("evaluate")

TABLE_COLS = ["version", "model", "features", "sequence_length", "prediction_horizon", "loss", "val_pr_auc",
              "pr_auc", "pr_auc_detectable", "recall", "precision", "f1", "roc_auc", "false_positive_rate",
              "event_recall_detectable", "lead_time_median_min", "false_alarms_per_day", "alarm_time_fraction",
              "inference_latency_ms", "parameters", "model_size_kb"]


def cross_asset(cfg: dict, meta: dict) -> dict:
    """Train on ETTm1 only, evaluate on the ETTm2 test period (unseen asset AND unseen time)."""
    set_seed(cfg["seed"])
    p = prepare(cfg, meta["feature_set"], meta["horizon"])
    a_tr, a_te = cfg["dataset"]["assets"]
    tr = p.idx["train"][p.asset[p.idx["train"]] == a_tr]
    va = p.idx["val"][p.asset[p.idx["val"]] == a_tr]
    te = p.idx["test"][p.asset[p.idx["test"]] == a_te]
    model = build_model(meta["model_type"], p.X.shape[1], meta["hidden_size"], meta["num_layers"], meta["dropout"])
    tcfg = dict(cfg["training"]) | {"loss": meta["loss"]}
    bc = resolve(cfg, "experiments_dir") / "best_config.json"
    st = json.loads(bc.read_text()) if bc.exists() else {}
    tcfg.update(st.get("train_overrides", {}))
    tcfg["lr"] = st.get("lr", tcfg["lr"])
    tcfg["negative_ratio"] = st.get("negative_ratio")
    res = train_model(model, p.X, p.y, tr, va, meta["seq_len"], tcfg, torch.device("cpu"), cfg["seed"])
    b = WindowBatcher(p.X, p.y, meta["seq_len"], torch.device("cpu"))
    pt = predict_proba(res.model, b, te)
    return {"train_asset": a_tr, "test_asset": a_te, "test_pr_auc": float(average_precision_score(p.y[te], pt)),
            "best_val_pr_auc_on_train_asset": res.best_val_pr_auc}


def version_progression(results: pd.DataFrame, stage_log: list) -> pd.DataFrame:
    """One row per iteration step: the configuration kept at that step (mean over its seeds)."""
    steps = [("v0", "best baseline (validation)", None), ("v1", "first GRU-attention (raw+eng., no tuning)", "v1-initial")]
    steps += [(f"v{i + 2}", f"after `{e['stage']}` → {e['selected']}", f"{e['stage']}:{e['selected']}")
              for i, e in enumerate(stage_log)]
    rows = []
    for v, label, key in steps:
        if key is None:
            b = results[results.family == "baseline"]
            if not len(b):
                continue
            g = b.loc[[b.val_pr_auc.idxmax()]]
        else:
            g = results[results.version == key]
            if not len(g):
                continue
        rows.append({"version": v, "step": label, "model": g.model.iloc[0], "seeds": len(g),
                     "val_pr_auc": g.val_pr_auc.mean(), "test_pr_auc": g.pr_auc.mean(),
                     "test_recall": g.recall.mean(), "test_precision": g.precision.mean(), "test_f1": g.f1.mean(),
                     "event_recall": g.event_recall_detectable.mean(), "lead_med_min": g.lead_time_median_min.mean(),
                     "fa_per_day": g.false_alarms_per_day.mean(), "latency_ms": g.inference_latency_ms.mean(),
                     "params": g.parameters.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-cross", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_style()
    fig_dir, rep_dir = resolve(cfg, "figures_dir"), resolve(cfg, "reports_dir")
    _, events = load_processed(cfg)

    log.info("Refreshing operational metrics for all logged runs …")
    results = R.refresh_event_metrics(cfg)

    art = load_artifacts(cfg)
    meta = art.meta
    p = prepare(cfg, meta["feature_set"], meta["horizon"])
    b = WindowBatcher(p.X, p.y, art.seq_len, torch.device("cpu"))
    pv, pt = predict_proba(art.model, b, p.idx["val"]), predict_proba(art.model, b, p.idx["test"])
    thr_w, thr_c = meta["active_threshold"], meta["critical_threshold"]
    final = {"model": meta["active_model"], "model_type": meta["model_type"], "thresholds": {
        "warning": thr_w, "critical": thr_c}, "horizon_steps": meta["horizon"], "seq_len": meta["seq_len"],
        "feature_set": meta["feature_set"], "n_features": meta["n_features"]}
    sudden = p.sudden_positive_mask(events)
    for split, pr in (("val", pv), ("test", pt)):
        y = p.y[p.idx[split]]
        m = classification_metrics(y, pr, thr_w)
        keep = ~sudden[p.idx[split]]
        m["pr_auc_detectable"] = classification_metrics(y[keep], pr[keep], thr_w)["pr_auc"]
        ev = evaluate_events(p, split, pr, thr_w, events, cfg)
        final[split] = m | ev["summary"]
        if split == "test":
            per, fa = ev["per_failure"], ev["false_alarms"]
    per.to_csv(rep_dir / "test_per_failure.csv", index=False)
    fa.to_csv(rep_dir / "test_false_alarms.csv", index=False)

    yt = p.y[p.idx["test"]].astype(int)
    R.fig_confusion(confusion_matrix(yt, (pt >= thr_w).astype(int)), fig_dir / "final_confusion_matrix.png",
                    f"Test confusion matrix (thr={thr_w:.3f})")
    R.fig_lead_time(per, fig_dir / "lead_time_distribution.png", fig_dir / "lead_time_timeline.png",
                    meta["horizon"] * 15)
    for a in p.offsets:
        R.fig_prob_timeline(p, pt, events, thr_w, thr_c, a, fig_dir / f"probability_timeline_{a}.png")
    R.fig_threshold_tradeoff(p, pv, events, cfg, thr_w, fig_dir / "threshold_tradeoff.png")
    R.fig_calibration(yt, pt, fig_dir / "calibration.png")

    # curves for all family models + best baseline (from saved predictions)
    curves = {}
    for name, spec in meta["family_models"].items():
        if spec["quantized"]:
            continue
        a2 = load_artifacts(cfg, name)
        b2 = WindowBatcher(p.X, p.y, a2.seq_len, torch.device("cpu"))
        curves[f"{name} ({spec['model_type']})"] = (yt, predict_proba(a2.model, b2, p.idx["test"]))
    base = results[(results.family == "baseline") & (results.prediction_horizon == meta["horizon"])]
    if len(base):
        bb = base.sort_values("val_pr_auc", ascending=False).iloc[0]
        d = np.load(resolve(cfg, "experiments_dir") / "preds" / f"{bb.run_id}.npz")
        curves[f"best baseline: {bb.model} ({bb.features})"] = (yt, d["test"])
    R.fig_curves(curves, fig_dir / "final_pr_roc_curves.png")

    log.info("Efficiency benchmarks …")
    eff = R.efficiency_table(cfg, p, events)
    eff.to_csv(rep_dir / "efficiency_table.csv", index=False)
    final["efficiency"] = eff.to_dict(orient="records")

    log.info("Explainability …")
    final["explainability"] = R.explain(cfg, art, p, events, fig_dir)
    final["explainability"] |= R.shap_baseline(cfg, fig_dir)

    final["generalisation_per_month"] = R.per_period_metrics(p, pt, thr_w).to_dict(orient="records")
    final["failure_cases"] = R.failure_case_analysis(per, fa, events)
    if not args.skip_cross:
        log.info("Cross-asset generalisation run …")
        final["cross_asset"] = R_cross = cross_asset(cfg, meta)
        same = p.idx["test"][p.asset[p.idx["test"]] == R_cross["test_asset"]]
        final["cross_asset"]["in_domain_test_pr_auc_same_asset"] = float(
            average_precision_score(p.y[same], pt[p.asset[p.idx["test"]] == R_cross["test_asset"]]))

    # ------------------------------------------------------------- tables
    results = pd.read_csv(resolve(cfg, "experiments_dir") / "results.csv")
    md = ["# Experiment tables (generated by evaluate.py)\n",
          "Selection used **validation** metrics only. Test columns are reported, never used for selection.\n",
          "`pr_auc_detectable` excludes positives that belong to *sudden* (no-precursor) failures — a diagnostic only.\n"]
    sl = resolve(cfg, "experiments_dir") / "stage_log.json"
    stage_log = json.loads(sl.read_text()) if sl.exists() else []
    for e in stage_log:
        md.append(f"\n## Stage `{e['stage']}` — selection metric `{e['metric']}` → kept **{e['selected']}**\n")
        md.append(R.md_table(pd.DataFrame(e["table"]), 4))
    prog = version_progression(results, stage_log)
    prog.to_csv(rep_dir / "version_progression.csv", index=False)
    md.append("\n## Version progression (kept configuration after each iteration step; mean over seeds)\n")
    md.append(R.md_table(prog, 3))
    md.append("\n## Baselines (same chronological split, same target rows)\n")
    bl = results[results.family == "baseline"].drop_duplicates(["model", "features", "prediction_horizon"], keep="last")
    md.append(R.md_table(bl[[c for c in TABLE_COLS if c in bl]].sort_values("val_pr_auc", ascending=False)))
    md.append("\n## Final candidates (multi-objective selection on validation)\n")
    fc = pd.read_csv(resolve(cfg, "experiments_dir") / "final_candidates.csv")
    md.append(R.md_table(fc[["file", "model", "selection_score", "val_pr_auc", "val_event_recall",
                             "val_false_alarms_per_day", "val_lead_time_median_min", "inference_latency_ms",
                             "model_size_kb", "parameters"]]))
    md.append("\n## Efficiency (measured on this machine)\n")
    md.append(R.md_table(eff, 3))
    md.append("\n## All runs\n")
    md.append(R.md_table(results[[c for c in TABLE_COLS if c in results]]))
    (rep_dir / "results_tables.md").write_text("\n".join(md))
    (rep_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str))
    t = final["test"]
    log.info("TEST | PR-AUC %.3f ROC-AUC %.3f P %.3f R %.3f F1 %.3f | event recall (detectable) %.2f | "
             "lead median %.0f min | FA/day %.2f", t["pr_auc"], t["roc_auc"], t["precision"], t["recall"], t["f1"],
             t["event_recall_detectable"], t["lead_time_median_min"], t["false_alarms_per_day"])


if __name__ == "__main__":
    main()
