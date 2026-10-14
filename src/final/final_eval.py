"""Final validated metrics of the DEPLOYED system (final operating point) + final false-alarm analysis.

Everything here is evaluation: the model, thresholds and calibrator were fixed beforehand
(reports/model_selection_protocol.md).  Test = original held-out benchmark period.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_processed
from src.data.pipeline import prepare
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.features.engineering import FeatureEngineer
from src.final.common import FINAL
from src.phase2.analysis import ece
from src.phase2.common import P2_REPORTS, REPORTS, write_md
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import load_config

CATS = ["harmless simulated disturbance", "sensor noise", "seasonal / slow thermal behaviour",
        "unusual real sensor behaviour", "model error", "unknown / simulation artefact"]


def _real_features(cfg, dfs, bounds):
    out = {}
    for a, df in dfs.items():
        real = df.copy()
        for c in cfg["dataset"]["sensor_cols"]:
            real[c] = df[c] - df[f"sim_delta_{c}"]
        fe = FeatureEngineer.from_config(cfg)
        fe.fit(real, a, np.arange(len(real)) < bounds[a].train_end)
        F = fe.transform(real, a)
        rs = F["OT_resid"].iloc[: bounds[a].train_end].std()
        z = F[[c for c in F.columns if c.endswith("_zbase")]].abs().max(axis=1)
        # stuck-sensor check only on channels that are rarely flat in normal (training) operation:
        # low-resolution channels repeat values all the time and would make "flat" meaningless
        flats = {c: (real[c].diff() == 0).astype(int).rolling(8).sum() >= 8 for c in cfg["dataset"]["sensor_cols"]}
        usable = [c for c, f in flats.items() if f.iloc[: bounds[a].train_end].mean() < 0.05]
        flat = sum(flats[c] for c in usable) > 0 if usable else pd.Series(False, index=real.index)
        out[a] = pd.DataFrame({"zmax": z, "resid_sigma": (F["OT_resid"] / rs).abs(), "flat": flat})
    return out


def categorise(window: pd.DataFrame, nuis: pd.DataFrame, after_outage: bool) -> str:
    if len(nuis):
        return "sensor noise" if (nuis.type == "sensor_glitch").all() else "harmless simulated disturbance"
    if window.flat.any():
        return "sensor noise"
    if after_outage:
        return "unknown / simulation artefact"
    if window.zmax.max() >= 3:
        return "unusual real sensor behaviour"
    if window.resid_sigma.max() >= 3:
        return "seasonal / slow thermal behaviour"
    return "model error"


def run() -> dict:
    from backend.intelligence import load_deployment
    from src.phase2.calibration import confidence_bin, confidence_from_members
    cfg = load_config()
    dep = load_deployment(cfg)
    art, cal, warn, crit = dep.art, dep.calibrator, dep.warn_raw, dep.crit_raw
    p = prepare(cfg, art.meta["feature_set"], art.meta["horizon"])
    dfs, events = load_processed(cfg)
    b = WindowBatcher(p.X, p.y, art.seq_len, torch.device("cpu"))
    out = {"operating_point": {"warning_raw": warn, "critical_raw": crit, "warning_risk": dep.warn_cal, "critical_risk": dep.crit_cal,
                               "calibration": cal.method}}
    sudden = p.sudden_positive_mask(events)
    det = pd.read_csv(P2_REPORTS / "detectability_per_failure.csv")
    for split in ("val", "test"):
        rows = p.idx[split]
        pr = predict_proba(art.model, b, rows)
        y = p.y[rows]
        m = classification_metrics(y, pr, warn)
        keep = ~sudden[rows]
        m["pr_auc_detectable"] = classification_metrics(y[keep], pr[keep], warn)["pr_auc"]
        ev = evaluate_events(p, split, pr, warn, events, cfg)
        per = ev["per_failure"].merge(det[["asset", "event_id", "class"]], on=["asset", "event_id"], how="left")
        lt = per[per.detected].lead_time_min.astype(float)
        risk = cal(pr)
        m |= ev["summary"] | {
            "recall_all_failures": float(per.detected.mean()),
            "recall_predictable_failures": float(per[per["class"] != "D"].detected.mean()),
            "recall_strong_signal_failures": float(per[per["class"].isin(["A", "B"])].detected.mean()),
            "recall_by_class": {c: f"{int(g.detected.sum())}/{len(g)}" for c, g in per.groupby("class")},
            "lead_time_p25_min": float(lt.quantile(.25)) if len(lt) else None, "lead_time_p75_min": float(lt.quantile(.75)) if len(lt) else None,
            "brier_raw": float(np.mean((pr - y) ** 2)), "brier_calibrated": float(np.mean((risk - y) ** 2)),
            "ece_raw": ece(y, pr), "ece_calibrated": ece(y, risk)}
        if split == "test":
            mp = np.stack([predict_proba(mm, b, rows) for mm in dep.members])
            conf = confidence_from_members(mp)
            alarm = pr >= warn
            m["confidence_bins"] = {lab: {"alarms": int((alarm & sel).sum()), "precision": float((y[alarm & sel] == 1).mean()) if (alarm & sel).any() else None}
                                    for lab, sel in ((b, confidence_bin(conf) == b) for b in ("HIGH", "MEDIUM", "LOW"))}
            fa_eps, per_test, pr_test, conf_test = ev["false_alarms"], per, pr, conf
        out[split] = m
    # ---------------- false-alarm analysis (test)
    real = _real_features(cfg, dfs, p.bounds)
    nu = events[events.kind == "nuisance"]
    fails = events[events.kind == "failure"]
    rows_fa = []
    ts_all = p.timestamps
    for r in fa_eps.itertuples():
        a = r.asset
        df = dfs[a]
        i0 = int(np.searchsorted(df.timestamp.values, np.datetime64(r.start)))
        i1 = int(np.searchsorted(df.timestamp.values, np.datetime64(r.end)))
        n = nu[(nu.asset == a) & (nu.onset_idx <= i1 + 8) & (nu.end_idx >= i0 - 8)]
        after_out = bool(((fails.asset == a) & (fails.end_idx <= i0) & (fails.end_idx >= i0 - 24)).any())
        w = real[a].iloc[max(0, i0 - 8): i1 + 1]
        cat = categorise(w, n, after_out)
        sel = (p.asset[p.idx["test"]] == a) & (ts_all[p.idx["test"]] >= np.datetime64(r.start)) & (ts_all[p.idx["test"]] <= np.datetime64(r.end))
        k = int(np.argmax(np.where(sel, pr_test, -1)))
        nxt = fails[(fails.asset == a) & (fails.failure_idx > i1)].failure_idx.min()
        rows_fa.append({"asset": a, "start": str(r.start), "end": str(r.end), "index": i0 + 0, "duration_h": r.duration_steps / 4,
                        "category": cat, "injected_event": n.type.iloc[0] if len(n) else None,
                        "peak_raw_score": float(pr_test[k]), "peak_risk": float(cal([pr_test[k]])[0]),
                        "confidence_at_peak": float(conf_test[k]),
                        "max_real_abs_zscore": float(w.zmax.max()), "max_real_OT_residual_sigma": float(w.resid_sigma.max()),
                        "hours_to_next_failure": float((nxt - i1) / 4) if nxt == nxt else None})
    fa = pd.DataFrame(rows_fa)
    # base rates of each criterion in random event-free windows of the same median length (context)
    rng = np.random.default_rng(0)
    L = int(max(fa.duration_h.median() * 4, 1) + 8) if len(fa) else 12
    base = {"unusual real sensor behaviour": 0, "seasonal / slow thermal behaviour": 0, "sensor noise": 0}
    nwin = 0
    for a, df in dfs.items():
        lo, hi = p.bounds[a].ranges()["test"]
        ph = df.sim_phase.to_numpy()
        for s in rng.integers(lo, hi - L, 400):
            if (ph[s: s + L] != "normal").any() or df.offline.iloc[s: s + L].any():
                continue
            w = real[a].iloc[s: s + L]
            nwin += 1
            base["unusual real sensor behaviour"] += bool(w.zmax.max() >= 3)
            base["seasonal / slow thermal behaviour"] += bool(w.zmax.max() < 3 and w.resid_sigma.max() >= 3)
            base["sensor noise"] += bool(w.flat.any())
    base = {k: v / max(nwin, 1) for k, v in base.items()}
    out["false_alarms"] = {"episodes": len(fa), "per_day": out["test"]["false_alarms_per_day"],
                           "by_category": fa.category.value_counts().reindex(CATS, fill_value=0).to_dict() if len(fa) else {},
                           "base_rates_event_free_windows": base, "window_steps": L, "n_windows": nwin}
    (FINAL / "false_alarms.json").write_text(fa.to_json(orient="records", indent=1))
    (FINAL / "final_metrics.json").write_text(json.dumps(out, indent=1, default=float))
    per_test.to_csv(FINAL / "test_per_failure.csv", index=False)
    _write_fa_report(out, fa)
    return out


def _write_fa_report(out: dict, fa: pd.DataFrame) -> None:
    t = out["test"]
    f = out["false_alarms"]
    L = ["# Final false-alarm analysis (deployed system, final operating point)\n",
         f"Test period, both transformers. Warning threshold (raw) {out['operating_point']['warning_raw']:.3f}; "
         f"{f['episodes']} false-alarm episodes = **{t['false_alarms_per_day']:.3f} per transformer-day** "
         f"({t['false_alarms_per_week']:.2f} per week). A false alarm is an alarm episode (2 consecutive steps over threshold, "
         "gaps ≤ 1 h merged) that does not overlap the 24 h before any (simulated) failure.\n",
         "## Categories (assigned in this order)\n",
         "1. **harmless simulated disturbance** — overlaps (±2 h) an injected benign thermal bump / load surge / oscillation",
         "2. **sensor noise** — overlaps an injected sensor glitch, or a *real* channel that is normally never flat is flat-lined for ≥ 2 h (stuck sensor)",
         "3. **unknown / simulation artefact** — starts ≤ 6 h after an outage ended (the simulator's blend-back)",
         "4. **unusual real sensor behaviour** — the *real* (pre-injection) signal deviates ≥ 3 σ from its 24 h baseline on some channel",
         "5. **seasonal / slow thermal behaviour** — no short-term deviation, but the real oil temperature is ≥ 3 σ above what load explains (slow ambient/seasonal heating)",
         "6. **model error** — none of the above: no identifiable anomaly\n",
         "| Category | Episodes | Share |", "|---|---|---|"]
    for k, v in f["by_category"].items():
        L.append(f"| {k} | {v} | {v / max(f['episodes'], 1):.0%} |")
    L += ["", "## How unusual are these conditions anyway? (base rates)\n",
          f"Share of random event-free {f['window_steps']}-step windows in the test period meeting each criterion (n = {f['n_windows']}):\n",
          "| Criterion | Base rate |", "|---|---|"]
    for k, v in f["base_rates_event_free_windows"].items():
        L.append(f"| {k} | {v:.0%} |")
    if len(fa):
        meaningful = fa.category.isin(CATS[:4]).mean()
        L += ["", "## Interpretation\n",
              f"* {meaningful:.0%} of false alarms coincide with an identifiable anomaly (injected harmless disturbance, sensor noise, or "
              "unusual / slow real operating behaviour). These are anomaly detections that did not precede a simulated failure.",
              f"* {(fa.category == 'model error').mean():.0%} have no identifiable anomaly → model errors.",
              f"* Median duration {fa.duration_h.median():.2f} h; median confidence at peak {fa.confidence_at_peak.median():.2f} "
              f"(vs. the alarm-confidence bins in `reports/final/final_metrics.json`).",
              f"* {int((fa.hours_to_next_failure.dropna() <= 48).sum())} episodes preceded a failure by 24–48 h (outside the scoring window).",
              "", "Every episode (with timestamps, peak risk, confidence, real-signal deviations) is listed in "
              "`reports/final/false_alarms.json` and can be inspected in the UI (Failure Intelligence → False-alarm explorer)."]
    write_md(REPORTS / "final_false_alarm_analysis.md", "\n".join(L))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    o = run()
    for s in ("val", "test"):
        t = o[s]
        print(s, {k: (round(t[k], 4) if isinstance(t[k], float) else t[k]) for k in (
            "pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "recall_all_failures", "recall_predictable_failures",
            "recall_strong_signal_failures", "recall_by_class", "lead_time_median_min", "lead_time_mean_min", "lead_time_min_min",
            "lead_time_max_min", "false_alarms_per_day", "brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated")})
    print(o["test"]["confidence_bins"])
    print(json.dumps(o["false_alarms"], indent=1))
