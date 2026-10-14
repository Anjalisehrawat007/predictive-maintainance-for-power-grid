"""Phase-2 diagnostics: distribution shift, detectability, false-alarm anatomy.

Simulator ground truth (``sim_*`` columns, events.csv) is used here **for analysis only**;
none of it is a model input.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from src.data.dataset import load_processed
from src.data.pipeline import prepare
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics, select_threshold
from src.features.engineering import FeatureEngineer
from src.phase2 import resim
from src.phase2.common import P2_REPORTS, REPORTS, write_md
from src.streaming.artifacts import load_artifacts
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import load_config
from src.utils.plotting import INK2, STATUS, TYPE_COLOR, apply_style, plt, save

FIG = P2_REPORTS / "figures"

# Detectability classes — thresholds fixed a priori (before looking at model recall).
SNR_STRONG, SNR_MODERATE = 3.0, 1.5          # thermal / oscillation: peak injected amplitude / local sigma
SPIKES_STRONG, SPIKES_MODERATE = 4, 2        # arcing: visible spikes (> 3 local sigma of Δ) in last 4 h


def ece(y, p, bins=15) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def psi(a: np.ndarray, b: np.ndarray, bins=10) -> float:
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    pa = np.histogram(np.clip(a, q[0], q[-1]), q)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def local_sigma_series(x: pd.Series) -> pd.Series:
    """Causal local variability: rolling std (7 d) of the deviation from the 24 h rolling mean."""
    dev = x - x.rolling(96, min_periods=24).mean()
    return dev.rolling(672, min_periods=96).std()


class Ctx:
    """Everything the analyses share (loaded once)."""

    def __init__(self):
        self.cfg = load_config()
        self.art = load_artifacts(self.cfg)
        self.meta = self.art.meta
        self.thr = self.meta["active_threshold"]
        self.p = prepare(self.cfg, self.meta["feature_set"], self.meta["horizon"])
        self.dfs, self.events = load_processed(self.cfg)
        b = WindowBatcher(self.p.X, self.p.y, self.art.seq_len, torch.device("cpu"))
        self.probs = {s: predict_proba(self.art.model, b, self.p.idx[s]) for s in ("train", "val", "test")}
        self.ev = {s: evaluate_events(self.p, s, self.probs[s], self.thr, self.events, self.cfg)
                   for s in ("val", "test")}

    def split_of(self, asset: str, idx: int) -> str:
        b = self.p.bounds[asset]
        return "train" if idx < b.train_end else "val" if idx < b.val_end else "test"


# --------------------------------------------------------------------------- detectability
def detectability(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for a, df in ctx.dfs.items():
        real = {c: df[c] - df[f"sim_delta_{c}"] for c in ("OT", "HUFL", "HULL", "MULL")}
        ls = {c: local_sigma_series(real[c]) for c in real}
        ls_d = {c: real[c].diff().rolling(672, min_periods=96).std() for c in ("HULL", "MULL")}
        for r in ctx.events[(ctx.events.asset == a) & (ctx.events.kind == "failure")].itertuples():
            tf, t0 = int(r.failure_idx), int(r.onset_idx)
            ref = max(t0 - 1, 0)                     # variability measured just BEFORE the precursor
            w2, w4 = slice(max(t0, tf - 8), tf), slice(max(t0, tf - 16), tf)
            snr, metric = np.nan, ""
            if r.type == "thermal_runaway":
                snr = df["sim_delta_OT"].iloc[w2].abs().max() / ls["OT"].iloc[ref]; metric = "OT peak / local σ"
            elif r.type == "load_oscillation":
                snr = df["sim_delta_HUFL"].iloc[w2].abs().max() / ls["HUFL"].iloc[ref]; metric = "HUFL peak / local σ"
            elif r.type == "intermittent_arcing":
                dd = df["sim_delta_HULL"].iloc[w4].diff().abs()
                snr = float((dd > 3 * ls_d["HULL"].iloc[ref]).sum()); metric = "visible spikes (4 h)"
            if r.type == "sudden":
                cls = "D"
            elif r.type == "intermittent_arcing":
                cls = "A" if snr >= SPIKES_STRONG else "B" if snr >= SPIKES_MODERATE else "C"
            else:
                cls = "A" if snr >= SNR_STRONG else "B" if snr >= SNR_MODERATE else "C"
            rows.append({"asset": a, "event_id": r.event_id, "type": r.type, "split": ctx.split_of(a, tf),
                         "failure_time": r.failure_time, "precursor_h": (tf - t0) / 4, "signal_metric": metric,
                         "signal_strength": snr, "class": cls})
    det = pd.DataFrame(rows)
    per = pd.concat([ctx.ev[s]["per_failure"].assign(split=s) for s in ("val", "test")])
    det = det.merge(per[["asset", "event_id", "detected", "lead_time_min"]], on=["asset", "event_id"], how="left")
    # val/test rows only count if the failure is inside the scored range (same population as event metrics)
    det = det[(det.split == "train") | det.detected.notna()].copy()
    det.loc[det.detected.notna(), "detected"] = det.loc[det.detected.notna(), "detected"].astype(bool)
    return det


def interpret(t: pd.DataFrame) -> list[str]:
    """Data-driven interpretation (no pre-written conclusion)."""
    t = t[t["class"] != "D"].assign(detected=lambda d: d.detected.astype(bool))
    rec = t.groupby("class").detected.mean()
    out = ["* Test recall by class: " + ", ".join(f"{k} {rec[k]:.2f} (n={int((t['class'] == k).sum())})" for k in rec.index)]
    mono = all(rec.get(a, 1) >= rec.get(b, 0) for a, b in (("A", "B"), ("B", "C")) if a in rec and b in rec)
    out.append("* Recall is monotonic in signal strength (A ≥ B ≥ C)." if mono else
               "* Recall is **not** monotonic in signal strength: weaker classes are not detected less often, so the "
               "misses are **not** explained by precursor SNR alone.")
    miss = t[~t.detected]
    out.append(f"* Missed detectable failures: {len(miss)} — by type/class: "
               + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in miss.groupby(['type', 'class']).size().items()))
    strong = miss[miss["class"] == "A"]
    if len(strong):
        out.append(f"* {len(strong)} missed failures are class **A** (strong signal) — e.g. "
                   + ", ".join(f"{r.type} {r.asset} {str(r.failure_time)[:10]} ({r.signal_strength:.1f})" for r in strong.head(5).itertuples())
                   + ". These are model/threshold errors, not sensor limits.")
    return out


def write_detectability(ctx: Ctx, det: pd.DataFrame) -> dict:
    out = {}
    lines = ["# Failure detectability analysis\n",
             "Each simulated failure is graded by how much precursor signal the *available sensors* carry, "
             "measured from simulator ground truth (analysis only, never a model input). "
             "Local σ is the 7-day std of the deviation from the 24 h mean of the **real** signal "
             "just before the precursor starts, so it captures the season and asset the failure happens in.\n",
             "| Class | Definition (fixed before computing recall) |", "|---|---|",
             f"| A — strongly detectable | thermal/oscillation peak injected amplitude in the last 2 h ≥ {SNR_STRONG} local σ; arcing ≥ {SPIKES_STRONG} visible spikes in last 4 h |",
             f"| B — moderately detectable | {SNR_MODERATE}–{SNR_STRONG} σ; arcing {SPIKES_MODERATE}–{SPIKES_STRONG - 1} spikes |",
             f"| C — weakly detectable | < {SNR_MODERATE} σ; arcing < {SPIKES_MODERATE} spikes |",
             "| D — fundamentally unpredictable | `sudden` failures: no precursor in any sensor |", ""]
    cnt = det.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(columns=list("ABCD"), fill_value=0)
    lines += ["## Class counts per split\n", cnt.to_markdown(), ""]
    for s in ("val", "test"):
        d = det[det.split == s]
        rec = d.assign(detected=d.detected.astype(int)).groupby("class").detected.agg(["sum", "count"])
        o = {"total_failures": int(len(d)), "detectable_failures (A+B+C)": int((d["class"] != "D").sum()),
             "undetectable (D)": int((d["class"] == "D").sum()),
             "recall_all": float(d.detected.mean()),
             "recall_detectable (A+B+C)": float(d[d["class"] != "D"].detected.mean()),
             "recall_A+B": float(d[d["class"].isin(["A", "B"])].detected.mean()),
             "recall_by_class": {k: f"{int(v['sum'])}/{int(v['count'])}" for k, v in rec.iterrows()}}
        out[s] = o
        lines += [f"## {s} split (selected Phase-1 model, threshold {ctx.thr:.3f})\n"]
        lines += [f"* {k}: **{v if not isinstance(v, float) else f'{v:.3f}'}**" for k, v in o.items()]
        lines.append("")
        tbl = d.assign(detected=d.detected.astype(int)).groupby(["type", "class"]).detected.agg(["sum", "count"]).reset_index()
        tbl["recall"] = (tbl["sum"] / tbl["count"]).round(2)
        lines += [tbl.rename(columns={"sum": "detected", "count": "failures"}).to_markdown(index=False), ""]
    t = det[det.split == "test"]
    lines += ["## Signal strength vs detection (test)\n",
              "![](phase2/figures/detectability_scatter.png)\n",
              "## Interpretation\n",
              f"* {int((t['class'] == 'D').sum())} of {len(t)} test failures (sudden) carry no precursor; any recall "
              "over all failures is capped at "
              f"{(t['class'] != 'D').mean():.0%}.",
              *interpret(t),
              "", "Per-failure table: `reports/phase2/detectability_per_failure.csv`."]
    write_md(REPORTS / "detectability_analysis.md", "\n".join(lines))
    det.to_csv(P2_REPORTS / "detectability_per_failure.csv", index=False)
    (P2_REPORTS / "detectability_summary.json").write_text(json.dumps(out, indent=2, default=str))
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, typ in zip(axes, ["thermal_runaway", "load_oscillation", "intermittent_arcing"]):
        d = det[(det.type == typ) & det.split.isin(["val", "test"])]
        for ok, mk, lab in ((True, "o", "warned"), (False, "x", "missed")):
            dd = d[d.detected == ok]
            ax.scatter(dd.signal_strength, dd.precursor_h, marker=mk, s=40, color=TYPE_COLOR[typ] if ok else STATUS["CRITICAL"],
                       label=lab)
        lim = (SPIKES_MODERATE, SPIKES_STRONG) if typ == "intermittent_arcing" else (SNR_MODERATE, SNR_STRONG)
        for x in lim:
            ax.axvline(x, color=INK2, ls=":", lw=0.8)
        ax.set_xlabel(d.signal_metric.iloc[0] if len(d) else ""); ax.set_ylabel("precursor length (h)")
        ax.set_title(f"{typ} (val+test)"); ax.legend()
    save(fig, FIG / "detectability_scatter.png")
    return out


# --------------------------------------------------------------------------- distribution shift
def distribution_shift(ctx: Ctx, n_resim: int = 8) -> dict:
    p, cfg = ctx.p, ctx.cfg
    out: dict = {}
    L = []
    # 1. feature drift (scaled model inputs), per asset
    drift = []
    for a, _off in p.offsets.items():
        sel = {s: p.idx[s][p.asset[p.idx[s]] == a] for s in ("train", "val", "test")}
        for j, f in enumerate(p.feature_names):
            tr = p.X[sel["train"], j]
            for s in ("val", "test"):
                drift.append({"asset": a, "split": s, "feature": f, "psi": psi(tr, p.X[sel[s], j]),
                              "ks": ks_2samp(tr[::5], p.X[sel[s], j][::2]).statistic})
    drift = pd.DataFrame(drift)
    agg = drift.groupby(["asset", "split"]).agg(mean_psi=("psi", "mean"), n_psi_gt_0_25=("psi", lambda x: int((x > 0.25).sum())),
                                                mean_ks=("ks", "mean")).round(3)
    top = drift[drift.split == "test"].sort_values("psi", ascending=False).head(10)[["asset", "feature", "psi", "ks"]].round(3)
    out["feature_drift"] = agg.reset_index().to_dict(orient="records")
    # 2. seasonal variability & effective SNR (real signals)
    var_rows = []
    for a, df in ctx.dfs.items():
        b = p.bounds[a]
        for c in ("OT", "HUFL", "HULL"):
            real = df[c] - df[f"sim_delta_{c}"]
            dev = real - real.rolling(96, min_periods=24).mean()
            s_tr = dev.iloc[: b.train_end].std()
            for s, (lo, hi) in (("train", (0, b.train_end)), ("val", (b.train_end, b.val_end)), ("test", (b.val_end, b.n))):
                sd = dev.iloc[lo:hi].std()
                var_rows.append({"asset": a, "channel": c, "split": s, "local_sigma": sd, "ratio_to_train": sd / s_tr})
    var = pd.DataFrame(var_rows)
    out["local_variability"] = var.round(3).to_dict(orient="records")
    # 3. failure mix & precursor statistics
    det = pd.read_csv(P2_REPORTS / "detectability_per_failure.csv")
    mix = det.groupby(["split", "type"]).size().unstack(fill_value=0)
    cls = det.groupby(["split", "class"]).size().unstack(fill_value=0)
    snr = det[det.type != "sudden"].groupby(["split", "type"]).signal_strength.median().unstack().round(2)
    prec = det[det.type != "sudden"].groupby("split").precursor_h.mean().round(2)
    # 4. labels
    lab = {s: float(np.nanmean(p.y[p.idx[s]])) for s in ("train", "val", "test")}
    # 5. threshold transfer & score distribution
    thr_rows = []
    for s in ("train", "val", "test"):
        y, pr = p.y[p.idx[s]], ctx.probs[s]
        m = classification_metrics(y, pr, ctx.thr)
        thr_rows.append({"split": s, "recall@thr": m["recall"], "precision@thr": m["precision"],
                         "fpr@thr": m["fpr"], "median_score_pos": float(np.median(pr[y == 1])),
                         "median_score_neg": float(np.median(pr[y == 0])), "p99_score_neg": float(np.quantile(pr[y == 0], .99)),
                         "thr_for_recall_0.7 (diagnostic)": select_threshold(y, pr, "f1_min_recall", 0.7),
                         "brier": float(np.mean((pr - y) ** 2)), "ece": ece(y, pr),
                         "mean_score": float(pr.mean()), "positive_rate": float(y.mean())})
    thr = pd.DataFrame(thr_rows).round(4)
    out["threshold_and_calibration"] = thr.to_dict(orient="records")
    # 6. per-asset / per-month
    month = []
    for s in ("val", "test"):
        idx = p.idx[s]
        d = pd.DataFrame({"asset": p.asset[idx], "month": pd.to_datetime(p.timestamps[idx]).to_period("M").astype(str),
                          "y": p.y[idx], "p": ctx.probs[s]})
        for (a, mth), g in d.groupby(["asset", "month"]):
            if g.y.sum() > 0:
                m = classification_metrics(g.y.values, g.p.values, ctx.thr)
                month.append({"split": s, "asset": a, "month": mth, "pos": int(g.y.sum()), "pr_auc": round(m["pr_auc"], 3),
                              "recall": round(m["recall"], 3)})
    month = pd.DataFrame(month)
    # 7. controlled re-simulation: same real data, new event realisations, fixed model
    rs = []
    for k in range(n_resim):
        dfs, evs = resim.simulate(cfg, 5000 + 10 * k)
        r = resim.evaluate_fixed_model(cfg, ctx.art, dfs, evs, ctx.thr)
        for s in ("val", "test"):
            rs.append({"resim": k, "split": s, "pr_auc": r[s]["pr_auc"], "pr_auc_detectable": r[s]["pr_auc_detectable"],
                       "recall": r[s]["recall"], "event_recall_detectable": r[s]["event_recall_detectable"],
                       "fa_day": r[s]["false_alarms_per_day"]})
    rs = pd.DataFrame(rs)
    rsp = rs.pivot(index="resim", columns="split", values="pr_auc_detectable")
    gap = (rsp["val"] - rsp["test"])
    out["resim"] = {"n": n_resim, "val_pr_auc_det_mean": float(rsp["val"].mean()), "test_pr_auc_det_mean": float(rsp["test"].mean()),
                    "gap_mean": float(gap.mean()), "gap_std": float(gap.std()), "test_lower_in": int((gap > 0).sum())}
    rs.to_csv(P2_REPORTS / "resim_benchmark.csv", index=False)
    # 8. false alarms / misses per split
    fam = {s: ctx.ev[s]["summary"] for s in ("val", "test")}
    pr_v = classification_metrics(p.y[p.idx['val']], ctx.probs['val'], ctx.thr)['pr_auc']
    pr_t = classification_metrics(p.y[p.idx['test']], ctx.probs['test'], ctx.thr)['pr_auc']
    # ------------- report
    L += ["# Why is test worse than validation? — quantitative analysis\n",
          f"Model: `{ctx.meta['active_model']}` (Phase-1 selection), threshold {ctx.thr:.3f}. "
          "Validation = 2017-11-22 → 2018-03-09 (winter), test = 2018-03-10 → 2018-06-26 (spring).\n",
          "## 1. Headline gap\n",
          f"| | val | test |\n|---|---|---|\n| PR-AUC | {pr_v:.3f} | {pr_t:.3f} |\n"
          f"| event recall (detectable) | {fam['val']['event_recall_detectable']:.3f} | {fam['test']['event_recall_detectable']:.3f} |\n"
          f"| false alarms / asset-day | {fam['val']['false_alarms_per_day']:.3f} | {fam['test']['false_alarms_per_day']:.3f} |\n",
          "## 2. Controlled experiment: is the gap the *period* or the *particular failures*?\n",
          f"The real sensor data were kept; the failure/benign events were **re-simulated {n_resim} times** with new seeds "
          "(same simulator settings); the Phase-1 model and preprocessor were applied unchanged.\n",
          rs.groupby("split")[["pr_auc", "pr_auc_detectable", "recall", "event_recall_detectable", "fa_day"]].agg(["mean", "std"]).round(3).to_markdown(),
          "",
          f"Validation minus test PR-AUC (detectable positives): **{gap.mean():+.3f} ± {gap.std():.3f}**; test was lower in "
          f"**{int((gap > 0).sum())}/{n_resim}** re-simulations.",
          ("→ The gap persists across independent event realisations, so it is a property of the **test period "
           "(real operating data)**, not of the particular simulated failures." if (gap > 0).mean() >= 0.75 else
           "→ The gap is **not** systematic across re-simulations: a large part of it is sampling variability of the "
           "particular simulated events in the original benchmark."), "",
          "## 3. Seasonal variability of the real signals (local σ = std of deviation from 24 h mean)\n",
          var.pivot_table(index=["asset", "channel"], columns="split", values="ratio_to_train").round(2).to_markdown(),
          "",
          "Values are relative to the training period. Injected signatures are scaled to the *training* σ, so a ratio "
          "> 1 means the same fault is buried in more natural variability (lower effective SNR).\n",
          "## 4. Feature drift of model inputs (scaled; reference = training rows of the same asset)\n",
          agg.to_markdown(), "", "Top-10 drifting features in test:\n", top.to_markdown(index=False),
          "", "PSI > 0.25 is conventionally 'major shift'.\n",
          "## 5. Failure mix, detectability and precursor statistics\n",
          mix.to_markdown(), "", cls.to_markdown(), "",
          "Median signal strength of precursors (see detectability report for units):\n", snr.to_markdown(), "",
          f"Mean precursor length (h): {prec.to_dict()}\n",
          f"Positive (label) rate: {', '.join(f'{k} {v:.4f}' for k, v in lab.items())}\n",
          "## 6. Threshold transfer and calibration\n", thr.to_markdown(index=False), "",
          "## 7. Per asset × month (PR-AUC at sample level)\n", month.to_markdown(index=False), "",
          ]
    write_md(REPORTS / "phase2_distribution_shift.md", "\n".join(L))
    (P2_REPORTS / "distribution_shift.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- false alarms
def false_alarms(ctx: Ctx) -> pd.DataFrame:
    rows = []
    cfg = ctx.cfg
    nu = ctx.events[ctx.events.kind == "nuisance"]
    fails = ctx.events[ctx.events.kind == "failure"]
    real_feats = {}
    for a, df in ctx.dfs.items():
        real = df.copy()
        for c in cfg["dataset"]["sensor_cols"]:
            real[c] = df[c] - df[f"sim_delta_{c}"]
        fe = FeatureEngineer.from_config(cfg)
        fe.fit(real, a, np.arange(len(real)) < ctx.p.bounds[a].train_end)
        real_feats[a] = fe.transform(real, a)
    for s in ("val", "test"):
        fa = ctx.ev[s]["false_alarms"]
        for r in fa.itertuples():
            a = r.asset
            df = ctx.dfs[a]
            i0 = int(np.searchsorted(df.timestamp.values, np.datetime64(r.start)))
            i1 = int(np.searchsorted(df.timestamp.values, np.datetime64(r.end)))
            win = slice(max(0, i0 - 8), i1 + 1)
            n = nu[(nu.asset == a) & (nu.onset_idx <= i1 + 8) & (nu.end_idx >= i0 - 8)]
            fprev = fails[(fails.asset == a) & (fails.end_idx <= i0) & (fails.end_idx >= i0 - 24)]
            fnext = fails[(fails.asset == a) & (fails.failure_idx > i1) & (fails.failure_idx <= i1 + 96 * 2)]
            rf = real_feats[a].iloc[win]
            zmax = float(np.nanmax(np.abs(rf[[c for c in rf.columns if c.endswith("_zbase")]].to_numpy())))
            ot_res = float(np.nanmax(np.abs(rf["OT_resid"].to_numpy() / (real_feats[a]["OT_resid"].iloc[:ctx.p.bounds[a].train_end].std() + 1e-9))))
            if len(n):
                t = n.type.iloc[0]
                cat = {"benign_thermal_bump": "thermal bump (benign, injected)", "benign_load_surge": "load fluctuation (benign, injected)",
                       "benign_oscillation": "load fluctuation (benign, injected)", "sensor_glitch": "sensor noise (injected glitch)"}[t]
                meaning = "meaningful anomaly (harmless)"
            elif len(fprev):
                cat, meaning = "outage-like / post-outage recovery", "simulation artefact (outage blend-back)"
            elif zmax >= 3 or ot_res >= 3:
                cat, meaning = "seasonal / real-data anomaly", "meaningful anomaly in REAL data"
            else:
                cat, meaning = "unknown", "model mistake (no anomaly found)"
            rows.append({"split": s, "asset": a, "start": r.start, "end": r.end, "duration_h": r.duration_steps / 4,
                         "category": cat, "interpretation": meaning, "max_real_|zbase|": round(zmax, 2),
                         "max_real_OT_resid_sigma": round(ot_res, 2), "failure_within_48h_after": bool(len(fnext)),
                         "peak_score": float(ctx.probs[s][(ctx.p.asset[ctx.p.idx[s]] == a) &
                                                         (ctx.p.timestamps[ctx.p.idx[s]] >= np.datetime64(r.start)) &
                                                         (ctx.p.timestamps[ctx.p.idx[s]] <= np.datetime64(r.end))].max())})
    fa = pd.DataFrame(rows)
    fa.to_csv(P2_REPORTS / "false_alarms_categorised.csv", index=False)
    # base rate of the "real-data anomaly" criterion in random NON-alarm windows (same split & duration)
    rng = np.random.default_rng(0)
    base = {}
    for s in ("val", "test"):
        hits, n = 0, 0
        for a, df in ctx.dfs.items():
            lo, hi = ctx.p.bounds[a].ranges()[s]
            zcols = [c for c in real_feats[a].columns if c.endswith("_zbase")]
            Z = np.abs(real_feats[a][zcols].to_numpy())
            R = np.abs(real_feats[a]["OT_resid"].to_numpy() / (real_feats[a]["OT_resid"].iloc[:ctx.p.bounds[a].train_end].std() + 1e-9))
            dur = max(int(fa[fa.split == s].duration_h.median() * 4), 1) + 8
            for st in rng.integers(lo, hi - dur, 400):
                if df.offline.iloc[st:st + dur].any() or df.sim_phase.iloc[st:st + dur].ne("normal").any():
                    continue
                n += 1
                hits += bool(np.nanmax(Z[st:st + dur]) >= 3 or np.nanmax(R[st:st + dur]) >= 3)
        base[s] = hits / max(n, 1)
    fa.attrs["base_rate"] = base
    L = ["# False-alarm analysis (selected Phase-1 model)\n",
         "A false alarm = an alarm episode (≥ 2 consecutive steps above the warning threshold, gaps ≤ 1 h merged) "
         "that does not overlap the 24 h window before any failure.\n",
         "Categorisation order: (1) overlaps an injected *benign* event (±2 h) → (2) starts ≤ 6 h after an outage ended → "
         "(3) the **real** (pre-injection) signal shows an anomaly (|24 h z-score| ≥ 3 on any channel or "
         "|OT residual| ≥ 3 σ_train) → (4) otherwise *unknown*.\n"]
    for s in ("val", "test"):
        d = fa[fa.split == s]
        L += [f"## {s}: {len(d)} false-alarm episodes ({ctx.ev[s]['summary']['false_alarms_per_day']:.3f} per asset-day)\n",
              d.groupby(["category", "interpretation"]).size().rename("episodes").reset_index().to_markdown(index=False), "",
              f"Median duration {d.duration_h.median():.2f} h; {int(d.failure_within_48h_after.sum())} episodes were followed "
              "by a failure within 48 h (outside the 24 h scoring window: early but unscored warnings).\n"]
    t = fa[fa.split == "test"]
    meaningful = t.interpretation.str.startswith("meaningful").mean() if len(t) else float("nan")
    br = fa.attrs.get("base_rate", {})
    obs = (t.category == "seasonal / real-data anomaly").sum() / max((~t.category.str.contains("injected|outage")).sum(), 1)
    L += ["## How meaningful is the 'real-data anomaly' category?\n",
          f"The same criterion applied to random *non-alarm, event-free* windows of equal length is met in "
          f"**{br.get('test', float('nan')):.0%}** (test) / {br.get('val', float('nan')):.0%} (val) of windows. "
          f"Among test false alarms not explained by an injected event, {obs:.0%} meet it. "
          + ("The model's alarms are therefore *enriched* for real anomalies, but the criterion is common; treat this "
             "category as 'unusual real operating condition', not as a confirmed fault." if obs > br.get('test', 1)
             else "The model's alarms are **not** enriched for real anomalies relative to chance — this category should be "
                  "read as 'no injected cause', i.e. largely model mistakes."), ""]
    L += ["## Conclusion\n",
          f"* {meaningful:.0%} of test false alarms coincide with a genuine anomaly (an injected harmless disturbance or "
          "an anomaly in the real data); these are *meaningful anomaly detections* that are not failure precursors.",
          f"* {(t.category == 'outage-like / post-outage recovery').mean():.0%} are caused by the simulator's outage "
          "blend-back (an artefact of the simulation layer, not of the real data).",
          f"* {(t.category == 'unknown').mean():.0%} have no identifiable anomaly → genuine model mistakes.",
          "", "Episode list: `reports/phase2/false_alarms_categorised.csv`."]
    write_md(REPORTS / "false_alarm_analysis.md", "\n".join(L))
    return fa


def run_all() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    det = detectability(ctx)
    write_detectability(ctx, det)
    false_alarms(ctx)
    distribution_shift(ctx)


if __name__ == "__main__":
    run_all()
