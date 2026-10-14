"""Final evaluation report: metrics, lead time, efficiency, explanations, tables, figures."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.dataset import load_processed
from src.data.pipeline import Prepared, prepare
from src.evaluation.efficiency import model_size_kb, torch_latency
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.explainability import importance as xai
from src.streaming.artifacts import load_artifacts
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import PROJECT_ROOT, resolve
from src.utils.logging import get_logger
from src.utils.plotting import FAILURE_TYPES, INK2, SERIES, STATUS, TYPE_COLOR, plt, save

log = get_logger("report")


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- metrics refresh
def refresh_event_metrics(cfg: dict) -> pd.DataFrame:
    """Recompute operational metrics for every logged run from its saved predictions,
    so all rows share the final metric definitions (e.g. alarm_time_fraction)."""
    path = resolve(cfg, "experiments_dir") / "results.csv"
    df = pd.read_csv(path)
    _, events = load_processed(cfg)
    preps: dict[int, Prepared] = {}
    for i, r in df.iterrows():
        f = resolve(cfg, "experiments_dir") / "preds" / f"{r.run_id}.npz"
        if not f.exists():
            continue
        H = int(r.prediction_horizon)
        if H not in preps:
            preps[H] = prepare(cfg, "raw", H)
        p = preps[H]
        d = np.load(f)
        sudden = p.sudden_positive_mask(events)
        for split, pre in (("val", "val_"), ("test", "")):
            ev = evaluate_events(p, split, d[split], float(r.threshold), events, cfg)["summary"]
            df.loc[i, f"{pre}alarm_time_fraction"] = ev["alarm_time_fraction"]
            keep = ~sudden[p.idx[split]]
            df.loc[i, f"{pre}pr_auc_detectable"] = classification_metrics(
                p.y[p.idx[split]][keep], d[split][keep], float(r.threshold))["pr_auc"]
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- figures
def fig_confusion(cm: np.ndarray, path, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b", fontsize=11)
    ax.set_xticks([0, 1], ["pred normal", "pred failure≤H"])
    ax.set_yticks([0, 1], ["normal", "failure≤H"])
    ax.set_title(title)
    save(fig, path)


def fig_curves(curves: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (name, (y, p)) in enumerate(curves.items()):
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        c = SERIES[k % len(SERIES)]
        axes[0].plot(rc, pr, color=c, label=name)
        axes[1].plot(fpr, tpr, color=c, label=name)
    axes[0].axhline(np.mean(list(curves.values())[0][0]), color=INK2, ls=":", lw=0.8, label="prevalence")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("Precision–recall (test)")
    axes[1].plot([0, 1], [0, 1], color=INK2, ls=":", lw=0.8)
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate"); axes[1].set_title("ROC (test)")
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-4, 1)
    axes[0].legend(fontsize=7)
    save(fig, path)


def fig_lead_time(per: pd.DataFrame, path_dist, path_timeline, horizon_min: int):
    det = per[per.detected]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(det.lead_time_min / 60, bins=np.arange(0, 25, 1), color=SERIES[0], rwidth=0.9)
    axes[0].axvline(det.lead_time_min.median() / 60, color=SERIES[1], ls="--",
                    label=f"median {det.lead_time_min.median() / 60:.1f} h")
    axes[0].axvline(horizon_min / 60, color=INK2, ls=":", label=f"label horizon {horizon_min / 60:.0f} h")
    axes[0].set_xlabel("lead time (hours before failure)"); axes[0].set_ylabel("failures")
    axes[0].set_title(f"Lead-time distribution — {len(det)} detected test failures"); axes[0].legend()
    types = [t for t in FAILURE_TYPES if t in set(per.type)]
    data = [per[(per.type == t) & per.detected].lead_time_min / 60 for t in types]
    bp = axes[1].boxplot(data, vert=False, patch_artist=True, widths=0.5)
    for patch, t in zip(bp["boxes"], types):
        patch.set_facecolor(TYPE_COLOR[t]); patch.set_alpha(0.7)
    axes[1].set_yticks(range(1, len(types) + 1),
                       [f"{t}\n({int(per[(per.type == t)].detected.sum())}/{int((per.type == t).sum())} detected)"
                        for t in types])
    axes[1].set_xlabel("lead time (h)"); axes[1].set_title("Lead time by failure mechanism")
    save(fig, path_dist)

    # timeline: anomaly onset -> warning -> failure (aligned at failure = 0)
    d = per.sort_values(["type", "precursor_min"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.2 * len(d) + 1)))
    for i, r in d.iterrows():
        ax.plot([-r.precursor_min / 60, 0], [i, i], color=TYPE_COLOR[r.type], lw=3, alpha=0.35,
                solid_capstyle="butt")
        ax.scatter([-r.precursor_min / 60], [i], marker="|", s=60, color=TYPE_COLOR[r.type])
        if r.detected:
            ax.scatter([-r.lead_time_min / 60], [i], marker="v", s=36, color=STATUS["WARNING"],
                       edgecolor="#0b0b0b", linewidth=0.5, zorder=3)
        else:
            ax.text(0.2, i, "missed", va="center", fontsize=6, color=STATUS["CRITICAL"])
        ax.scatter([0], [i], marker="X", s=36, color=STATUS["CRITICAL"], zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("hours relative to failure  (time →)")
    ax.set_xlim(-24.5, 1.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=TYPE_COLOR[t], lw=3, alpha=0.5, label=f"precursor: {t}") for t in FAILURE_TYPES]
    h += [Line2D([], [], marker="v", ls="", color=STATUS["WARNING"], markeredgecolor="#0b0b0b", label="model warning"),
          Line2D([], [], marker="X", ls="", color=STATUS["CRITICAL"], label="failure")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=7, ncol=6)
    ax.set_title("Normal operation → anomaly begins (bar start) → model warning (▼) → failure (✕), every test failure")
    save(fig, path_timeline)


def fig_prob_timeline(p: Prepared, probs: np.ndarray, events: pd.DataFrame, thr_w: float, thr_c: float,
                      asset: str, path, days: int = 10):
    off = p.offsets[asset]
    lo, hi = p.bounds[asset].ranges()["test"]
    m = (p.idx["test"] >= off + lo) & (p.idx["test"] < off + hi)
    rows, pr = p.idx["test"][m], probs[m]
    fe = events[(events.asset == asset) & (events.kind == "failure")]
    fe = fe[(fe.failure_idx > lo + 96 * 2) & (fe.failure_idx < hi)]
    t_mid = int(fe.failure_idx.iloc[min(2, len(fe) - 1)]) if len(fe) else lo + 96 * days
    a, b = t_mid - 96 * days // 2, t_mid + 96 * days // 2
    sel = (rows - off >= a) & (rows - off < b)
    ts = p.timestamps[rows[sel]]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(ts, pr[sel], color=SERIES[0], lw=1, label="failure probability")
    ax.axhline(thr_w, color=STATUS["WARNING"], ls="--", lw=1, label=f"WARNING ≥ {thr_w:.2f}")
    ax.axhline(thr_c, color=STATUS["CRITICAL"], ls="--", lw=1, label=f"CRITICAL ≥ {thr_c:.2f}")
    for r in events[(events.asset == asset)].itertuples():
        if r.kind == "failure" and a <= r.failure_idx < b:
            ax.axvline(p.timestamps[off + r.failure_idx], color=STATUS["CRITICAL"], lw=1.2)
            if r.type != "sudden":
                ax.axvspan(p.timestamps[off + r.onset_idx], p.timestamps[off + r.failure_idx],
                           color=TYPE_COLOR[r.type], alpha=0.15, lw=0)
            ax.text(p.timestamps[off + r.failure_idx], 1.02, r.type, fontsize=7, ha="center", color=INK2)
        elif r.kind == "nuisance" and a <= r.onset_idx < b:
            ax.axvspan(p.timestamps[off + r.onset_idx], p.timestamps[off + r.end_idx], color="#b0afa9",
                       alpha=0.25, lw=0)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{asset} test period — shaded: precursor (colour) / benign transient (grey); red line = failure")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    save(fig, path)


def fig_threshold_tradeoff(p: Prepared, pv: np.ndarray, events: pd.DataFrame, cfg: dict, thr: float, path):
    yv = p.y[p.idx["val"]]
    grid = np.unique(np.quantile(pv, np.linspace(0.80, 0.9995, 40)))
    rows = []
    for t in grid:
        m = classification_metrics(yv, pv, t)
        ev = evaluate_events(p, "val", pv, t, events, cfg)["summary"]
        rows.append({"thr": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "event_recall": ev["event_recall_detectable"], "fa_day": ev["false_alarms_per_day"]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, c in enumerate(["precision", "recall", "f1", "event_recall"]):
        axes[0].plot(d.thr, d[c], color=SERIES[k], label=c.replace("_", " "))
    axes[1].plot(d.thr, d.fa_day, color=SERIES[7], label="false alarms / day")
    for ax in axes:
        ax.axvline(thr, color=INK2, ls="--", lw=1)
        ax.set_xlabel("decision threshold")
        ax.legend()
    axes[0].set_title("Validation: sample & event metrics vs threshold (dashed = selected)")
    axes[1].set_title("Validation: false-alarm episodes per day vs threshold")
    save(fig, path)
    return d


def fig_calibration(y, p, path):
    bins = np.quantile(p, np.linspace(0, 1, 16))
    bins = np.unique(bins)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
    xs = [p[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 1], [0, 1], color=INK2, ls=":")
    ax.plot(xs, ys, marker="o", color=SERIES[0], ms=5)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed failure rate")
    ax.set_title("Reliability (test, quantile bins)")
    save(fig, path)


# --------------------------------------------------------------------------- efficiency
def memory_probe(model_file: str) -> dict:
    """Peak RSS of a fresh process: after loading, and after a batch-512 inference."""
    code = f"""
import resource, json, torch, sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from src.utils.config import load_config
from src.streaming.artifacts import load_artifacts
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
art = load_artifacts(load_config(), {model_file!r})
r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.no_grad():
    art.model(torch.randn(512, art.seq_len, len(art.feature_names)))
r2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({{"rss_after_load_mb": r1/1048576, "inference_rss_increase_mb": (r2-r1)/1048576}}))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def onnx_latency(model, L: int, F: int, path) -> float | None:
    try:
        import onnxruntime as ort

        x = torch.randn(1, L, F)
        torch.onnx.export(model, x, str(path), input_names=["x"], output_names=["logit"],
                          dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        xn = x.numpy()
        ref = model(x).detach().numpy()
        got = sess.run(None, {"x": xn})[0]
        if not np.allclose(ref, got, atol=1e-4):
            log.warning("ONNX output mismatch")
            return None
        for _ in range(20):
            sess.run(None, {"x": xn})
        ts = []
        for _ in range(200):
            t = time.perf_counter(); sess.run(None, {"x": xn}); ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))
    except Exception as e:
        log.warning("ONNX export/runtime skipped: %s", e)
        return None


def efficiency_table(cfg: dict, p: Prepared, events: pd.DataFrame) -> pd.DataFrame:
    meta = json.loads((resolve(cfg, "models_dir") / "model_metadata.json").read_text())
    rows = []
    has_mps = torch.backends.mps.is_available()
    for name, spec in meta["family_models"].items():
        art = load_artifacts(cfg, name)
        m, L, F = art.model, art.seq_len, len(art.feature_names)
        b = WindowBatcher(p.X, p.y, L, torch.device("cpu"))
        pv, pt = predict_proba(m, b, p.idx["val"]), predict_proba(m, b, p.idx["test"])
        thr = spec["threshold"]
        mt = classification_metrics(p.y[p.idx["test"]], pt, thr)
        ev = evaluate_events(p, "test", pt, thr, events, cfg)["summary"]
        cpu1 = torch_latency(m, L, F, batch=1, threads=1)
        cpu512 = torch_latency(m, L, F, batch=512, threads=4, n_iter=30, warmup=5)
        mps = (torch_latency(m, L, F, batch=1, device="mps", n_iter=100)["latency_ms_median"]
               if has_mps and not spec["quantized"] else np.nan)
        onnx = (onnx_latency(m, L, F, resolve(cfg, "models_dir") / f"{name}.onnx")
                if not spec["quantized"] else None)
        n_par = sum(x.numel() for x in m.parameters())   # 0 for packed int8 modules
        mem = memory_probe(name)
        rows.append({"model": name, "type": spec["model_type"] + (" (int8)" if spec["quantized"] else ""),
                     "params": n_par if n_par else np.nan, "size_kb": model_size_kb(m),
                     "val_pr_auc": float(classification_metrics(p.y[p.idx["val"]], pv, thr)["pr_auc"]),
                     "test_pr_auc": mt["pr_auc"], "test_f1": mt["f1"], "test_recall": mt["recall"],
                     "event_recall": ev["event_recall_detectable"], "lead_median_min": ev["lead_time_median_min"],
                     "fa_per_day": ev["false_alarms_per_day"],
                     "cpu_latency_ms_b1": cpu1["latency_ms_median"], "cpu_p95_ms_b1": cpu1["latency_ms_p95"],
                     "cpu_throughput_per_s_b512": cpu512["throughput_per_s"], "mps_latency_ms_b1": mps,
                     "onnx_cpu_latency_ms_b1": onnx if onnx is not None else np.nan,
                     "inference_rss_increase_mb": mem.get("inference_rss_increase_mb", np.nan)})
    df = pd.DataFrame(rows)
    fp = df[~df.type.str.contains("int8")]
    params = fp.set_index("model").params
    df["params"] = [params.get(m.replace("_int8", ""), np.nan) for m in df.model]
    df["pr_auc_per_10k_params"] = df.val_pr_auc / (df.params / 1e4)
    df["pr_auc_per_ms"] = df.val_pr_auc / df.cpu_latency_ms_b1
    return df


# --------------------------------------------------------------------------- explanations
def explain(cfg: dict, art, p: Prepared, events: pd.DataFrame, fig_dir) -> dict:
    L, names = art.seq_len, art.feature_names
    rows = xai.eval_sample(p.y, p.idx["test"], n_neg=4000, seed=0)
    W = xai.gather_windows(p.X, rows, L)
    y = p.y[rows]
    feat, grp = xai.permutation_importance(art.model, W, y, names, n_repeats=3)
    out = {"permutation_groups": grp.to_dict(orient="records"),
           "permutation_top15": feat.head(15).to_dict(orient="records")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    g = grp.iloc[::-1]
    axes[0].barh(g.group, g.pr_auc_drop, xerr=g["std"], color=SERIES[0], height=0.6)
    axes[0].set_title("Permutation importance by sensor group\n(drop in test PR-AUC when shuffled)")
    f = feat.head(15).iloc[::-1]
    axes[1].barh(f.feature, f.pr_auc_drop, xerr=f["std"], color=SERIES[2], height=0.6)
    axes[1].set_title("Top-15 individual features")
    save(fig, fig_dir / "xai_permutation_importance.png")

    # integrated gradients on positive windows, split by failure type
    pos_rows = rows[y == 1][:600]
    ftype = np.array([""] * len(p.y), dtype=object)
    for r in events[events.kind == "failure"].itertuples():
        tf = p.offsets[r.asset] + int(r.failure_idx)
        ftype[max(0, tf - p.horizon): tf] = r.type
    attr = xai.integrated_gradients(art.model, xai.gather_windows(p.X, pos_rows, L))
    heat = xai.group_attribution(attr, names)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(heat.T.values, aspect="auto", cmap="Blues")
    ax.grid(False)
    ax.set_yticks(range(heat.shape[1]), heat.columns)
    ticks = np.linspace(0, L - 1, 7).astype(int)
    ax.set_xticks(ticks, [f"-{(L - 1 - t) * 15 / 60:.1f}h" for t in ticks])
    ax.set_xlabel("position in input window (0 h = prediction time)")
    ax.set_title("Integrated gradients: mean |attribution| by sensor group and time (positive test windows)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, fig_dir / "xai_temporal_attribution.png")

    share = {}
    for t in FAILURE_TYPES:
        sel = ftype[pos_rows] == t
        if sel.sum() >= 5:
            h = xai.group_attribution(attr[sel], names).sum()
            share[t] = (h / h.sum()).to_dict()
    sh = pd.DataFrame(share).fillna(0)
    out["ig_share_by_failure_type"] = sh.to_dict()
    if len(sh.columns):
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(sh.index))
        w = 0.8 / len(sh.columns)
        for k, t in enumerate(sh.columns):
            ax.bar(x + k * w, sh[t], width=w * 0.92, color=TYPE_COLOR[t], label=t)
        ax.set_xticks(x + 0.4 - w / 2, sh.index, rotation=20)
        ax.set_ylabel("share of total |attribution|")
        ax.set_title("Which sensors drive the warning, per failure mechanism (integrated gradients, associational)")
        ax.legend()
        save(fig, fig_dir / "xai_attribution_by_failure_type.png")

    att = xai.attention_profile(art.model, W)
    if att is not None:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        pos_x = np.arange(-L + 1, 1) * 15 / 60
        ax.plot(pos_x, att[y == 1].mean(0), color=SERIES[1], label="failure ≤ H windows")
        ax.plot(pos_x, att[y == 0].mean(0), color=SERIES[0], label="normal windows")
        ax.set_xlabel("hours before prediction time"); ax.set_ylabel("mean attention weight")
        ax.set_title("Temporal attention profile (test)"); ax.legend()
        save(fig, fig_dir / "xai_attention_profile.png")
        out["attention_last_6_steps_share_pos"] = float(att[y == 1][:, -6:].sum(1).mean())
    return out


def shap_baseline(cfg: dict, fig_dir) -> dict:
    from src.models.baselines import gbdt_model

    p = prepare(cfg, "stationary", cfg["labels"]["horizon_steps"])
    m, kind = gbdt_model(cfg["seed"])
    tr = p.idx["train"]
    m.fit(p.X[tr], p.y[tr].astype(int))
    res = xai.shap_tree(m, p.X[p.idx["test"]], p.feature_names)
    if res is None:
        return {}
    df, (sv, Xs) = res
    top = df.head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].barh(top.feature, top.mean_abs_shap, color=SERIES[0], height=0.6)
    axes[0].set_title(f"SHAP mean |value| — {kind} baseline (test)")
    names = list(top.feature)[::-1][:8]
    for k, n in enumerate(names):
        j = p.feature_names.index(n)
        axes[1].scatter(sv[:, j], np.full(len(sv), k) + np.random.default_rng(k).normal(0, 0.08, len(sv)),
                        c=np.clip(Xs[:, j], -3, 3), cmap="coolwarm", s=4, alpha=0.6)
    axes[1].set_yticks(range(len(names)), names); axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_title("SHAP values (colour = feature value, red = high)")
    save(fig, fig_dir / "xai_shap_gbdt.png")
    return {"shap_top10": df.head(10).to_dict(orient="records")}


# --------------------------------------------------------------------------- generalisation
def per_period_metrics(p: Prepared, probs: np.ndarray, thr: float) -> pd.DataFrame:
    idx = p.idx["test"]
    d = pd.DataFrame({"asset": p.asset[idx], "month": pd.to_datetime(p.timestamps[idx]).to_period("M").astype(str),
                      "y": p.y[idx], "p": probs})
    out = []
    for (a, mth), g in d.groupby(["asset", "month"]):
        if g.y.sum() > 0:
            m = classification_metrics(g.y.values, g.p.values, thr)
            out.append({"asset": a, "month": mth, "positives": int(g.y.sum()), "pr_auc": m["pr_auc"],
                        "recall": m["recall"], "precision": m["precision"]})
    return pd.DataFrame(out)


def failure_case_analysis(per: pd.DataFrame, fa: pd.DataFrame, events: pd.DataFrame) -> dict:
    missed = per[~per.detected]
    nu = events[events.kind == "nuisance"]
    overlaps = []
    for r in fa.itertuples():
        n = nu[(nu.asset == r.asset) & (nu.onset_time <= r.end + pd.Timedelta(hours=2)) &
               (nu.end_time >= r.start - pd.Timedelta(hours=2))]
        overlaps.append(n.type.iloc[0] if len(n) else "no injected event (real-data behaviour / outage recovery)")
    fa = fa.assign(coincides_with=overlaps)
    return {"missed_by_type": missed.type.value_counts().to_dict(),
            "missed_detectable": missed[missed.type != "sudden"][["asset", "type", "precursor_min",
                                                                  "failure_time"]].astype(str).to_dict(orient="records"),
            "false_alarm_causes": fa.coincides_with.value_counts().to_dict(),
            "false_alarm_duration_steps_median": float(fa.duration_steps.median()) if len(fa) else float("nan")}
