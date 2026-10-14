"""Exploratory data analysis -> reports/figures/eda_*.png + reports/eda_summary.json."""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from src.data.dataset import load_processed
from src.data.download import load_raw
from src.data.pipeline import prepare
from src.data.simulation import STEPS_PER_DAY
from src.utils.config import resolve
from src.utils.logging import get_logger
from src.utils.plotting import DIVERGING, FAILURE_TYPES, INK2, SERIES, TYPE_COLOR, apply_style, plt, save

log = get_logger("eda")
SPLIT_SHADE = {"val": "#eda10022", "test": "#e3494822"}


def _shade_splits(ax, p, asset, ts):
    b = p.bounds[asset]
    for s in ("val", "test"):
        lo, hi = b.period(s)
        ax.axvspan(ts[lo], ts[hi - 1], color=SPLIT_SHADE[s], lw=0)


def run_eda(cfg: dict) -> dict:
    apply_style()
    fig_dir = resolve(cfg, "figures_dir")
    dfs, events = load_processed(cfg)
    p = prepare(cfg, "raw_engineered", keep_raw=True)
    cols = cfg["dataset"]["sensor_cols"]
    summary: dict = {"assets": {}}

    # ---------- data quality (on the REAL raw data) ----------
    for a in dfs:
        raw = load_raw(cfg, a)
        z = (raw[cols] - raw[cols].mean()) / raw[cols].std()
        dt = raw.timestamp.diff().dropna()
        summary["assets"][a] = {
            "rows": len(raw), "start": str(raw.timestamp.min()), "end": str(raw.timestamp.max()),
            "missing_values": int(raw[cols].isna().sum().sum()),
            "duplicate_timestamps": int(raw.timestamp.duplicated().sum()),
            "sampling_interval_minutes": dt.dt.total_seconds().div(60).value_counts().to_dict(),
            "outliers_abs_z_gt_4": {c: int((z[c].abs() > 4).sum()) for c in cols},
            "flatline_runs_ge_4h": {c: int(((raw[c].diff() == 0).astype(int)
                                           .groupby((raw[c].diff() != 0).cumsum()).sum() >= 16).sum()) for c in cols},
            "describe": raw[cols].describe().round(3).to_dict(),
        }

    # 1. sensor time-series (real data, daily means) with split shading
    fig, axes = plt.subplots(len(cols), 2, figsize=(14, 12), sharex=True)
    for j, a in enumerate(dfs):
        raw = load_raw(cfg, a).set_index("timestamp")[cols].resample("1D").mean()
        ts = dfs[a].timestamp.to_numpy()
        for i, c in enumerate(cols):
            ax = axes[i, j]
            ax.plot(raw.index, raw[c], color=SERIES[0], lw=0.9)
            _shade_splits(ax, p, a, ts)
            ax.set_ylabel(c)
            if i == 0:
                ax.set_title(f"{a} — daily mean of real measurements (val = amber, test = red band)")
    save(fig, fig_dir / "eda_sensor_timeseries.png")

    # 2. failure-event timeline
    fig, ax = plt.subplots(figsize=(14, 3.6))
    for k, a in enumerate(dfs):
        ev = events[events.asset == a]
        nu = ev[ev.kind == "nuisance"]
        ax.scatter(nu.onset_time, np.full(len(nu), k + 0.25), marker="|", s=60, color="#b0afa9",
                   label="benign transient" if k == 0 else None)
        for t in FAILURE_TYPES:
            f = ev[(ev.kind == "failure") & (ev.type == t)]
            ax.scatter(f.failure_time, np.full(len(f), k), s=22, color=TYPE_COLOR[t], label=t if k == 0 else None,
                       edgecolor="white", linewidth=0.6)
        _shade_splits(ax, p, a, dfs[a].timestamp.to_numpy())
    ax.set_yticks([0, 1], list(dfs))
    ax.set_ylim(-0.6, 1.8)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Simulated failure events (dots) and benign transients (ticks) on the real timeline", pad=28)
    save(fig, fig_dir / "eda_failure_timeline.png")

    # 3. correlation heatmaps (training period only, real data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, a in zip(axes, dfs):
        raw = load_raw(cfg, a).iloc[: p.bounds[a].train_end]
        corr = raw[cols].corr()
        im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=45)
        ax.set_yticks(range(len(cols)), cols)
        ax.grid(False)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iloc[i, j]) > 0.6 else "#0b0b0b")
        ax.set_title(f"{a}: Pearson correlation (train period)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    save(fig, fig_dir / "eda_correlation_heatmap.png")

    # 4. class distribution per split
    rows = []
    for s in ("train", "val", "test"):
        y = p.y[p.idx[s]]
        rows.append({"split": s, "negatives": int((y == 0).sum()), "positives": int((y == 1).sum()),
                     "positive_rate": float(y.mean())})
    cd = pd.DataFrame(rows)
    summary["class_distribution_H8"] = rows
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(cd.split, cd.negatives, color=SERIES[0], width=0.55, label="normal (y=0)")
    axes[0].bar(cd.split, cd.positives, bottom=cd.negatives, color=SERIES[1], width=0.55, label="failure ≤ 2 h (y=1)")
    axes[0].legend(); axes[0].set_title("Target rows per split")
    axes[1].bar(cd.split, cd.positive_rate * 100, color=SERIES[1], width=0.55)
    for x, v in zip(cd.split, cd.positive_rate):
        axes[1].text(x, v * 100 + 0.03, f"{v*100:.2f}%", ha="center", color=INK2)
    axes[1].set_title("Positive rate (%) — class imbalance ≈ 1 : 60")
    save(fig, fig_dir / "eda_class_distribution.png")

    # 5. feature distributions: train vs test (drift), real data
    fig, axes = plt.subplots(2, len(cols), figsize=(16, 5))
    for i, a in enumerate(dfs):
        raw = load_raw(cfg, a)
        b = p.bounds[a]
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.hist(raw[c].iloc[: b.train_end], bins=50, density=True, color=SERIES[0], alpha=0.6, label="train")
            ax.hist(raw[c].iloc[b.val_end:], bins=50, density=True, color=SERIES[1], alpha=0.6, label="test")
            ax.set_title(f"{a} {c}", fontsize=9)
            ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Distribution shift: training vs test period (real measurements)", fontweight="bold")
    save(fig, fig_dir / "eda_feature_distributions.png")

    # 6. seasonal / temporal patterns + drift
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for k, a in enumerate(dfs):
        raw = load_raw(cfg, a)
        raw["hour"] = raw.timestamp.dt.hour
        prof = raw.groupby("hour")[["OT", "HUFL"]].mean()
        axes[0].plot(prof.index, (prof.OT - prof.OT.mean()) / raw.OT.std(), color=SERIES[k], label=f"{a} OT")
        axes[0].plot(prof.index, (prof.HUFL - prof.HUFL.mean()) / raw.HUFL.std(), color=SERIES[k], ls="--",
                     label=f"{a} HUFL")
        m = raw.set_index("timestamp")[["OT"]].resample("30D").mean()
        axes[1].plot(m.index, m.OT, color=SERIES[k], marker="o", ms=3, label=a)
        r = raw.set_index("timestamp")["OT"].rolling("30D").std()
        axes[2].plot(r.index, r, color=SERIES[k], label=a)
    axes[0].set_title("Daily profile (z-units)"); axes[0].set_xlabel("hour of day"); axes[0].legend()
    axes[1].set_title("Oil temperature, 30-day mean (seasonality / drift)"); axes[1].legend()
    axes[2].set_title("Oil temperature, 30-day rolling std"); axes[2].legend()
    save(fig, fig_dir / "eda_seasonality_drift.png")

    # 7. pre-failure vs normal behaviour (training period, engineered features)
    feats = ["OT_resid", "HUFL_volratio", "HULL_spikes16", "OT_diff16"]
    rf = p.raw_features
    W = 24 * 4
    fig, axes = plt.subplots(1, len(feats), figsize=(16, 3.6))
    tr_ev = events[(events.kind == "failure")]
    for j, f in enumerate(feats):
        ax = axes[j]
        for t in FAILURE_TYPES:
            curves = []
            for r in tr_ev[tr_ev.type == t].itertuples():
                if r.failure_idx >= p.bounds[r.asset].train_end or r.failure_idx < W + 200:
                    continue
                off = p.offsets[r.asset]
                seg = rf[f].iloc[off + r.failure_idx - W: off + r.failure_idx].to_numpy()
                sd = rf[f].iloc[off: off + p.bounds[r.asset].train_end].std()
                curves.append(seg / (sd + 1e-9))
            if curves:
                c = np.nanmedian(np.vstack(curves), axis=0)
                ax.plot(np.arange(-W, 0) / 4, c, color=TYPE_COLOR[t], label=t)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{f} (median, σ-units)")
        ax.set_xlabel("hours before failure")
    axes[0].legend()
    fig.suptitle("Pre-failure signatures (training failures) — sudden failures show no precursor",
                 fontweight="bold")
    save(fig, fig_dir / "eda_prefailure_signatures.png")

    # 8. example events: real vs observed (simulation layer made explicit)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    a = list(dfs)[0]
    df = dfs[a]
    for ax, t in zip(axes, FAILURE_TYPES):
        r = events[(events.asset == a) & (events.type == t)].iloc[2]
        ch = {"thermal_runaway": "OT", "load_oscillation": "HUFL", "intermittent_arcing": "HULL", "sudden": "OT"}[t]
        lo, hi = int(r.failure_idx) - 24 * 4, int(r.failure_idx) + 12 * 4
        seg = df.iloc[lo:hi]
        ax.plot(seg.timestamp, seg[ch] - seg[f"sim_delta_{ch}"], color=SERIES[0], label="real measurement")
        ax.plot(seg.timestamp, seg[ch], color=SERIES[1], label="observed (real + injected)", lw=1)
        ax.axvline(r.failure_time, color="#d03b3b", ls="--", lw=1)
        if t != "sudden":
            ax.axvline(r.onset_time, color=INK2, ls=":", lw=1)
        ax.set_title(f"{t} — {ch}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend()
    fig.suptitle("Example events (dotted = precursor onset, dashed = failure)", fontweight="bold")
    save(fig, fig_dir / "eda_example_events.png")

    # 9. univariate feature relevance (mutual information, TRAIN rows only)
    from sklearn.feature_selection import mutual_info_classif
    tr = p.idx["train"]
    rng = np.random.default_rng(0)
    pos, neg = tr[p.y[tr] == 1], tr[p.y[tr] == 0]
    sel = np.concatenate([pos, rng.choice(neg, 20 * len(pos), replace=False)])
    mi = pd.Series(mutual_info_classif(p.X[sel], p.y[sel].astype(int), random_state=0),
                   index=p.feature_names).sort_values()
    summary["mutual_information_top15"] = mi.tail(15)[::-1].round(4).to_dict()
    fig, ax = plt.subplots(figsize=(7, 6))
    top = mi.tail(20)
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    ax.set_title("Mutual information with label (train rows, H = 2 h)")
    save(fig, fig_dir / "eda_feature_importance_mi.png")

    fe = events[events.kind == "failure"]
    summary["events"] = {"failures": int(len(fe)), "by_type": fe.type.value_counts().to_dict(),
                         "nuisance": int((events.kind == "nuisance").sum()),
                         "mean_days_between_failures": float(
                             np.mean([len(d) / STEPS_PER_DAY / max(1, (events[(events.asset == a) & (events.kind == 'failure')].shape[0]))
                                      for a, d in dfs.items()])),
                         "precursor_hours_mean": float(fe[fe.type != "sudden"].params.apply(
                             lambda s: ast.literal_eval(s)["precursor_steps"] if isinstance(s, str) else s["precursor_steps"]).mean() / 4)}
    (resolve(cfg, "reports_dir") / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("EDA figures written to %s", fig_dir)
    return summary
