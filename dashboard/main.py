"""Streamlit decision-support dashboard.  Launch with:  streamlit run app.py"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.dataset import load_processed, make_labels
from src.preprocessing.splits import compute_bounds
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import load_artifacts
from src.streaming.engine import StreamingPredictor
from src.utils.config import load_config, resolve
from src.utils.plotting import FAILURE_TYPES, SERIES, STATUS, TYPE_COLOR

ICON = {"NORMAL": "✅", "WARNING": "⚠️", "CRITICAL": "🛑", "OFFLINE": "⏸️", "WARMUP": "⏳"}
STATUS_ALL = STATUS | {"OFFLINE": "#8a8984", "WARMUP": "#8a8984"}
SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SENSOR_LABEL = {"HUFL": "High-voltage useful load", "HULL": "High-voltage useless (reactive) load",
                "MUFL": "Mid-voltage useful load", "MULL": "Mid-voltage useless load",
                "LUFL": "Low-voltage useful load", "LULL": "Low-voltage useless load", "OT": "Oil temperature (°C)"}


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def _cfg():
    return load_config()


@st.cache_resource
def _art():
    return load_artifacts(_cfg())


@st.cache_data
def _data():
    dfs, ev = load_processed(_cfg())
    return dfs, ev


def _json(key: str, name: str) -> dict:
    p = resolve(_cfg(), key) / name
    return json.loads(p.read_text()) if p.exists() else {}


def _csv(key: str, name: str) -> pd.DataFrame:
    p = resolve(_cfg(), key) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _fig_path(name: str):
    p = resolve(_cfg(), "figures_dir") / name
    return str(p) if p.exists() else None


def _layout(fig: go.Figure, h: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title,
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def badge(status: str, big: bool = False) -> str:
    c = STATUS_ALL.get(status, "#8a8984")
    size = "1.6rem" if big else "0.95rem"
    return (f"<span style='display:inline-block;padding:0.25em 0.8em;border-radius:6px;border:2px solid {c};"
            f"font-weight:700;font-size:{size}'>{ICON.get(status, '')} {status}</span>")


# ------------------------------------------------------------------ pages
def page_overview():
    fm = _json("reports_dir", "final_metrics.json")
    meta = _art().meta
    st.subheader("System overview")
    st.info("**Decision-support only.** Probabilities indicate elevated risk of an impending fault within the "
            f"next {meta['horizon'] * 15 // 60} h. They do not guarantee detection or prevention of outages. "
            "Measurements are real (ETT transformers); **failure events are simulated** on top of them — see README.")
    if not fm:
        st.warning("Run `python evaluate.py` to populate metrics.")
        return
    t = fm["test"]
    c = st.columns(6)
    c[0].metric("Test PR-AUC", f"{t['pr_auc']:.3f}")
    c[1].metric("Recall / Precision", f"{t['recall']:.2f} / {t['precision']:.2f}")
    c[2].metric("Failures warned (detectable)", f"{t['event_recall_detectable']:.0%}")
    c[3].metric("Median lead time", f"{t['lead_time_median_min'] / 60:.1f} h")
    c[4].metric("False alarms / day", f"{t['false_alarms_per_day']:.2f}")
    eff = pd.DataFrame(fm["efficiency"]).set_index("model")
    lat = eff.loc[meta["active_model"], "cpu_latency_ms_b1"] if meta["active_model"] in eff.index else np.nan
    c[5].metric("CPU latency / step", f"{lat:.2f} ms")
    st.markdown(f"**Model:** `{meta['active_model']}` ({meta['model_type']}, {meta['seq_len']} × 15-min window, "
                f"{meta['n_features']} causal features) · **WARNING** ≥ {meta['active_threshold']:.3f} · "
                f"**CRITICAL** ≥ {meta['critical_threshold']:.3f} · alarm needs "
                f"{_cfg()['threshold']['persistence_k']} consecutive steps")
    st.markdown("```\nsensor stream → ring buffer → causal features → scaler (train-fit) → recurrent model "
                "→ P(failure ≤ H) → alert policy → NORMAL / WARNING / CRITICAL\n```")
    p = _fig_path("probability_timeline_ETTm1.png")
    if p:
        st.image(p, caption="Example: test-period probability, thresholds and failures (ETTm1)")


def page_live():
    cfg, art = _cfg(), _art()
    dfs, ev = _data()
    st.subheader("Live sensor stream (real streaming engine, replaying the held-out test period)")
    c1, c2, c3 = st.columns([1, 2, 1])
    asset = c1.selectbox("Asset", list(dfs), key="live_asset")
    df = dfs[asset]
    b = compute_bounds(len(df), cfg, art.fe.lookback)
    lo, hi = b.period("test")
    fails = ev[(ev.asset == asset) & (ev.kind == "failure") & (ev.failure_idx >= lo + 200) & (ev.failure_idx < hi)]
    opts = {f"{r.failure_time:%Y-%m-%d %H:%M} — {r.type} (start 18 h before)": int(r.failure_idx) - 72
            for r in fails.itertuples()}
    choice = c2.selectbox("Jump to (start of replay)", ["start of test period"] + list(opts))
    start = lo if choice == "start of test period" else opts[choice]
    speed = c3.select_slider("Steps per tick", [1, 4, 8, 16], value=4)
    key = (asset, start)
    if st.session_state.get("live_key") != key:
        sp = StreamingPredictor(art, asset, AlertPolicy.from_config(cfg, art.meta), cfg["threshold"]["merge_gap_steps"])
        warm = art.fe.lookback + art.seq_len - 1
        for i in range(max(0, start - warm), start):
            r = df.iloc[i]
            sp.update(r["timestamp"], r, offline=bool(r["offline"]))
        st.session_state.update(live_key=key, sp=sp, pos=start, hist=[])
    y = make_labels(df["failure"].to_numpy(), art.meta["horizon"])
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    play = b1.toggle("▶ Auto-play", key="autoplay")
    step = b2.button("Step ⏭")
    if b3.button("Reset ↺"):
        st.session_state.pop("live_key", None)
        st.rerun()
    n = speed if (play or step) else 0
    sp = st.session_state.sp
    for _ in range(n):
        i = st.session_state.pos
        if i >= hi:
            break
        r = df.iloc[i]
        o = sp.update(r["timestamp"], r, offline=bool(r["offline"]))
        st.session_state.hist.append(o | {"failure_event": int(r["failure"]), "actual": y[i],
                                          **{c: r[c] for c in SENSORS}})
        st.session_state.pos += 1
    h = pd.DataFrame(st.session_state.hist)
    if h.empty:
        st.caption("Press **Step** or **Auto-play** to stream observations into the engine.")
        return
    cur = h.iloc[-1]
    k = st.columns([1.3, 1, 1, 1.4])
    k[0].markdown(f"**Risk status**<br>{badge(cur.status, big=True)}", unsafe_allow_html=True)
    k[1].metric("Failure probability", "—" if np.isnan(cur.failure_probability) else f"{cur.failure_probability:.3f}")
    k[2].metric("Engine latency", f"{cur.latency_ms:.1f} ms")
    if cur.warning_since is not None and not pd.isna(cur.warning_since):
        dur = (pd.Timestamp(cur.timestamp) - pd.Timestamp(cur.warning_since)).total_seconds() / 3600
        k[3].markdown(f"**Early warning** active since<br>`{pd.Timestamp(cur.warning_since):%Y-%m-%d %H:%M}` "
                      f"({dur:.1f} h)", unsafe_allow_html=True)
    else:
        k[3].markdown(f"**Early warning**<br>none active · `{pd.Timestamp(cur.timestamp):%Y-%m-%d %H:%M}`",
                      unsafe_allow_html=True)
    win = h.tail(96 * 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.timestamp, y=win.failure_probability, name="P(failure ≤ H)",
                             line=dict(color=SERIES[0], width=2)))
    fig.add_hline(y=art.meta["active_threshold"], line=dict(color=STATUS["WARNING"], dash="dash"),
                  annotation_text="WARNING")
    fig.add_hline(y=art.meta["critical_threshold"], line=dict(color=STATUS["CRITICAL"], dash="dash"),
                  annotation_text="CRITICAL")
    for t in win[win.failure_event == 1].timestamp:
        fig.add_vline(x=t, line=dict(color=STATUS["CRITICAL"], width=2))
    st.plotly_chart(_layout(fig, 280, "Failure probability (last 48 h)").update_yaxes(range=[0, 1.05]),
                    use_container_width=True)
    sens = st.multiselect("Sensors", SENSORS, default=["OT", "HUFL", "HULL"], key="live_sens")
    fig = go.Figure()
    for _j, c in enumerate(sens):
        s = win[c]
        fig.add_trace(go.Scatter(x=win.timestamp, y=(s - s.mean()) / (s.std() + 1e-9), name=c,
                                 line=dict(color=SERIES[SENSORS.index(c)], width=1.5)))
    st.plotly_chart(_layout(fig, 260, "Sensor stream (z-score within window)"), use_container_width=True)
    if play:
        time.sleep(0.25)
        st.rerun()


def page_trends():
    dfs, ev = _data()
    st.subheader("Sensor trends")
    c1, c2, c3 = st.columns([1, 2, 1])
    asset = c1.selectbox("Asset", list(dfs), key="tr_asset")
    df = dfs[asset]
    sens = c2.multiselect("Sensors", SENSORS, default=["OT", "HUFL"], key="tr_sens")
    show_real = c3.toggle("Overlay real (pre-injection) signal", value=True)
    dmin, dmax = df.timestamp.min().date(), df.timestamp.max().date()
    rng = st.slider("Date range", dmin, dmax, (pd.Timestamp("2018-04-01").date(), pd.Timestamp("2018-04-15").date()))
    m = (df.timestamp.dt.date >= rng[0]) & (df.timestamp.dt.date <= rng[1])
    d = df[m]
    for c in sens:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.timestamp, y=d[c], name="observed", line=dict(color=SERIES[1], width=1.2)))
        if show_real:
            fig.add_trace(go.Scatter(x=d.timestamp, y=d[c] - d[f"sim_delta_{c}"], name="real measurement",
                                     line=dict(color=SERIES[0], width=1.2)))
        e = ev[(ev.asset == asset) & (ev.kind == "failure")]
        for r in e[(e.failure_time >= d.timestamp.min()) & (e.failure_time <= d.timestamp.max())].itertuples():
            fig.add_vline(x=r.failure_time, line=dict(color=STATUS["CRITICAL"], width=1.5))
        st.plotly_chart(_layout(fig, 240, SENSOR_LABEL[c]), use_container_width=True)


def page_explain():
    fm = _json("reports_dir", "final_metrics.json")
    st.subheader("Model explanation")
    st.caption("Feature attributions are **associational**: they show what the model relies on, not physical causes.")
    if fm.get("explainability"):
        g = pd.DataFrame(fm["explainability"]["permutation_groups"])
        fig = go.Figure(go.Bar(x=g.pr_auc_drop[::-1], y=g.group[::-1], orientation="h",
                               marker_color=SERIES[0], error_x=dict(array=g["std"][::-1])))
        st.plotly_chart(_layout(fig, 340, "Permutation importance: drop in test PR-AUC when a sensor group is shuffled"),
                        use_container_width=True)
    for f, cap in [("xai_attribution_by_failure_type.png", "Which sensors drive warnings for each failure mechanism"),
                   ("xai_temporal_attribution.png", "When in the input window the evidence sits"),
                   ("xai_attention_profile.png", "Temporal attention"),
                   ("xai_shap_gbdt.png", "SHAP for the gradient-boosting baseline")]:
        p = _fig_path(f)
        if p:
            st.image(p, caption=cap)


def page_history():
    dfs, ev = _data()
    st.subheader("Historical failures")
    per = _csv("reports_dir", "test_per_failure.csv")
    c1, c2 = st.columns(2)
    asset = c1.multiselect("Asset", list(dfs), default=list(dfs))
    types = c2.multiselect("Failure type", FAILURE_TYPES, default=FAILURE_TYPES)
    e = ev[(ev.kind == "failure") & ev.asset.isin(asset) & ev.type.isin(types)].copy()
    fig = go.Figure()
    for t in types:
        s = e[e.type == t]
        fig.add_trace(go.Scatter(x=s.failure_time, y=s.asset, mode="markers", name=t,
                                 marker=dict(color=TYPE_COLOR[t], size=9, line=dict(color="white", width=1))))
    st.plotly_chart(_layout(fig, 260, "All simulated failures on the real timeline"), use_container_width=True)
    st.markdown("**Test-period failures and whether the model warned in time**")
    if len(per):
        per = per[per.asset.isin(asset) & per.type.isin(types)]
        per["lead_time_h"] = (per.lead_time_min / 60).round(2)
        per["warned"] = np.where(per.detected, "✅ warned", "❌ missed")
        st.dataframe(per[["asset", "failure_time", "type", "precursor_min", "warned", "lead_time_h", "warning_time"]],
                     use_container_width=True, hide_index=True)


def page_lead():
    fm = _json("reports_dir", "final_metrics.json")
    st.subheader("Detection lead time")
    if fm:
        t = fm["test"]
        c = st.columns(5)
        c[0].metric("Failures warned", f"{t['n_detected']} / {t['n_failures']}")
        c[1].metric("Mean", f"{t['lead_time_mean_min'] / 60:.1f} h")
        c[2].metric("Median", f"{t['lead_time_median_min'] / 60:.1f} h")
        c[3].metric("Min", f"{t['lead_time_min_min'] / 60:.2f} h")
        c[4].metric("Max", f"{t['lead_time_max_min'] / 60:.1f} h")
    for f in ("lead_time_timeline.png", "lead_time_distribution.png"):
        p = _fig_path(f)
        if p:
            st.image(p)


def page_performance():
    fm = _json("reports_dir", "final_metrics.json")
    st.subheader("Model performance (selected model)")
    if fm:
        rows = []
        for s in ("val", "test"):
            m = fm[s]
            rows.append({"split": s, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                                           "pr_auc_detectable", "specificity", "fpr", "fnr",
                                                           "event_recall_detectable", "false_alarms_per_day",
                                                           "alarm_time_fraction")}})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
        st.caption("Accuracy is shown for completeness only — with ~1.6 % positives it is not informative.")
    cols = st.columns(2)
    for i, f in enumerate(("final_pr_roc_curves.png", "final_confusion_matrix.png", "threshold_tradeoff.png",
                           "calibration.png")):
        p = _fig_path(f)
        if p:
            cols[i % 2].image(p)


def page_compare():
    st.subheader("Model comparison")
    res = _csv("experiments_dir", "results.csv")
    eff = _csv("reports_dir", "efficiency_table.csv")
    if len(eff):
        st.markdown("**Efficiency vs performance (measured on this machine, CPU, batch = 1)**")
        fig = go.Figure()
        for j, r in enumerate(eff.itertuples()):
            fig.add_trace(go.Scatter(x=[r.cpu_latency_ms_b1], y=[r.val_pr_auc], mode="markers+text", text=[r.model],
                                     textposition="top center", name=r.model,
                                     marker=dict(size=10 + r.size_kb / 15, color=SERIES[j % len(SERIES)])))
        fig.update_xaxes(title="CPU latency per step (ms)")
        fig.update_yaxes(title="validation PR-AUC")
        st.plotly_chart(_layout(fig, 360).update_layout(showlegend=False, hovermode="closest"),
                        use_container_width=True)
        st.dataframe(eff.round(4), hide_index=True, use_container_width=True)
    if len(res):
        st.markdown("**All experiment runs** (selection used validation columns only)")
        fam = st.multiselect("Family", sorted(res.family.unique()), default=sorted(res.family.unique()))
        cols = ["version", "model", "features", "sequence_length", "prediction_horizon", "loss", "val_pr_auc",
                "pr_auc", "recall", "precision", "f1", "event_recall_detectable", "lead_time_median_min",
                "false_alarms_per_day", "inference_latency_ms", "parameters"]
        st.dataframe(res[res.family.isin(fam)][[c for c in cols if c in res]].round(4), hide_index=True,
                     use_container_width=True)


PAGES = {"System overview": page_overview, "Live stream & early warning": page_live, "Sensor trends": page_trends,
         "Model explanation": page_explain, "Historical failures": page_history, "Detection lead time": page_lead,
         "Model performance": page_performance, "Model comparison": page_compare}


def run():
    st.set_page_config(page_title="Grid Failure Early Warning", page_icon="⚡", layout="wide")
    st.title("⚡ Transformer failure early-warning")
    page = st.sidebar.radio("View", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.markdown("Risk levels  \n" + "  \n".join(badge(s) for s in ("NORMAL", "WARNING", "CRITICAL")),
                        unsafe_allow_html=True)
    PAGES[page]()
