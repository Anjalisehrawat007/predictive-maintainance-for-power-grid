"""Final model-selection protocol: 6 candidates × 3 seeds × 5 fresh validation realisations."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import torch

from src.final.common import (
    FINAL,
    REAL_SEEDS,
    SEEDS,
    candidates,
    featurizer,
    load_phase2_model,
    phase2_threshold,
    predict,
    realisation,
    reference_prepared,
    split_metrics,
)
from src.phase2.common import P2_EXP, phase2_config

KEYS = ["pr_auc", "pr_auc_detectable", "recall", "precision", "f1", "false_alarms_per_day", "lead_time_median_min",
        "event_recall_detectable"]


def latency(model, n_features: int, n: int = 400) -> dict:
    torch.set_num_threads(1)
    x = torch.randn(1, 48, n_features)
    with torch.no_grad():
        for _ in range(30):
            model(x)
        ts = []
        import time
        for _ in range(n):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def run() -> pd.DataFrame:
    cfg = phase2_config()
    p = reference_prepared(cfg)
    art = featurizer(p)
    cands = candidates()
    models = {(name, s): (load_phase2_model(g, c, st, s, p.X.shape[1]), phase2_threshold(g, c, s))
              for name, (g, c, st) in cands.items() for s in SEEDS}
    rows = []
    for k, rs in enumerate(REAL_SEEDS):
        q = realisation(cfg, art, rs)
        for (name, s), (m, thr) in models.items():
            pv = predict(m, q, q.idx["val"])
            met = split_metrics(cfg, q, "val", pv, thr)
            rows.append({"candidate": name, "seed": s, "realisation": k} | {kk: met.get(kk) for kk in KEYS})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "protocol_runs.csv", index=False)
    res = pd.read_csv(P2_EXP / "results.csv")
    summ = []
    for name, (g, c, _st) in cands.items():
        d = df[df.candidate == name]
        m0 = models[(name, SEEDS[0])][0]
        buf = io.BytesIO(); torch.save(m0.state_dict(), buf)
        r = res[(res.group == g) & (res.candidate == c)]
        row = {"candidate": name, "n": len(d), "params": int(sum(x.numel() for x in m0.parameters())),
               "size_kb": buf.getbuffer().nbytes / 1024} | latency(m0, p.X.shape[1])
        for kk in KEYS:
            v = d[kk].astype(float)
            row |= {f"{kk}_mean": v.mean(), f"{kk}_std": v.std(), f"{kk}_min": v.min(), f"{kk}_max": v.max()}
        row |= {"test_pr_auc_mean": r.test_pr_auc.mean(), "test_pr_auc_std": r.test_pr_auc.std(),
                "test_rs_pr_auc_mean": r.test_rs_pr_auc.mean()}
        summ.append(row)
    s = pd.DataFrame(summ)
    inc = df[df.candidate == "GRU + attention (incumbent)"].groupby("seed").pr_auc.mean()
    dec = []
    for name in s.candidate:
        x = df[df.candidate == name].groupby("seed").pr_auc.mean()
        diff = x.mean() - inc.mean()
        se = float(np.sqrt(x.var(ddof=1) / len(x) + inc.var(ddof=1) / len(inc)))
        dec.append({"candidate": name, "diff_vs_incumbent": diff, "se": se, "required": max(0.01, se),
                    "replaces": bool(name != "GRU + attention (incumbent)" and diff > max(0.01, se))})
    s = s.merge(pd.DataFrame(dec), on="candidate")
    (FINAL / "protocol_summary.json").write_text(s.replace({np.nan: None}).to_json(orient="records", indent=1))
    return s


def write_results(s: pd.DataFrame) -> None:
    def f(r, k, nd=3):
        return f"{r[k + '_mean']:.{nd}f} ± {r[k + '_std']:.{nd}f} [{r[k + '_min']:.{nd}f}, {r[k + '_max']:.{nd}f}]"
    lines = ["\n## Results (appended by `python -m src.final.run`)\n",
             "Fresh-validation metrics over 3 seeds × 5 realisations (mean ± std [min, max]); thresholds are each run's "
             "validation-selected threshold. Test columns are for reporting only.\n",
             "| Candidate | PR-AUC | PR-AUC (predictable) | Recall | Precision | F1 | FA/day | Lead (min) | Test PR-AUC | Params | Size KB | Latency p50 / p95 / p99 (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.sort_values("pr_auc_mean", ascending=False).iterrows():
        lines.append(f"| {r.candidate} | {f(r, 'pr_auc')} | {f(r, 'pr_auc_detectable')} | {f(r, 'recall')} | {f(r, 'precision')} | "
                     f"{f(r, 'f1')} | {f(r, 'false_alarms_per_day')} | {r['lead_time_median_min_mean']:.0f} ± {r['lead_time_median_min_std']:.0f} | "
                     f"{r.test_pr_auc_mean:.3f} ± {r.test_pr_auc_std:.3f} | {r.params:,} | {r.size_kb:.0f} | "
                     f"{r.p50_ms:.2f} / {r.p95_ms:.2f} / {r.p99_ms:.2f} |")
    lines += ["", "**Decision (rule above):**\n", "| Candidate | Δ mean PR-AUC vs incumbent | SE | required | replaces? |", "|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.candidate} | {r.diff_vs_incumbent:+.4f} | {r.se:.4f} | {r.required:.4f} | {'**yes**' if r.replaces else 'no'} |")
    win = s[s.replaces]
    lines.append(f"\n**Outcome:** {'replace incumbent by ' + win.iloc[0].candidate if len(win) else 'no candidate meets the rule — GRU + attention is kept.'}\n")
    p = FINAL.parent / "model_selection_protocol.md"
    txt = p.read_text().split("\n## Results (appended")[0]
    p.write_text(txt.rstrip() + "\n" + "\n".join(lines))
    (FINAL / "protocol_decision.json").write_text(json.dumps({"winner": None if win.empty else win.iloc[0].candidate}))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    s = run()
    write_results(s)
    print(s[["candidate", "pr_auc_mean", "pr_auc_std", "pr_auc_min", "pr_auc_max", "diff_vs_incumbent", "replaces", "p50_ms", "p99_ms"]].round(4).to_string())
