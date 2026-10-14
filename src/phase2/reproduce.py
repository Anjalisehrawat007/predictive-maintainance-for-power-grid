"""Phase-2 / step 1: reproduce the reported Phase-1 results from saved artefacts only.

Nothing is read from final_metrics.json for the *recomputation*; the stored report is
only used for the comparison.  Discrepancies are reported, never silently fixed.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_processed
from src.data.pipeline import prepare
from src.evaluation.efficiency import model_size_kb, torch_latency
from src.evaluation.events import evaluate_events
from src.evaluation.metrics import classification_metrics
from src.phase2.common import P2_REPORTS, REPORTS, write_md
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import load_artifacts
from src.streaming.engine import replay
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import PROJECT_ROOT, load_config

# numbers as claimed in README / Phase-1 summary (test split)
CLAIMED = {"precision": 0.318, "recall": 0.488, "f1": 0.385, "roc_auc": 0.845, "pr_auc": 0.326,
           "false_alarms_per_day": 0.12, "lead_time_median_min": 120.0, "params": 31425, "size_kb": 127,
           "torch_latency_ms": 0.42, "onnx_latency_ms": 0.07, "stream_ms": 8.4}


def onnx_latency(path, L, F, n=300) -> float | None:
    try:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, L, F).astype(np.float32)
        for _ in range(30):
            s.run(None, {"x": x})
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); s.run(None, {"x": x}); t.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(t))
    except Exception as e:  # pragma: no cover
        print("onnx failed", e)
        return None


def run() -> dict:
    cfg = load_config()
    art = load_artifacts(cfg)
    meta = art.meta
    p = prepare(cfg, meta["feature_set"], meta["horizon"])
    _, events = load_processed(cfg)
    b = WindowBatcher(p.X, p.y, art.seq_len, torch.device("cpu"))
    out = {"model": meta["active_model"], "threshold": meta["active_threshold"]}
    for split in ("val", "test"):
        pr = predict_proba(art.model, b, p.idx[split])
        m = classification_metrics(p.y[p.idx[split]], pr, meta["active_threshold"])
        ev = evaluate_events(p, split, pr, meta["active_threshold"], events, cfg)["summary"]
        out[split] = m | ev
    out["params"] = int(sum(x.numel() for x in art.model.parameters()))
    out["size_kb_state_dict"] = model_size_kb(art.model)
    out["size_kb_file"] = (PROJECT_ROOT / "models" / f"{meta['active_model']}.pt").stat().st_size / 1024
    out["torch_latency_ms"] = torch_latency(art.model, art.seq_len, len(art.feature_names), 1, threads=1,
                                            n_iter=400)["latency_ms_median"]
    out["onnx_latency_ms"] = onnx_latency(PROJECT_ROOT / "models" / f"{meta['active_model']}.onnx",
                                          art.seq_len, len(art.feature_names))
    dfs, _ = load_processed(cfg)
    df = dfs["ETTm1"]
    b0 = p.bounds["ETTm1"].period("test")[0]
    r = replay(art, df, "ETTm1", AlertPolicy.from_config(cfg, meta), meta["horizon"], b0, b0 + 800)
    out["stream_ms_median"] = float(r.latency_ms.iloc[200:].median())

    stored = json.loads((REPORTS / "final_metrics.json").read_text())
    rows = []
    for k, claimed in CLAIMED.items():
        if k in ("params", "size_kb", "torch_latency_ms", "onnx_latency_ms", "stream_ms"):
            got = {"params": out["params"], "size_kb": out["size_kb_state_dict"],
                   "torch_latency_ms": out["torch_latency_ms"], "onnx_latency_ms": out["onnx_latency_ms"],
                   "stream_ms": out["stream_ms_median"]}[k]
            st = None
        else:
            got, st = out["test"][k], stored["test"][k]
        tol = 0.005 if k not in ("params", "size_kb", "lead_time_median_min") else 0.5
        timing = k.endswith("_ms")
        status = ("timing (re-measured)" if timing else
                  "MATCH" if abs(got - claimed) <= max(tol, abs(claimed) * 0.02) else "DISCREPANCY")
        rows.append({"metric": k, "claimed": claimed, "stored_report": st, "reproduced": got, "status": status})
    tab = pd.DataFrame(rows)
    out["comparison"] = rows
    (P2_REPORTS / "reproduction.json").write_text(json.dumps(out, indent=2, default=float))
    lines = ["# Phase 2 · Step 1 — Reproduction of Phase-1 results\n",
             f"Recomputed from `models/{meta['active_model']}.pt` + `models/preprocessor.pkl` + processed data; "
             "the stored `reports/final_metrics.json` is only used for comparison.\n",
             "| metric | claimed (README) | stored report | reproduced now | status |", "|---|---|---|---|---|"]
    for r_ in rows:
        def f(v):
            return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {r_['metric']} | {f(r_['claimed'])} | {f(r_['stored_report'])} | {f(r_['reproduced'])} | {r_['status']} |")
    lines += ["", "Timing values are hardware/load dependent and were re-measured (median of ≥300 calls, 1 thread); "
              "they are reported, not compared for equality.",
              f"\nValidation (reproduced): PR-AUC {out['val']['pr_auc']:.4f}, event recall (detectable) "
              f"{out['val']['event_recall_detectable']:.3f}, FA/day {out['val']['false_alarms_per_day']:.3f}."]
    write_md(P2_REPORTS / "reproduction.md", "\n".join(lines))
    print(tab.to_string())
    return out


if __name__ == "__main__":
    run()
