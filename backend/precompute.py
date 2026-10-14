"""Replay the whole held-out test period through the intelligence engine once and store
the outputs ("engine log").  The UI's global time-scrubber reads this log; live/replay
modes run the same engine on demand.  Also verifies that alarm states equal the Phase-1
streaming log (same model, same thresholds → must be identical)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from backend.intelligence import SENSORS, load_deployment
from backend.sessions import Session
from src.data.dataset import load_processed
from src.preprocessing.splits import compute_bounds
from src.utils.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "reports" / "phase2" / "engine_log"


def flatten(rec: dict) -> dict:
    h = rec.get("health") or {}
    sig = rec.get("signals") or []
    d = {"index": rec["index"], "timestamp": rec["timestamp"], **{c: rec["sensors"][c] for c in SENSORS},
         "offline": rec["offline"], "status": rec["status"], "raw_score": rec["raw_score"], "risk": rec["risk"],
         "confidence": rec["confidence"], "trend": rec.get("trend"), "trend_delta_1h": rec.get("trend_delta_1h"),
         "warning_since": rec.get("warning_since"), "health": h.get("score"),
         **{f"h_{k}": v for k, v in (h.get("components") or {}).items()},
         "signal1": sig[0]["signal"] if sig else None, "signal1_share": sig[0]["share"] if sig else None,
         "signal2": sig[1]["signal"] if len(sig) > 1 else None, "signal2_share": sig[1]["share"] if len(sig) > 1 else None,
         "attribution": json.dumps({s["signal"]: round(s["share"], 4) for s in rec.get("attribution") or []}),
         "failure_event": rec["ground_truth"]["failure_event"], "sim_phase": rec["ground_truth"]["sim_phase"],
         "sim_event_type": rec["ground_truth"]["sim_event_type"], "lead_time_min": rec.get("lead_time_min"),
         "t_total_ms": rec["timings_ms"].get("total")}
    return d


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    dep = load_deployment(cfg)
    dfs, _ = load_processed(cfg)
    a0 = cfg["dataset"]["assets"][0]
    b = compute_bounds(len(dfs[a0]), cfg, dep.art.fe.lookback)
    lo, hi = b.period("test")
    t = time.time()
    s = Session(dep, dfs, list(dfs), lo, hi, mode="log", persistence_k=cfg["threshold"]["persistence_k"],
                merge_gap=cfg["threshold"]["merge_gap_steps"])
    rows = {a: [] for a in dfs}
    events = []
    timings = []
    while s.cursor < s.stop:
        out = s.step(256)
        for a, recs in out["records"].items():
            rows[a].extend(flatten(r) for r in recs)
            timings.extend(r["timings_ms"] for r in recs if r["risk"] is not None)
        events.extend(out["events"])
    for a, r in rows.items():
        pd.DataFrame(r).to_parquet(OUT / f"{a}.parquet", index=False)
    pd.DataFrame(events).to_parquet(OUT / "events.parquet", index=False)
    alerts = pd.DataFrame(list(s.alerts.values()))
    alerts["source"] = "engine log (held-out test period)"
    alerts["signals"] = alerts["signals"].apply(json.dumps)
    alerts.to_parquet(OUT / "alerts.parquet", index=False)
    tim = pd.DataFrame(timings)
    stats = {"median_ms": tim.median().round(4).to_dict(), "p95_ms": tim.quantile(.95).round(4).to_dict(),
             "n_steps": len(tim), "onnx_inference": dep.onnx is not None, "model": dep.name,
             "wall_s": round(time.time() - t, 1)}
    # consistency with the Phase-1 stream logs (same model & thresholds)
    cons = {}
    for a in dfs:
        p1 = PROJECT_ROOT / "reports" / "stream" / f"stream_{a}_test.csv"
        if p1.exists():
            old = pd.read_csv(p1)
            new = pd.DataFrame(rows[a])
            n = min(len(old), len(new))
            same_thr = abs(dep.warn_raw - dep.art.meta["active_threshold"]) < 1e-12
            cons[a] = {"rows_compared": n,
                       "max_abs_raw_score_diff": float(np.nanmax(np.abs(old.failure_probability.values[160:n] - new.raw_score.astype(float).values[160:n]))),
                       "status_identical_after_warmup": float((old.status.values[160:n] == new.status.values[160:n]).mean()) if same_thr else None,
                       "note": "raw scores must match the Phase-1 stream exactly; statuses are only comparable when the "
                               "thresholds are the same (the final operating point uses re-derived thresholds)"}
    stats["consistency_with_phase1_stream"] = cons
    (OUT / "timings.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
