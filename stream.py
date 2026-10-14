"""Streaming simulation: replay historical data one observation at a time.

python stream.py                         # asset/split from config (default ETTm1 test)
python stream.py --asset ETTm2 --steps 2000 --speed 0.01 --verbose
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

from src.data.dataset import load_processed
from src.preprocessing.splits import compute_bounds
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import load_artifacts
from src.streaming.engine import replay
from src.utils.config import load_config, resolve
from src.utils.logging import get_logger

warnings.filterwarnings("ignore")
log = get_logger("stream")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--split", default=None, choices=["val", "test"])
    ap.add_argument("--steps", type=int, default=None, help="limit number of replayed observations")
    ap.add_argument("--speed", type=float, default=0.0, help="seconds to sleep between observations")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asset = args.asset or cfg["streaming"]["asset"]
    split = args.split or cfg["streaming"]["split"]
    art = load_artifacts(cfg)
    policy = AlertPolicy.from_config(cfg, art.meta)
    dfs, _ = load_processed(cfg)
    df = dfs[asset]
    b = compute_bounds(len(df), cfg, art.fe.lookback)
    lo, hi = b.period(split)
    # warm-up starts inside the split's embargo, so no earlier-period data is needed
    stop = hi if args.steps is None else min(hi, lo + args.steps)
    log.info("Replaying %s/%s rows %d..%d with %s (warn=%.3f, crit=%.3f, k=%d)", asset, split, lo, stop,
             art.meta["active_model"], policy.warning_threshold, policy.critical_threshold, policy.persistence_k)
    out = replay(art, df, asset, policy, art.meta["horizon"], lo, stop, cfg["threshold"]["merge_gap_steps"],
                 args.speed, cfg["streaming"]["latency_log_every"] if args.verbose else 0)
    sdir = resolve(cfg, "stream_dir")
    path = sdir / f"stream_{asset}_{split}.csv"
    out.to_csv(path, index=False)
    scored = out[out.status.isin(["NORMAL", "WARNING", "CRITICAL"])]
    fails = out[out.failure_event == 1]
    summary = {
        "asset": asset, "split": split, "model": art.meta["active_model"], "observations": int(len(out)),
        "status_counts": out.status.value_counts().to_dict(),
        "failures": int(len(fails)), "failures_with_active_warning": int(fails.lead_time_min.notna().sum()),
        "online_lead_time_median_min": float(fails.lead_time_min.median()) if fails.lead_time_min.notna().any() else None,
        "latency_ms_median": float(out.latency_ms.median()), "latency_ms_p95": float(np.percentile(out.latency_ms, 95)),
        "latency_ms_max": float(out.latency_ms.max()),
        "warning_threshold": policy.warning_threshold, "critical_threshold": policy.critical_threshold,
        "alarm_time_fraction": float(scored.alarm.mean()) if len(scored) else None,
    }
    (sdir / f"stream_{asset}_{split}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote %s\n%s", path, json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
