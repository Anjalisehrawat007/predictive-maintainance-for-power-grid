"""Build the processed dataset: real measurements + simulation layer + labels."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.download import download_raw, load_raw
from src.data.simulation import FaultSimulator
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("data.dataset")


def build_processed(cfg: dict, force: bool = False) -> None:
    """Download raw data, inject simulated events and save one parquet per asset."""
    proc = resolve(cfg, "processed_dir")
    if not force and all((proc / f"{a}.parquet").exists() for a in cfg["dataset"]["assets"]) \
            and (proc / "events.csv").exists():
        log.info("Processed data already present (use --force to rebuild)")
        return
    download_raw(cfg)
    events = []
    card = {"dataset": cfg["dataset"]["name"], "assets": {}}
    for i, asset in enumerate(cfg["dataset"]["assets"]):
        raw = load_raw(cfg, asset)
        sim = FaultSimulator(cfg["simulation"], cfg["dataset"]["sensor_cols"],
                             seed=cfg["simulation"]["seed"] + i)
        df, ev = sim.simulate(raw, asset)
        df.to_parquet(proc / f"{asset}.parquet", index=False)
        events.append(ev)
        fe = ev[ev.kind == "failure"]
        card["assets"][asset] = {
            "rows": int(len(df)), "start": str(df.timestamp.min()), "end": str(df.timestamp.max()),
            "n_failures": int(len(fe)), "failures_by_type": fe.type.value_counts().to_dict(),
            "n_nuisance": int((ev.kind == "nuisance").sum()),
            "offline_rows": int(df.offline.sum()),
        }
        log.info("%s: %d rows, %d failures, %d nuisance events", asset, len(df), len(fe),
                 (ev.kind == "nuisance").sum())
    pd.concat(events, ignore_index=True).to_csv(proc / "events.csv", index=False)
    (proc / "dataset_card.json").write_text(json.dumps(card, indent=2, default=str))


def load_processed(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    proc = resolve(cfg, "processed_dir")
    if not (proc / "events.csv").exists():
        build_processed(cfg)
    dfs = {a: pd.read_parquet(proc / f"{a}.parquet") for a in cfg["dataset"]["assets"]}
    events = pd.read_csv(proc / "events.csv", parse_dates=["onset_time", "failure_time", "end_time"])
    return dfs, events


def make_labels(failure: np.ndarray, horizon: int) -> np.ndarray:
    """y[t] = 1 iff a failure occurs in (t, t + horizon].

    Rows whose horizon extends past the end of the series get NaN (unknown).
    Only *future* failure flags are used for the *target*; they never enter inputs.
    """
    f = np.asarray(failure, dtype=np.int64)
    n = len(f)
    cs = np.concatenate([[0], np.cumsum(f)])          # cs[i] = sum f[:i]
    t = np.arange(n)
    hi = np.minimum(t + horizon, n - 1)
    y = (cs[hi + 1] - cs[t + 1] > 0).astype(np.float32)
    y[t + horizon >= n] = np.nan
    return y
