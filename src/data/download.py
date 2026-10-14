"""Download the raw ETT transformer data (real measurements)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("data.download")


def download_raw(cfg: dict, force: bool = False) -> dict[str, Path]:
    """Download each asset CSV into ``data/raw`` (skipped if already present)."""
    raw_dir = resolve(cfg, "raw_dir")
    out = {}
    for asset in cfg["dataset"]["assets"]:
        path = raw_dir / f"{asset}.csv"
        if force or not path.exists():
            url = f"{cfg['dataset']['base_url']}/{asset}.csv"
            log.info("Downloading %s", url)
            urllib.request.urlretrieve(url, path)
        out[asset] = path
    return out


def load_raw(cfg: dict, asset: str) -> pd.DataFrame:
    """Load one raw asset, parse timestamps and validate the schema."""
    path = resolve(cfg, "raw_dir") / f"{asset}.csv"
    if not path.exists():
        download_raw(cfg)
    ts = cfg["dataset"]["timestamp_col"]
    df = pd.read_csv(path, parse_dates=[ts]).rename(columns={ts: "timestamp"})
    missing = set(cfg["dataset"]["sensor_cols"]) - set(df.columns)
    if missing:
        raise ValueError(f"{asset}: missing sensor columns {missing}")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    expected = pd.Timedelta(minutes=cfg["dataset"]["freq_minutes"])
    gaps = (df["timestamp"].diff().dropna() != expected).sum()
    if gaps:
        log.warning("%s: %d irregular timestamp gaps", asset, gaps)
    return df
