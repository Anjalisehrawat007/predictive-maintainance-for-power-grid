"""Batch prediction on a sensor CSV.

python predict.py --input data/raw/ETTm1.csv --asset ETTm1 --output reports/predictions_ETTm1.csv

The CSV needs a timestamp column (``date`` or ``timestamp``) and the sensor
columns HUFL, HULL, MUFL, MULL, LUFL, LULL, OT at 15-min resolution.  The
asset must be one the preprocessor was fitted on (per-asset scaling).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import load_artifacts
from src.streaming.inference import predict_frame
from src.utils.config import load_config
from src.utils.logging import get_logger

warnings.filterwarnings("ignore")
log = get_logger("predict")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = load_artifacts(cfg)
    df = pd.read_csv(args.input)
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    missing = set(art.fe.sensor_cols) - set(df.columns)
    if missing:
        raise SystemExit(f"input is missing sensor columns: {sorted(missing)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    probs = predict_frame(art, df, args.asset)
    policy = AlertPolicy.from_config(cfg, art.meta)
    status = [policy.update(p) if not np.isnan(p) else "WARMUP" for p in probs]
    out = df[["timestamp"]].assign(failure_probability=probs, status=status)
    path = args.output or f"reports/predictions_{args.asset}.csv"
    out.to_csv(path, index=False)
    log.info("Wrote %d predictions to %s | status counts: %s", len(out), path, out.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
