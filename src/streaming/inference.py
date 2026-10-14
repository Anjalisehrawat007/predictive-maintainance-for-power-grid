"""Batch inference shared by the CLI (predict.py), the API (/api/predict) and the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def predict_frame(art, df: pd.DataFrame, asset: str, batch: int = 2048, model=None) -> np.ndarray:
    """Raw model scores for every row that has a complete, exact input window (NaN before)."""
    if asset not in art.scaler.mean_:
        raise ValueError(f"asset {asset!r} unknown to the preprocessor; known: {list(art.scaler.mean_)}")
    model = model or art.model
    Z = art.features(df, asset).astype(np.float32)
    L = art.seq_len
    first = art.fe.lookback + L - 2           # first row whose window only contains exact features
    rows = np.arange(first, len(df))
    probs = np.full(len(df), np.nan)
    off = np.arange(-L + 1, 1)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            probs[r] = torch.sigmoid(model(torch.from_numpy(Z[r[:, None] + off]))).numpy()
    return probs


def min_rows(art) -> int:
    """Rows of history needed before the first prediction."""
    return art.fe.lookback + art.seq_len - 1
