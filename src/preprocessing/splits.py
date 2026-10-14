"""Chronological per-asset splitting with an embargo.

Layout per asset (row indices)::

    |warm-up|------ train targets ------|emb_out|emb_in|--- val targets ---|emb_out|emb_in|--- test ---|emb_out|
    0                                   a                                  b                         n

* ``emb_out`` (= max horizon) — the last training target's label window
  (t, t+H] never reaches into the validation period.
* ``emb_in``  (= max sequence length) — every input window of a validation /
  test target lies entirely inside its own period.

All experiments share the *same* target rows (computed with the maximum
sequence length and horizon), so every model is scored on identical samples.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from src.utils.config import resolve

MAX_SEQ_LEN = 96
MAX_HORIZON = 16


@dataclass
class SplitBounds:
    n: int
    train_end: int
    val_end: int
    warmup: int
    emb_in: int = MAX_SEQ_LEN
    emb_out: int = MAX_HORIZON

    def ranges(self) -> dict[str, tuple[int, int]]:
        return {
            "train": (self.warmup, self.train_end - self.emb_out),
            "val": (self.train_end + self.emb_in, self.val_end - self.emb_out),
            "test": (self.val_end + self.emb_in, self.n - self.emb_out),
        }

    def period(self, split: str) -> tuple[int, int]:
        """Full row period (inputs + targets) owned by a split."""
        return {"train": (0, self.train_end), "val": (self.train_end, self.val_end),
                "test": (self.val_end, self.n)}[split]


def compute_bounds(n: int, cfg: dict, lookback: int) -> SplitBounds:
    s = cfg["split"]
    if abs(s["train_frac"] + s["val_frac"] + s["test_frac"] - 1) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    a = int(n * s["train_frac"])
    b = int(n * (s["train_frac"] + s["val_frac"]))
    return SplitBounds(n=n, train_end=a, val_end=b, warmup=lookback + MAX_SEQ_LEN)


def target_indices(bounds: SplitBounds, split: str, offline: np.ndarray) -> np.ndarray:
    lo, hi = bounds.ranges()[split]
    idx = np.arange(lo, hi)
    return idx[~np.asarray(offline, dtype=bool)[idx]]


def save_bounds(cfg: dict, bounds: dict[str, SplitBounds], timestamps: dict) -> None:
    out = {}
    for a, b in bounds.items():
        ts = timestamps[a]
        out[a] = asdict(b) | {k: [str(ts[lo]), str(ts[hi - 1])] for k, (lo, hi) in b.ranges().items()}
    (resolve(cfg, "splits_dir") / "split_bounds.json").write_text(json.dumps(out, indent=2))
