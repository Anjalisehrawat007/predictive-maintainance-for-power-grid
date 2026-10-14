"""Assemble model-ready arrays for any (feature set, horizon) combination.

All assets are concatenated into one array; every target index is chosen so
its input window lies inside a single asset *and* a single split period, so
windows never straddle assets or splits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.dataset import load_processed, make_labels
from src.features.engineering import FeatureEngineer
from src.preprocessing.scaling import AssetScaler
from src.preprocessing.splits import SplitBounds, compute_bounds, save_bounds, target_indices
from src.utils.logging import get_logger

log = get_logger("data.pipeline")


@dataclass
class Prepared:
    X: np.ndarray                      # (N, F) scaled float32
    y: np.ndarray                      # (N,)  float32, NaN = unscored
    feature_names: list[str]
    idx: dict[str, np.ndarray]         # split -> global target rows
    timestamps: np.ndarray             # (N,) datetime64
    asset: np.ndarray                  # (N,) asset name per row
    offline: np.ndarray                # (N,) bool
    failure: np.ndarray                # (N,) int
    offsets: dict[str, int]
    bounds: dict[str, SplitBounds]
    horizon: int
    fe: FeatureEngineer
    scaler: AssetScaler
    raw_features: pd.DataFrame | None = None   # unscaled features (for EDA / explanations)
    extras: dict = field(default_factory=dict)

    def sudden_positive_mask(self, events: pd.DataFrame) -> np.ndarray:
        """Rows whose positive label comes from a *sudden* (no-precursor) failure.

        Used only for the diagnostic metric ``pr_auc_detectable``.
        """
        m = np.zeros(len(self.y), dtype=bool)
        for r in events[(events.kind == "failure") & (events.type == "sudden")].itertuples():
            tf = self.offsets[r.asset] + int(r.failure_idx)
            m[max(tf - self.horizon, 0): tf] = True
        return m

    def failure_rows(self, split: str) -> np.ndarray:
        """Global indices of failures whose time falls inside the split's scored period."""
        out = []
        for a, off in self.offsets.items():
            lo, hi = self.bounds[a].ranges()[split]
            f = np.where(self.failure[off + lo: off + hi + self.horizon] == 1)[0] + off + lo
            out.append(f)
        return np.concatenate(out)


def _select_features(F: np.ndarray, y: np.ndarray, names: list[str], k: int, seed: int) -> list[str]:
    """Rank features by mutual information with the label on TRAINING rows only."""
    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg = rng.choice(neg, size=min(len(neg), 20 * max(len(pos), 1)), replace=False)
    sel = np.concatenate([pos, neg])
    mi = mutual_info_classif(F[sel], y[sel].astype(int), random_state=seed)
    order = np.argsort(mi)[::-1][:k]
    return [names[i] for i in sorted(order)]


def prepare(cfg: dict, feature_set: str | None = None, horizon: int | None = None,
            keep_raw: bool = False, dfs: dict | None = None) -> Prepared:
    """Build model-ready arrays.  ``dfs`` (optional) = an in-memory simulated benchmark with the same
    schema as data/processed (used for robustness experiments); default = the reference benchmark."""
    feature_set = feature_set or cfg["features"]["set"]
    horizon = int(horizon or cfg["labels"]["horizon_steps"])
    external = dfs is not None
    if not external:
        dfs, _ = load_processed(cfg)
    fe = FeatureEngineer.from_config(cfg, adaptive=feature_set == "adaptive", robust=feature_set == "robust")
    sets = fe.feature_sets()
    base_set = "raw_engineered" if feature_set == "selected" else feature_set
    cols = sets[base_set]

    scaler = AssetScaler()
    Xs, ys, ts, assets, offl, fails, raws = [], [], [], [], [], [], []
    idx = {"train": [], "val": [], "test": []}
    offsets, bounds = {}, {}
    off = 0
    for a, df in dfs.items():
        b = compute_bounds(len(df), cfg, fe.lookback)
        train_mask = np.zeros(len(df), bool)
        train_mask[: b.train_end] = True
        fe.fit(df, a, train_mask)
        feats = fe.transform(df, a)[cols]
        F = feats.to_numpy(dtype=np.float64)
        offline = df["offline"].to_numpy(bool)
        scaler.fit(F[: b.train_end][~offline[: b.train_end]], a)
        Xs.append(scaler.transform(F, a))
        if keep_raw:
            raws.append(feats.assign(asset=a, timestamp=df["timestamp"].values))
        y = make_labels(df["failure"].to_numpy(), horizon)
        y[offline] = np.nan
        ys.append(y)
        ts.append(df["timestamp"].to_numpy())
        assets.append(np.full(len(df), a, dtype=object))
        offl.append(offline)
        fails.append(df["failure"].to_numpy())
        for s in idx:
            ti = target_indices(b, s, offline)
            ti = ti[~np.isnan(y[ti])]
            idx[s].append(ti + off)
        offsets[a], bounds[a] = off, b
        off += len(df)

    p = Prepared(
        X=np.concatenate(Xs), y=np.concatenate(ys), feature_names=list(cols),
        idx={s: np.concatenate(v) for s, v in idx.items()},
        timestamps=np.concatenate(ts), asset=np.concatenate(assets),
        offline=np.concatenate(offl), failure=np.concatenate(fails),
        offsets=offsets, bounds=bounds, horizon=horizon, fe=fe, scaler=scaler,
        raw_features=pd.concat(raws, ignore_index=True) if keep_raw else None,
        extras={"scaler_columns": list(cols), "feature_set": feature_set},
    )
    if not external:
        save_bounds(cfg, bounds, {a: dfs[a]["timestamp"].astype(str).to_numpy() for a in dfs})
    if feature_set == "selected":
        tr = p.idx["train"]
        keep = _select_features(p.X[tr], p.y[tr], p.feature_names, cfg["features"]["n_selected"], cfg["seed"])
        ki = [p.feature_names.index(c) for c in keep]
        p.X, p.feature_names = p.X[:, ki], keep
    log.info("Prepared %s | H=%d | F=%d | train/val/test targets = %d/%d/%d | pos rate train=%.4f val=%.4f test=%.4f",
             feature_set, horizon, len(p.feature_names), *(len(p.idx[s]) for s in ("train", "val", "test")),
             *(np.nanmean(p.y[p.idx[s]]) for s in ("train", "val", "test")))
    return p
