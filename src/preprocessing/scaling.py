"""Per-asset standardisation fitted on training rows only."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AssetScaler:
    clip: float = 10.0
    mean_: dict[str, np.ndarray] = field(default_factory=dict)
    std_: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, X: np.ndarray, asset: str) -> AssetScaler:
        self.mean_[asset] = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.std_[asset] = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, X: np.ndarray, asset: str) -> np.ndarray:
        Z = (X - self.mean_[asset]) / self.std_[asset]
        Z = np.clip(Z, -self.clip, self.clip)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)
