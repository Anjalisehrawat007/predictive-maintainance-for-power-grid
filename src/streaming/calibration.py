"""Serving-time calibrator: applies a fitted calibration map (no scikit-learn / SciPy / PyTorch needed).

Fitting lives in src/phase2/calibration.py (research). For isotonic calibration, prediction is the
linear interpolation between the fitted thresholds, clipped at both ends — exactly what scikit-learn's
``IsotonicRegression(out_of_bounds="clip").predict`` computes (tested)."""
from __future__ import annotations

import numpy as np

EPS = 1e-6
# Model-confidence bins (lower bounds). Single definition used by evaluation, the API and the UI.
CONFIDENCE_BINS = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.0}


def confidence_bin(conf) -> np.ndarray:
    """Label each confidence value HIGH / MEDIUM / LOW using CONFIDENCE_BINS."""
    c = np.asarray(conf, dtype=float)
    return np.where(c >= CONFIDENCE_BINS["HIGH"], "HIGH", np.where(c >= CONFIDENCE_BINS["MEDIUM"], "MEDIUM", "LOW"))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class ServingCalibrator:
    def __init__(self, method: str = "identity", T: float = 1.0, b: float = 0.0, iso_x=None, iso_y=None):
        self.method, self.T, self.b = method, float(T), float(b)
        self.iso_x = None if iso_x is None else np.asarray(iso_x, dtype=float)
        self.iso_y = None if iso_y is None else np.asarray(iso_y, dtype=float)

    @classmethod
    def from_dict(cls, d: dict) -> ServingCalibrator:
        return cls(d["method"], d.get("T", 1.0), d.get("b", 0.0), d.get("iso_x"), d.get("iso_y"))

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if self.method in ("temperature", "platt"):
            return 1 / (1 + np.exp(-(_logit(p) / self.T + self.b)))
        if self.method == "isotonic":
            return np.interp(p, self.iso_x, self.iso_y)
        return p
