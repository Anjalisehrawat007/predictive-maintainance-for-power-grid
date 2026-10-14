"""Configurable alert policy: NORMAL / WARNING / CRITICAL."""
from __future__ import annotations

from dataclasses import dataclass

NORMAL, WARNING, CRITICAL, OFFLINE, WARMUP = "NORMAL", "WARNING", "CRITICAL", "OFFLINE", "WARMUP"


@dataclass
class AlertPolicy:
    warning_threshold: float
    critical_threshold: float
    persistence_k: int = 2

    def __post_init__(self):
        if not 0 <= self.warning_threshold <= self.critical_threshold <= 1:
            raise ValueError("need 0 <= warning_threshold <= critical_threshold <= 1")
        self._run = 0

    @classmethod
    def from_config(cls, cfg: dict, meta: dict, manifest: dict | None = None) -> AlertPolicy:
        """Resolve the operating thresholds. Precedence (one place, used by the API and every CLI):
        1. explicit values in configs/config.yaml `alerts`;
        2. the deployment manifest (models/phase2/deployment.json → thresholds), derived on fresh validation;
        3. the Phase-1 values stored with the model (model_metadata.json)."""
        if manifest is None:
            from src.streaming.artifacts import deployment_manifest
            manifest = deployment_manifest(cfg)
        th = manifest.get("thresholds") or {}
        warn = th.get("warning_raw", meta.get("active_threshold", meta["warning_threshold"]))
        crit = th.get("critical_raw", meta["critical_threshold"])
        a = cfg["alerts"]
        if a["warning_threshold"] is not None:
            warn = a["warning_threshold"]
        if a["critical_threshold"] is not None:
            crit = a["critical_threshold"]
        return cls(float(warn), float(max(crit, warn)), int(cfg["threshold"]["persistence_k"]))

    def reset(self) -> None:
        self._run = 0

    def update(self, prob: float) -> str:
        """Stateful: an alarm needs ``persistence_k`` consecutive steps above the warning level."""
        self._run = self._run + 1 if prob >= self.warning_threshold else 0
        if self._run < self.persistence_k:
            return NORMAL
        return CRITICAL if prob >= self.critical_threshold else WARNING
