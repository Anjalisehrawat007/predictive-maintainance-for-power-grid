"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config (``path`` > $SGPM_CONFIG > configs/config.yaml), deep-merging ``overrides``."""
    with open(path or os.environ.get("SGPM_CONFIG") or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    sim = cfg.get("simulation_config")
    if sim and "simulation" not in cfg:            # failure simulation lives in its own documented file
        with open(PROJECT_ROOT / sim) as f:
            cfg.update(yaml.safe_load(f))
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for ``cfg['paths'][key]``; the directory is created if missing."""
    p = PROJECT_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
