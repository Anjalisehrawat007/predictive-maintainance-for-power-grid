import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402

SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synthetic_df():
    """30 days of smooth synthetic 15-min sensor data (no download needed)."""
    n = 96 * 30
    rng = np.random.default_rng(0)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 96)
    d = {"timestamp": pd.date_range("2020-01-01", periods=n, freq="15min")}
    for i, c in enumerate(SENSORS):
        d[c] = 5 + 2 * base * (1 + 0.1 * i) + rng.normal(0, 0.3, n)
    d["OT"] = 20 + 3 * base + 0.002 * t + rng.normal(0, 0.2, n)
    return pd.DataFrame(d)
