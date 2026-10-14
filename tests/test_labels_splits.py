import numpy as np
import pytest

from src.data.dataset import make_labels
from src.preprocessing.splits import compute_bounds, target_indices


def test_labels_only_look_forward():
    f = np.zeros(20, dtype=int)
    f[10] = 1
    y = make_labels(f, 3)
    assert list(np.where(y == 1)[0]) == [7, 8, 9]     # failure in (t, t+3]
    assert y[10] == 0                                   # the failure step itself is not "in the future"
    assert np.isnan(y[-3:]).all() and not np.isnan(y[:-3]).any()


def test_split_bounds_and_embargo(cfg):
    n = 10_000
    b = compute_bounds(n, cfg, lookback=114)
    r = b.ranges()
    assert r["train"][1] + b.emb_out <= b.train_end
    assert r["val"][0] - b.emb_in >= b.train_end
    assert r["val"][1] + b.emb_out <= b.val_end
    assert r["test"][0] - b.emb_in >= b.val_end
    off = np.zeros(n, bool)
    s = {k: set(target_indices(b, k, off)) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"]) and not (s["val"] & s["test"])
    assert max(s["train"]) < min(s["val"]) and max(s["val"]) < min(s["test"])


def test_offline_rows_excluded(cfg):
    b = compute_bounds(5000, cfg, lookback=114)
    off = np.zeros(5000, bool)
    off[4000:4010] = True
    assert not set(range(4000, 4010)) & set(target_indices(b, "test", off))


def test_bad_fractions(cfg):
    bad = dict(cfg) | {"split": {"train_frac": 0.7, "val_frac": 0.2, "test_frac": 0.2}}
    with pytest.raises(ValueError):
        compute_bounds(100, bad, 10)
