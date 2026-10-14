import numpy as np
import pytest

from src.features.engineering import FeatureEngineer
from src.preprocessing.leakage import check_forbidden_columns
from src.preprocessing.scaling import AssetScaler


def _fe(cfg, df, adaptive=False):
    fe = FeatureEngineer.from_config(cfg, adaptive=adaptive, robust=True)
    fe.fit(df, "A", np.arange(len(df)) < len(df) // 2)
    return fe


@pytest.mark.parametrize("adaptive", [False, True])
def test_features_are_causal(cfg, synthetic_df, adaptive):
    fe = _fe(cfg, synthetic_df, adaptive)
    base = fe.transform(synthetic_df, "A")
    t = 1500
    pert = synthetic_df.copy()
    pert.loc[t + 1:, ["OT", "HUFL", "HULL"]] *= 5
    new = fe.transform(pert, "A")
    np.testing.assert_array_equal(np.nan_to_num(base.iloc[: t + 1].to_numpy(), nan=-9),
                                  np.nan_to_num(new.iloc[: t + 1].to_numpy(), nan=-9))


@pytest.mark.parametrize("adaptive", [False, True])
def test_lookback_is_sufficient(cfg, synthetic_df, adaptive):
    """Features from only the last `lookback` rows equal the full-history features."""
    fe = _fe(cfg, synthetic_df, adaptive)
    full = fe.transform(synthetic_df, "A")
    for t in (900, 1800, len(synthetic_df) - 1):
        tail = synthetic_df.iloc[t - fe.lookback + 1: t + 1]
        np.testing.assert_allclose(fe.transform(tail, "A").iloc[-1].to_numpy(), full.iloc[t].to_numpy(),
                                   rtol=1e-9, atol=1e-9)


def test_fit_uses_training_rows_only(cfg, synthetic_df):
    fe1 = _fe(cfg, synthetic_df)
    mod = synthetic_df.copy()
    mod.loc[len(mod) // 2:, "OT"] += 50          # change only the non-training part
    fe2 = _fe(cfg, mod)
    np.testing.assert_allclose(fe1.fits["A"].thermal_coef, fe2.fits["A"].thermal_coef)
    assert fe1.fits["A"].diff_sigma == fe2.fits["A"].diff_sigma


def test_feature_sets_have_no_forbidden_columns(cfg):
    sets = FeatureEngineer.from_config(cfg).feature_sets()
    for cols in sets.values():
        assert check_forbidden_columns(cols)[0]
    assert not check_forbidden_columns(["OT", "sim_delta_OT"])[0]
    assert not check_forbidden_columns(["failure"])[0]
    assert not any(c.startswith("doy") for c in sets["stationary"])


def test_scaler_train_stats():
    X = np.random.default_rng(0).normal(3, 2, (1000, 4))
    s = AssetScaler().fit(X[:700], "A")
    np.testing.assert_allclose(s.mean_["A"], X[:700].mean(0))
    Z = s.transform(X, "A")
    assert Z.dtype == np.float32 and np.abs(Z).max() <= 10
