"""Streaming engine must reproduce offline (batch) features and probabilities exactly."""
import numpy as np
import torch

from src.features.engineering import FeatureEngineer
from src.models.recurrent import build_model
from src.preprocessing.scaling import AssetScaler
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import Artifacts
from src.streaming.engine import StreamingPredictor


def _artifacts(cfg, df, L=12):
    fe = FeatureEngineer.from_config(cfg)
    fe.fit(df, "A", np.arange(len(df)) < 1500)
    cols = fe.feature_sets()["stationary"]
    F = fe.transform(df, "A")[cols].to_numpy(float)
    sc = AssetScaler().fit(F[:1500], "A")
    torch.manual_seed(0)
    model = build_model("gru_attention", len(cols), 8).eval()
    meta = {"seq_len": L, "horizon": 8, "warning_threshold": 0.5, "critical_threshold": 0.8}
    return Artifacts(model, meta, fe, sc, cols, cols)


def test_streaming_equals_batch(cfg, synthetic_df):
    from predict import predict_frame

    art = _artifacts(cfg, synthetic_df)
    df = synthetic_df.iloc[:400].reset_index(drop=True)
    batch = predict_frame(art, df, "A")
    sp = StreamingPredictor(art, "A", AlertPolicy(0.5, 0.8))
    stream = np.array([sp.update(r.timestamp, r._asdict())["failure_probability"] for r in df.itertuples()])
    ok = ~np.isnan(batch)
    assert ok.sum() > 100
    np.testing.assert_array_equal(np.isnan(stream), np.isnan(batch))
    np.testing.assert_allclose(stream[ok], batch[ok], rtol=1e-5, atol=1e-6)


def test_transform_last_equals_full_transform(cfg, synthetic_df):
    """The engine's fast newest-row feature path must equal the batch pandas features."""
    fe = FeatureEngineer.from_config(cfg)
    fe.fit(synthetic_df, "A", np.arange(len(synthetic_df)) < 1500)
    full = fe.transform(synthetic_df, "A")
    sens = fe.sensor_cols
    for t in (fe.lookback + 5, 1200, len(synthetic_df) - 1):
        buf = synthetic_df.iloc[t - fe.lookback + 1: t + 1]
        f = fe.transform_last(buf[sens].to_numpy(), buf.timestamp.iloc[-1], "A")
        assert list(f) == list(full.columns)
        np.testing.assert_allclose(np.array(list(f.values()), dtype=float), full.iloc[t].to_numpy(dtype=float), rtol=1e-8, atol=1e-9)
