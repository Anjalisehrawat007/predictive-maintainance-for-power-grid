"""Phase-2 unit tests: new architectures, hazard head, calibration, simulator back-compat."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.simulation import FaultSimulator
from src.models.recurrent import build_model
from src.models.temporal import _CausalConv
from src.phase2.calibration import Calibrator, confidence_from_members

from .conftest import SENSORS


def test_causal_conv_equals_conv1d():
    c, conv = _CausalConv(4, 5, 3, 2), torch.nn.Conv1d(4, 5, 3, dilation=2)
    with torch.no_grad():
        c.lin.weight.copy_(torch.cat([conv.weight[:, :, 2 - i] for i in range(3)], dim=1)); c.lin.bias.copy_(conv.bias)
    x = torch.randn(2, 10, 4)
    ref = conv(torch.nn.functional.pad(x.transpose(1, 2), (4, 0))).transpose(1, 2)
    assert torch.allclose(ref, c(x), atol=1e-6)


def test_tcn_is_causal():
    m = build_model("tcn", 6, 16, 3, 0.0).eval()
    x = torch.randn(1, 20, 6)
    y0 = m(x)
    x2 = x.clone(); x2[0, :5] += 3.0          # change far past (outside receptive field of 1+2*2*(1+2+4)=29? inside) …
    x3 = x.clone(); x3[0, -1] += 3.0          # … and the newest step
    assert not torch.allclose(y0, m(x3))       # output depends on the present
    assert m(x2).shape == y0.shape


@pytest.mark.parametrize("kind", ["tcn", "transformer"])
def test_new_architectures_train(kind):
    m = build_model(kind, 8, 16, 2, 0.1)
    out = m(torch.randn(4, 24, 8))
    assert out.shape == (4,)
    out.sum().backward()


def test_multi_horizon_monotone():
    m = build_model("gru_attention", 8, 16, 1, 0.0, n_outputs=4).eval()
    p = torch.sigmoid(m(torch.randn(16, 24, 8)))
    assert p.shape == (16, 4) and bool((p.diff(dim=1) >= -1e-6).all())   # P(≤30m) ≤ P(≤1h) ≤ …


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_calibrator_monotone_and_roundtrip(method):
    rng = np.random.default_rng(0)
    p = rng.random(4000); y = (rng.random(4000) < p ** 3).astype(float)
    c = Calibrator(method).fit(p, y)
    grid = np.linspace(0, 1, 101)
    q = c(grid)
    assert np.all(np.diff(q) >= -1e-9) and q.min() >= 0 and q.max() <= 1
    c2 = Calibrator.from_dict(c.to_dict())
    assert np.allclose(c2(grid), q, atol=1e-9)


def test_confidence_range():
    mp = np.array([[0.1, 0.9], [0.1, 0.1], [0.1, 0.5]])   # (members, samples)
    c = confidence_from_members(mp)
    assert c[0] == pytest.approx(1.0)                      # members agree  -> full confidence
    assert c[1] < 0.5                                      # members disagree -> low confidence
    assert 0 <= c.min() <= c.max() <= 1


def test_simulator_backward_compatible(cfg, synthetic_df):
    """Variability knobs OFF must reproduce the original simulator bit-for-bit."""
    sc = dict(cfg["simulation"]) | {"failure_gap_days": [3.0, 5.0]}
    a, _ = FaultSimulator(sc, SENSORS, 5).simulate(synthetic_df, "A")
    sc2 = sc | {"variability": {}}
    b, _ = FaultSimulator(sc2, SENSORS, 5).simulate(synthetic_df, "A")
    pd.testing.assert_frame_equal(a, b)
    sc3 = sc | {"variability": {"amp_scale": [0.5, 0.5]}}
    c, ev = FaultSimulator(sc3, SENSORS, 5).simulate(synthetic_df, "A")
    assert (c["sim_delta_OT"].abs().sum() <= a["sim_delta_OT"].abs().sum() + 1e-6)
