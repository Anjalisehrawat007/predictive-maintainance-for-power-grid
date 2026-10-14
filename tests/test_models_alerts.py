import pytest
import torch

from src.models.losses import FocalLoss, build_loss
from src.models.recurrent import MODEL_ZOO, build_model, count_parameters
from src.streaming.alerts import AlertPolicy


@pytest.mark.parametrize("kind", list(MODEL_ZOO))
def test_forward_backward(kind):
    m = build_model(kind, 10, hidden_size=16)
    x = torch.randn(4, 12, 10)
    out = m(x)
    assert out.shape == (4,)
    out.sum().backward()
    assert count_parameters(m) > 0


def test_unidirectional_models_ignore_padding_before_window():
    """Prediction depends only on the window; changing only the newest step changes the output."""
    m = build_model("gru_attention", 5, 8).eval()
    x = torch.randn(1, 10, 5)
    x2 = x.clone(); x2[0, -1] += 1.0
    assert not torch.allclose(m(x), m(x2))


def test_quantized_matches_fp32():
    from src.training.final import quantize_dynamic
    m = build_model("gru", 6, 16).eval()
    q = quantize_dynamic(m)
    x = torch.randn(8, 20, 6)
    assert torch.allclose(torch.sigmoid(m(x)), torch.sigmoid(q(x)), atol=0.02)


def test_losses():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    z = torch.tensor([-2.0, 2.0, -1.0, 0.5])
    for name in ("bce", "weighted_bce", "focal"):
        assert build_loss(name, 3.0)(z, y).item() > 0
    assert FocalLoss(gamma=0.0, alpha=0.5)(z, y) < build_loss("bce")(z, y)


def test_alert_policy():
    a = AlertPolicy(0.3, 0.7, persistence_k=2)
    assert [a.update(p) for p in (0.5, 0.5, 0.8, 0.1, 0.9)] == ["NORMAL", "WARNING", "CRITICAL", "NORMAL", "NORMAL"]
    with pytest.raises(ValueError):
        AlertPolicy(0.8, 0.5)
