"""Causal sequence classifiers: LSTM / GRU / (Bi)LSTM / attention / lightweight.

All models consume a window ``x[t-L+1 .. t]`` (B, L, F) of *past* observations
and output one logit for P(failure in (t, t+H]).  A bidirectional RNN is
included only as a comparison: it is still causal w.r.t. the prediction time
(the window contains only past data), but it cannot reuse state between
streaming steps and costs ~2x compute.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over time."""

    def __init__(self, hidden: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or max(8, hidden // 2)
        self.proj = nn.Linear(hidden, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.score(torch.tanh(self.proj(h))).squeeze(-1)    # (B, L)
        a = torch.softmax(e, dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a


class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, cell: str = "gru", bidirectional: bool = False,
                 attention: bool = False, head_hidden: int | None = None):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell, self.attention, self.bidirectional = cell, attention, bidirectional
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out = hidden_size * (2 if bidirectional else 1)
        self.attn = TemporalAttention(out) if attention else None
        feat = out * 2 if attention else out
        head_hidden = head_hidden if head_hidden is not None else max(16, hidden_size // 2)
        self.drop = nn.Dropout(dropout)
        self.head = (nn.Sequential(nn.Linear(feat, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
                     if head_hidden > 0 else nn.Linear(feat, 1))
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, hn = self.rnn(x)
        if self.cell == "lstm":
            hn = hn[0]
        last = torch.cat([hn[-2], hn[-1]], dim=1) if self.bidirectional else hn[-1]
        if self.attn is not None:
            ctx, a = self.attn(h)
            self.last_attention = a.detach()
            z = torch.cat([ctx, last], dim=1)
        else:
            z = last
        return self.head(self.drop(z)).squeeze(-1)


MODEL_ZOO = {
    "lstm": dict(cell="lstm"),
    "bilstm": dict(cell="lstm", bidirectional=True),
    "gru": dict(cell="gru"),
    "lstm_attention": dict(cell="lstm", attention=True),
    "gru_attention": dict(cell="gru", attention=True),
    "gru_lite": dict(cell="gru", head_hidden=0),
}


def build_model(model_type: str, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                dropout: float = 0.2, n_outputs: int = 1) -> nn.Module:
    """Build any architecture by name. ``n_outputs > 1`` returns a multi-horizon hazard model."""
    if n_outputs > 1:
        from src.models.temporal import MultiHorizonHazard

        base = build_model(model_type, input_size, hidden_size, num_layers, dropout)
        last = [m for m in base.modules() if isinstance(m, nn.Linear)][-1]
        new = nn.Linear(last.in_features, n_outputs)
        _replace_module(base, last, new)
        return MultiHorizonHazard(base)
    if model_type == "tcn":
        from src.models.temporal import TCNClassifier

        return TCNClassifier(input_size, hidden_size, num_layers, dropout)
    if model_type == "transformer":
        from src.models.temporal import TinyTransformer

        return TinyTransformer(input_size, hidden_size, num_layers, dropout)
    if model_type not in MODEL_ZOO:
        raise ValueError(f"unknown model type {model_type}; choose from {list(MODEL_ZOO)}")
    return SequenceClassifier(input_size, hidden_size, num_layers, dropout, **MODEL_ZOO[model_type])


def _replace_module(root: nn.Module, old: nn.Module, new: nn.Module) -> None:
    for _name, mod in root.named_modules():
        for cname, child in mod.named_children():
            if child is old:
                setattr(mod, cname, new)
                return
    raise ValueError("module not found")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class SeedEnsemble(nn.Module):
    """Averages member probabilities; returns the logit of the mean probability."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.stack([torch.sigmoid(m(x)) for m in self.members]).mean(0).clamp(1e-7, 1 - 1e-7)
        return torch.log(p / (1 - p))
