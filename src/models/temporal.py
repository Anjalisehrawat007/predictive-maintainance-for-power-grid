"""Phase-2 temporal architectures: TCN, lightweight transformer, multi-horizon hazard head.

All consume a window of PAST observations (B, L, F) and output logits; none looks
beyond the prediction time (the window ends at t).
"""
from __future__ import annotations

import torch
from torch import nn


class _CausalConv(nn.Module):
    """Causal dilated convolution on (B, L, C) tensors, written as a linear map over
    [x_t, x_{t-d}, …, x_{t-(k-1)d}] (identical maths to Conv1d, ~10x faster on CPU here)."""

    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.k, self.d = k, d
        self.lin = nn.Linear(cin * k, cout)

    def forward(self, x):                       # x: (B, L, C)
        L = x.shape[1]
        xp = nn.functional.pad(x, (0, 0, (self.k - 1) * self.d, 0))
        taps = [xp[:, (self.k - 1 - i) * self.d: (self.k - 1 - i) * self.d + L] for i in range(self.k)]
        return self.lin(torch.cat(taps, dim=-1))


class TCNClassifier(nn.Module):
    """Causal dilated residual TCN (Bai et al., 2018); receptive field 1 + 2·(k-1)·Σd = 63 steps."""

    def __init__(self, input_size, hidden_size=32, num_layers=5, dropout=0.2, kernel=3):
        super().__init__()
        self.inp = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            d = 2 ** i
            self.blocks.append(nn.ModuleDict({
                "c1": _CausalConv(hidden_size, hidden_size, kernel, d),
                "c2": _CausalConv(hidden_size, hidden_size, kernel, d),
                "drop": nn.Dropout(dropout)}))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, max(16, hidden_size // 2)), nn.ReLU(),
                                  nn.Linear(max(16, hidden_size // 2), 1))
        self.attn = None

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            r = h
            h = b["drop"](torch.relu(b["c1"](h)))
            h = b["drop"](torch.relu(b["c2"](h)))
            h = torch.relu(h + r)
        return self.head(h[:, -1]).squeeze(-1)


class TinyTransformer(nn.Module):
    """2-layer pre-norm transformer encoder over the window; reads out the last time step
    concatenated with the mean (the window only contains the past, so no mask is needed)."""

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2, nhead=4, max_len=128):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden_size, nhead, 2 * hidden_size, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_size), nn.Dropout(dropout), nn.Linear(2 * hidden_size, 1))
        self.attn = None

    def forward(self, x):
        L = x.shape[1]
        h = self.enc(self.proj(x) + self.pos[:, -L:])
        return self.head(torch.cat([h[:, -1], h.mean(1)], dim=1)).squeeze(-1)


class MultiHorizonHazard(nn.Module):
    """Wraps a backbone that emits K raw scores and turns them into *monotone* cumulative
    risks P(failure ≤ h_k) via discrete-time hazards: P_k = 1 − Π_{j≤k}(1 − σ(z_j)).
    Returns cumulative-probability logits (B, K)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.attn = getattr(backbone, "attn", None)

    @property
    def last_attention(self):
        return getattr(self.backbone, "last_attention", None)

    def forward(self, x):
        z = self.backbone(x)                                     # (B, K) hazard logits
        log_surv = torch.cumsum(-nn.functional.softplus(z), dim=1)   # log Π(1 − h_j)
        log_surv = log_surv.clamp(max=-1e-6)
        return torch.log(-torch.expm1(log_surv)) - log_surv      # logit(1 − S_k)
