"""Model explanations (associational, NOT causal).

* permutation importance (per feature and per sensor group) — drop in PR-AUC
  when a feature's whole input sequence is shuffled across samples;
* integrated gradients — time x feature attribution for positive windows;
* attention weights — where in the window an attention model looks;
* SHAP (TreeExplainer) for the gradient-boosting baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def feature_group(name: str) -> str:
    if name.startswith("OT_resid"):
        return "thermal residual (OT vs load)"
    for s in SENSORS:
        if name == s or name.startswith(s + "_"):
            return s
    if name.startswith("load_total"):
        return "total load"
    return "calendar (hour/day)"


def gather_windows(X: np.ndarray, rows: np.ndarray, L: int) -> np.ndarray:
    return X[rows[:, None] + np.arange(-L + 1, 1)[None, :]]


def eval_sample(y: np.ndarray, rows: np.ndarray, n_neg: int, seed: int) -> np.ndarray:
    """All positives + a fixed random subset of negatives (keeps memory bounded)."""
    rng = np.random.default_rng(seed)
    pos, neg = rows[y[rows] == 1], rows[y[rows] == 0]
    return np.sort(np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)]))


@torch.no_grad()
def _predict(model, W: np.ndarray, bs: int = 2048) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(W[i:i + bs]))).numpy()
                           for i in range(0, len(W), bs)])


def permutation_importance(model, W: np.ndarray, y: np.ndarray, names: list[str], n_repeats: int = 3,
                           seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, _predict(model, W))
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        groups.setdefault(feature_group(n), []).append(j)

    def drop(cols):
        d = []
        for _ in range(n_repeats):
            Wp = W.copy()
            perm = rng.permutation(len(W))
            Wp[:, :, cols] = W[perm][:, :, cols]
            d.append(base - average_precision_score(y, _predict(model, Wp)))
        return float(np.mean(d)), float(np.std(d))

    feat = pd.DataFrame([(n, *drop([j])) for j, n in enumerate(names)],
                        columns=["feature", "pr_auc_drop", "std"]).sort_values("pr_auc_drop", ascending=False)
    grp = pd.DataFrame([(g, *drop(c)) for g, c in groups.items()],
                       columns=["group", "pr_auc_drop", "std"]).sort_values("pr_auc_drop", ascending=False)
    feat["baseline_pr_auc"] = grp["baseline_pr_auc"] = base
    return feat, grp


def integrated_gradients(model, W: np.ndarray, steps: int = 24, bs: int = 256) -> np.ndarray:
    """IG w.r.t. a zero baseline (= training mean after scaling). Returns (N, L, F)."""
    model.eval()
    torch.backends.cudnn.enabled = False
    out = []
    alphas = torch.linspace(1.0 / steps, 1.0, steps)
    for i in range(0, len(W), bs):
        x = torch.from_numpy(W[i:i + bs])
        total = torch.zeros_like(x)
        for a in alphas:
            xi = (a * x).requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                model.train(False)
                logit = model(xi).sum()
            g, = torch.autograd.grad(logit, xi)
            total += g
        out.append((x * total / steps).detach().numpy())
    return np.concatenate(out)


def group_attribution(attr: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |IG| per sensor group and per time step (N, L, F) -> (L, groups)."""
    a = np.abs(attr).mean(axis=0)                       # (L, F)
    df = pd.DataFrame(a, columns=names)
    return df.T.groupby([feature_group(n) for n in names]).sum().T


@torch.no_grad()
def attention_profile(model, W: np.ndarray) -> np.ndarray | None:
    if getattr(model, "attn", None) is None:
        return None
    model.eval()
    ws = []
    for i in range(0, len(W), 2048):
        model(torch.from_numpy(W[i:i + 2048]))
        ws.append(model.last_attention.numpy())
    return np.concatenate(ws)


def shap_tree(model, X: np.ndarray, names: list[str], n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, tuple] | None:
    try:
        import shap

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        sv = shap.TreeExplainer(model).shap_values(X[idx])
        sv = sv[1] if isinstance(sv, list) else sv
        if sv.ndim == 3:
            sv = sv[..., 1]
        return pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
            "mean_abs_shap", ascending=False), (sv, X[idx])
    except Exception as e:  # pragma: no cover - optional dependency path
        print(f"SHAP unavailable: {e}")
        return None
