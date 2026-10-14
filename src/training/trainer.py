"""Training loop for sequence classifiers.

Windows are gathered on the fly from the (N, F) feature matrix, so no
(N, L, F) tensor is ever materialised.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.models.losses import build_loss
from src.utils.logging import get_logger

log = get_logger("training")


class WindowBatcher:
    """Gathers (B, L, F) windows ending at the given target rows."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, device: torch.device):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        self.L = seq_len
        self.offsets = torch.arange(-seq_len + 1, 1)
        self.device = device

    def get(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        xb = self.X[r[:, None] + self.offsets[None, :]]
        return xb.to(self.device, non_blocking=True), self.y[r].to(self.device, non_blocking=True)


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_epoch: int
    best_val_pr_auc: float
    history: list[dict] = field(default_factory=list)
    train_time_s: float = 0.0


@torch.no_grad()
def predict_proba(model: torch.nn.Module, batcher: WindowBatcher, rows: np.ndarray,
                  batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        xb, _ = batcher.get(rows[i: i + batch_size])
        out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _epoch_rows(train_rows: np.ndarray, y: np.ndarray, neg_ratio: float | None,
                rng: np.random.Generator) -> np.ndarray:
    if not neg_ratio:
        return rng.permutation(train_rows)
    pos = train_rows[y[train_rows] == 1]
    neg = train_rows[y[train_rows] == 0]
    neg = rng.choice(neg, size=min(len(neg), int(neg_ratio * len(pos))), replace=False)
    return rng.permutation(np.concatenate([pos, neg]))


def train_model(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, train_rows: np.ndarray,
                val_rows: np.ndarray, seq_len: int, tcfg: dict, device: torch.device,
                seed: int = 42, trial=None, ckpt_path=None) -> TrainResult:
    """Train with early stopping on validation PR-AUC (threshold-free, imbalance-aware)."""
    rng = np.random.default_rng(seed)
    torch.set_num_threads(int(tcfg.get("num_threads", 4)))
    model.to(device)
    batcher = WindowBatcher(X, y, seq_len, device)
    ytr = y[train_rows]
    pw = tcfg.get("pos_weight", "auto")
    if ytr.ndim == 2:                       # multi-horizon: one pos_weight per output
        n_pos, n_neg = (ytr == 1).sum(0).astype(float), (ytr == 0).sum(0).astype(float)
        pos_weight = torch.tensor(np.sqrt(n_neg / np.maximum(n_pos, 1)), dtype=torch.float32)
    else:
        n_pos, n_neg = float((ytr == 1).sum()), float((ytr == 0).sum())
        pos_weight = float(np.sqrt(n_neg / max(n_pos, 1))) if pw == "auto" else float(pw)
    primary = int(tcfg.get("primary_col", 0))
    criterion = build_loss(tcfg["loss"], pos_weight, tcfg.get("focal_gamma", 2.0),
                           tcfg.get("focal_alpha", 0.75)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    yval = y[val_rows] if y.ndim == 1 else y[val_rows][:, primary]
    best, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    t_start = time.time()
    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        rows = _epoch_rows(train_rows, y, tcfg.get("negative_ratio"), rng)
        losses = []
        t0 = time.time()
        for i in range(0, len(rows), tcfg["batch_size"]):
            xb, yb = batcher.get(rows[i: i + tcfg["batch_size"]])
            if tcfg.get("input_noise", 0.0):
                xb = xb + tcfg["input_noise"] * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        pv = predict_proba(model, batcher, val_rows)
        if pv.ndim == 2:
            pv = pv[:, primary]
        val_ap = float(average_precision_score(yval, pv))
        sched.step(val_ap)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": val_ap,
                        "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t0})
        log.info("epoch %02d loss=%.4f val_PR-AUC=%.4f lr=%.1e (%.1fs)", epoch, np.mean(losses), val_ap,
                 opt.param_groups[0]["lr"], time.time() - t0)
        if val_ap > best + 1e-4:
            best, best_epoch, bad = val_ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)
        else:
            bad += 1
        if trial is not None:
            trial.report(val_ap, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        if bad >= tcfg["patience"]:
            log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
            break
    model.load_state_dict(best_state)          # restore best-validation weights
    return TrainResult(model, best_epoch, best, history, time.time() - t_start)
