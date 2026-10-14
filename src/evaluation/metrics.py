"""Sample-level classification metrics and threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "threshold": float(thr),
    }


def select_threshold(y: np.ndarray, p: np.ndarray, objective: str = "f1_min_recall",
                     min_recall: float = 0.7, max_fpr: float = 0.01) -> float:
    """Choose the decision threshold on VALIDATION data.

    * ``f1``            — maximise F1.
    * ``f1_min_recall`` — maximise F1 among thresholds with recall >= min_recall
                          (falls back to max-F1 if infeasible).
    * ``recall_max_fpr``— maximise recall subject to FPR <= max_fpr.
    """
    y = np.asarray(y).astype(int)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if objective == "recall_max_fpr":
        neg = (y == 0).sum()
        order = np.argsort(p)[::-1]
        ps = np.asarray(p)[order]; ys = y[order]
        fp = np.cumsum(ys == 0); np.cumsum(ys == 1)
        ok = fp / neg <= max_fpr
        i = np.where(ok)[0].max() if ok.any() else 0
        return float(ps[i])
    if objective == "f1_min_recall":
        ok = rec >= min_recall
        if ok.any():
            i = np.where(ok)[0][np.argmax(f1[ok])]
            return float(thr[i])
    return float(thr[np.argmax(f1)])


def threshold_for_precision(y: np.ndarray, p: np.ndarray, target_precision: float) -> float:
    """Smallest threshold whose validation precision >= target (used for CRITICAL)."""
    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    ok = np.where(prec[:-1] >= target_precision)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[-1])
