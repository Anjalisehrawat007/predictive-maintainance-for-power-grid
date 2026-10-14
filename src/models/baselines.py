"""Classical baselines operating on the feature vector at time t.

Engineered features already summarise the recent past (rolling windows,
differences), so these are *strong* tabular baselines, not memory-less ones.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def gbdt_model(seed: int):
    """LightGBM if importable (needs libomp), else sklearn's histogram GBDT."""
    try:
        import lightgbm as lgb  # noqa: F401

        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  class_weight="balanced", random_state=seed, verbose=-1), "lightgbm"
    except Exception:
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, class_weight="balanced",
                                              early_stopping=False, random_state=seed), "hist_gbdt"


class IsolationForestDetector:
    """Unsupervised anomaly detector trained on normal training rows only."""

    def __init__(self, seed: int):
        self.m = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.m.fit(X[y == 0])
        return self

    def predict_proba(self, X):
        s = -self.m.score_samples(X)                  # higher = more anomalous
        s = 1 / (1 + np.exp(-(s - 0.5) * 10))
        return np.c_[1 - s, s]


class RuleBasedDetector:
    """Hand-crafted heuristic: max |standardised indicator| over expert-chosen features."""

    INDICATORS = ["OT_resid", "OT_resid_diff4", "HUFL_volratio", "MUFL_volratio",
                  "HULL_spikes16", "MULL_spikes16", "OT_zbase"]

    def __init__(self, feature_names: list[str]):
        self.cols = [feature_names.index(c) for c in self.INDICATORS if c in feature_names]
        if not self.cols:
            raise ValueError("rule-based detector needs engineered features")

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = np.abs(X[:, self.cols]).max(axis=1)
        s = 1 / (1 + np.exp(-(s - 3)))
        return np.c_[1 - s, s]


def build_baseline(name: str, seed: int, feature_names: list[str]):
    if name == "logistic_regression":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=5,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "gbdt":
        return gbdt_model(seed)[0]
    if name == "isolation_forest":
        return IsolationForestDetector(seed)
    if name == "rule_based":
        return RuleBasedDetector(feature_names)
    raise ValueError(name)
