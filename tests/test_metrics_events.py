import numpy as np
import pandas as pd

from src.evaluation.events import alarm_episodes, persistence_alarm
from src.evaluation.metrics import classification_metrics, select_threshold, threshold_for_precision


def test_classification_metrics():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.7, 0.4, 0.2, 0.9])
    m = classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["fpr"], 1 / 3)


def test_threshold_respects_min_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1)
    t = select_threshold(y, p, "f1_min_recall", min_recall=0.9)
    assert classification_metrics(y, p, t)["recall"] >= 0.9
    tc = threshold_for_precision(y, p, 0.5)
    assert tc >= t or classification_metrics(y, p, tc)["precision"] >= 0.5


def test_persistence_and_episodes():
    prob = np.array([0, .9, 0, .9, .9, .9, 0, 0, .9, .9, 0, 0, 0, 0, 0, 0, .9, .9])
    a = persistence_alarm(prob, 0.5, 2)
    assert list(np.where(a)[0]) == [4, 5, 9, 17]
    eps = alarm_episodes(a, np.ones(len(a), bool), merge_gap=4)
    assert eps == [(4, 9), (17, 17)]
    valid = np.ones(len(a), bool); valid[7] = False         # outage in between -> no merge
    assert alarm_episodes(a, valid, merge_gap=4) == [(4, 5), (9, 9), (17, 17)]
    assert not persistence_alarm(np.array([np.nan, 0.9]), 0.5, 2).any()


def test_event_lead_time_and_false_alarms(cfg):
    """Tiny synthetic Prepared-like object: one failure, one warning 2 h ahead, one false alarm."""
    from src.evaluation.events import evaluate_events
    from src.preprocessing.splits import SplitBounds

    n = 1000
    class P:  # minimal duck-typed Prepared
        offsets = {"A": 0}
        bounds = {"A": SplitBounds(n=n, train_end=100, val_end=200, warmup=0, emb_in=0, emb_out=0)}
        horizon = 8
        timestamps = pd.date_range("2020", periods=n, freq="15min").to_numpy()
        idx = {"test": np.arange(200, 1000)}
    probs = np.zeros(800)
    probs[600 - 200 - 8: 600 - 200] = 0.9      # alarm 8 steps (2 h) before failure at 600
    probs[300 - 200: 305 - 200] = 0.9          # false alarm
    ev = pd.DataFrame([{"asset": "A", "event_id": 0, "kind": "failure", "type": "thermal_runaway",
                        "onset_idx": 560, "failure_idx": 600}])
    out = evaluate_events(P, "test", probs, 0.5, ev, cfg)["summary"]
    assert out["n_detected"] == 1
    assert out["lead_time_median_min"] == 7 * 15     # first alarm after k=2 persistence
    assert out["false_alarm_episodes"] == 1
