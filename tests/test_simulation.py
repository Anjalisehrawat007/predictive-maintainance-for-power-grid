import numpy as np

from src.data.simulation import FaultSimulator

from .conftest import SENSORS


def _sim(cfg, df, seed=1):
    sc = dict(cfg["simulation"]) | {"failure_gap_days": [3.0, 5.0], "nuisance_per_day": 0.3}
    return FaultSimulator(sc, SENSORS, seed).simulate(df, "A")


def test_reproducible(cfg, synthetic_df):
    a, ea = _sim(cfg, synthetic_df)
    b, eb = _sim(cfg, synthetic_df)
    assert a[SENSORS].equals(b[SENSORS]) and ea.equals(eb)


def test_real_signal_recoverable(cfg, synthetic_df):
    out, _ = _sim(cfg, synthetic_df)
    for c in SENSORS:
        np.testing.assert_allclose(out[c] - out[f"sim_delta_{c}"], synthetic_df[c], atol=1e-9)


def test_event_structure(cfg, synthetic_df):
    out, ev = _sim(cfg, synthetic_df)
    fe = ev[ev.kind == "failure"]
    assert len(fe) >= 3
    assert out.failure.sum() == len(fe)
    for r in fe.itertuples():
        assert out.failure.iloc[r.failure_idx] == 1
        assert r.onset_idx <= r.failure_idx
        assert out.offline.iloc[r.failure_idx]                      # trip -> offline
        if r.type != "sudden":
            assert (out.sim_phase.iloc[r.onset_idx:r.failure_idx] == "precursor").all()
        else:
            assert r.onset_idx == r.failure_idx


def test_nuisance_never_overlaps_failures(cfg, synthetic_df):
    out, ev = _sim(cfg, synthetic_df)
    for r in ev[ev.kind == "nuisance"].itertuples():
        assert not out.offline.iloc[r.onset_idx:r.end_idx].any()
        assert not (out.sim_phase.iloc[r.onset_idx:r.end_idx] == "precursor").any()
