"""Formal temporal-integrity suite.

Each test checks one guarantee an ML interviewer would ask about, against the REAL pipeline,
data and artefacts (skipped if the artefacts have not been built).
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from src.utils.config import PROJECT_ROOT, load_config

HAVE = (PROJECT_ROOT / "data" / "processed" / "ETTm1.parquet").exists() and \
       (PROJECT_ROOT / "models" / "phase2" / "deployment.json").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="processed data / deployment artefacts not built")
SENS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def prep(cfg):
    from src.data.pipeline import prepare
    return prepare(cfg, "stationary", 8)


@pytest.fixture(scope="module")
def data(cfg):
    from src.data.dataset import load_processed
    return load_processed(cfg)


# ------------------------------------------------------------------ 1. no future information in features
def test_no_centered_or_negative_shift_windows_in_source():
    for f in ["src/features/engineering.py", "src/data/pipeline.py", "backend/intelligence.py"]:
        src = (PROJECT_ROOT / f).read_text()
        assert "center=True" not in src, f"centered rolling window in {f}"
        assert not re.search(r"shift\(\s*-", src), f"negative shift (future value) in {f}"


@pytest.mark.parametrize("variant", [{}, {"robust": True}, {"adaptive": True}])
def test_features_invariant_to_future_on_real_data(cfg, data, variant):
    from src.features.engineering import FeatureEngineer
    df = data[0]["ETTm1"].iloc[:5000].reset_index(drop=True)
    fe = FeatureEngineer.from_config(cfg, **variant)
    fe.fit(df, "ETTm1", np.arange(len(df)) < 2500)
    base = fe.transform(df, "ETTm1")
    for t in (1500, 3200, 4600):
        pert = df.copy()
        pert.loc[t + 1:, SENS] = pert.loc[t + 1:, SENS] * 2.3 - 7
        new = fe.transform(pert, "ETTm1")
        np.testing.assert_array_equal(np.nan_to_num(base.iloc[: t + 1].to_numpy(), nan=-1e9),
                                      np.nan_to_num(new.iloc[: t + 1].to_numpy(), nan=-1e9))


# ------------------------------------------------------------------ 2. train-only fitting
def test_scalers_fitted_on_training_rows_only(cfg, data, prep):
    for a, df in data[0].items():
        b = prep.bounds[a]
        F = prep.fe.transform(df, a)[prep.extras["scaler_columns"]].to_numpy(float)
        off = df.offline.to_numpy(bool)
        np.testing.assert_allclose(prep.scaler.mean_[a], np.nanmean(F[: b.train_end][~off[: b.train_end]], axis=0), rtol=1e-6)


def test_feature_fits_unchanged_by_non_training_data(cfg, data):
    from src.features.engineering import FeatureEngineer
    df = data[0]["ETTm1"]
    b = int(len(df) * cfg["split"]["train_frac"])
    fe1 = FeatureEngineer.from_config(cfg).fit(df, "A", np.arange(len(df)) < b)
    mod = df.copy(); mod.loc[b:, "OT"] += 40
    fe2 = FeatureEngineer.from_config(cfg).fit(mod, "A", np.arange(len(df)) < b)
    np.testing.assert_allclose(fe1.fits["A"].thermal_coef, fe2.fits["A"].thermal_coef)


# ------------------------------------------------------------------ 3. simulator metadata & target leakage
def test_features_ignore_simulator_metadata_and_labels(cfg, data, prep):
    df = data[0]["ETTm1"].iloc[:4000].reset_index(drop=True)
    base = prep.fe.transform(df, "ETTm1")
    tampered = df.copy()
    for c in [c for c in df.columns if c.startswith("sim_")] + ["failure", "offline"]:
        tampered[c] = np.random.default_rng(0).permutation(tampered[c].to_numpy())
    pd.testing.assert_frame_equal(base, prep.fe.transform(tampered, "ETTm1"))
    assert not [c for c in prep.feature_names if c.startswith(("sim_", "failure", "offline"))]


def test_labels_only_use_future_failures(prep):
    from src.data.dataset import make_labels
    for a, off in prep.offsets.items():
        n = prep.bounds[a].n
        f = prep.failure[off: off + n]
        y = make_labels(f, prep.horizon)
        i = np.where(f == 1)[0][3]
        assert y[i] == 0 and y[i - 1] == 1 and y[i - prep.horizon] == 1 and y[i - prep.horizon - 1] == 0


# ------------------------------------------------------------------ 4. split / sequence integrity & warm-up
@pytest.mark.parametrize("L", [48, 96])
def test_sequences_never_cross_split_boundaries(prep, L):
    for a, off in prep.offsets.items():
        b = prep.bounds[a]
        for s in ("train", "val", "test"):
            t = prep.idx[s][(prep.idx[s] >= off) & (prep.idx[s] < off + b.n)] - off
            lo, hi = b.period(s)
            assert (t - L + 1 >= lo).all() if s != "train" else (t - L + 1 >= 0).all()
            if s != "test":
                assert (t + prep.horizon < hi).all(), f"{s} label window reaches the next split"
        sel = {s: prep.idx[s][(prep.idx[s] >= off) & (prep.idx[s] < off + b.n)] for s in ("train", "val", "test")}
        assert sel["train"].max() < sel["val"].min() and sel["val"].max() < sel["test"].min()   # chronological per asset


def test_every_scored_window_is_past_warm_up(prep):
    """No scored target sees NaN-derived (incomplete) features anywhere in its 48-step window."""
    from src.data.dataset import load_processed
    dfs, _ = load_processed(load_config())
    for a, off in prep.offsets.items():
        F = prep.fe.transform(dfs[a], a)[prep.feature_names]
        complete = F.notna().all(axis=1).to_numpy()
        for s in ("train", "val", "test"):
            t = prep.idx[s][(prep.idx[s] >= off) & (prep.idx[s] < off + prep.bounds[a].n)] - off
            first = t.min()
            assert complete[first - 47: first + 1].all(), f"{a}/{s}: first window contains warm-up rows"


# ------------------------------------------------------------------ 5. decisions never depend on test data
def _scramble_test_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(1)
    for c in out.columns:
        if c.startswith("test") and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = rng.random(len(out))
    return out


def test_phase2_model_decision_ignores_test_columns():
    from src.phase2 import report as R
    r = pd.read_csv(PROJECT_ROOT / "experiments" / "phase2" / "results.csv")
    d1 = R.decide(r, R.summarise(r))
    d2 = R.decide(_scramble_test_columns(r), R.summarise(_scramble_test_columns(r)))
    assert d1["winner"] == d2["winner"]
    assert [x["replaces_incumbent"] for x in d1["table"]] == [x["replaces_incumbent"] for x in d2["table"]]


def test_phase1_final_selection_ignores_test_columns(cfg):
    from src.training.final import selection_score
    fc = pd.read_csv(PROJECT_ROOT / "experiments" / "final_candidates.csv")
    w = cfg["selection"]["weights"]
    a = selection_score(fc, w).to_numpy()
    b = selection_score(_scramble_test_columns(fc).assign(**{c: fc[c] for c in fc if not c.startswith("test")}), w).to_numpy()
    np.testing.assert_allclose(a, b)


def test_protocol_runs_contain_validation_only():
    p = PROJECT_ROOT / "reports" / "final" / "protocol_runs.csv"
    if not p.exists():
        pytest.skip("protocol not run")
    df = pd.read_csv(p)
    assert not [c for c in df.columns if c.startswith("test")]


# ------------------------------------------------------------------ 6. thresholds & calibration come from validation
def test_deployed_threshold_and_calibrator_reproduce_from_fresh_validation(cfg):
    """Re-derive the operating point from the pooled fresh VALIDATION realisations: must equal the deployment."""
    from src.evaluation.metrics import select_threshold, threshold_for_precision
    from src.final.common import REAL_SEEDS, predict, realisation
    from src.phase2.calibration import Calibrator
    from src.streaming.artifacts import load_artifacts
    man = json.loads((PROJECT_ROOT / "models" / "phase2" / "deployment.json").read_text())
    if "thresholds" not in man:
        pytest.skip("final operating point not applied")
    art = load_artifacts(cfg)
    ys, ps = [], []
    for s in REAL_SEEDS:
        q = realisation(cfg, art, s)
        ys.append(q.y[q.idx["val"]]); ps.append(predict(art.model, q, q.idx["val"]))
    y, p = np.concatenate(ys), np.concatenate(ps)
    t = cfg["threshold"]
    w = select_threshold(y, p, t["objective"], t["min_recall"], t["max_fpr"])
    assert w == pytest.approx(man["thresholds"]["warning_raw"], abs=1e-6)
    assert max(w, threshold_for_precision(y, p, cfg["alerts"]["critical_precision"])) == pytest.approx(man["thresholds"]["critical_raw"], abs=1e-6)
    cal = Calibrator.from_dict(json.loads((PROJECT_ROOT / "models" / "phase2" / man["calibrator"]).read_text()))
    grid = np.linspace(0, 1, 201)
    np.testing.assert_allclose(cal(grid), Calibrator("isotonic").fit(p, y)(grid), atol=1e-9)


# ------------------------------------------------------------------ 7. reproducibility of the benchmark
def test_reference_benchmark_regenerates_bit_for_bit(cfg, data):
    from src.phase2.resim import simulate
    dfs, ev = simulate(cfg, cfg["simulation"]["seed"])
    for a, df in dfs.items():
        pd.testing.assert_frame_equal(df, data[0][a])
