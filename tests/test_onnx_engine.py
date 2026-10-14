"""The dependency-light ONNX engine (serverless deployment) must reproduce the reference engine."""
import numpy as np
import pytest

from src.utils.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "ensemble_member0.onnx").exists(),
                                reason="ONNX member exports not built (python -m src.streaming.export_onnx)")


@pytest.fixture(scope="module")
def engines():
    from backend.intelligence import load_deployment
    return load_deployment(engine="torch"), load_deployment(engine="onnx")


def _run(dep, df, warm):
    from backend.sessions import Session
    s = Session(dep, {"ETTm2": df}, ["ETTm2"], start=warm, stop=len(df))
    return s.step(len(df))["records"]["ETTm2"]


def test_onnx_engine_matches_reference(engines):
    from src.scenario.scenario import build
    ref, lite = engines
    assert lite.art.model is None and type(lite.scorer).__name__ == "OnnxScorer"
    warm = ref.art.fe.lookback + ref.art.seq_len - 1
    df, meta = build("ETTm2", 7, "load_oscillation", 1.2, 8, 11, warm)
    a, b = _run(ref, df, warm), _run(lite, df, warm)
    assert len(a) == len(b) > 50
    for x, y in zip(a, b):
        assert x["status"] == y["status"]
        if x["raw_score"] is not None:
            assert abs(x["raw_score"] - y["raw_score"]) < 1e-5 and abs(x["risk"] - y["risk"]) < 1e-5
            assert abs(x["confidence"] - y["confidence"]) < 1e-5
    fa = [r for r in a if r["ground_truth"]["failure_event"]][0]
    fb = [r for r in b if r["ground_truth"]["failure_event"]][0]
    assert fa.get("lead_time_min") == fb.get("lead_time_min")
    last = [r for r in b if r["raw_score"] is not None][-1]
    assert abs(sum(s["share"] for s in last["attribution"]) - 1) < 1e-6 and len(last["temporal_attribution"]) == 48


def test_serving_calibrator_equals_sklearn_isotonic():
    import json

    from src.phase2.calibration import Calibrator
    from src.streaming.calibration import ServingCalibrator
    d = json.loads((PROJECT_ROOT / "models" / "phase2" / "calibrator_final.json").read_text())
    p = np.linspace(-0.05, 1.05, 2001)
    assert np.allclose(Calibrator.from_dict(d)(p), ServingCalibrator.from_dict(d)(p), atol=1e-12)
