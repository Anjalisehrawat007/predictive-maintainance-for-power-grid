"""API hardening: validation, graceful failures, service endpoints, security regressions."""
import pandas as pd
import pytest

from src.utils.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "reports" / "phase2" / "engine_log" / "ETTm1.parquet").exists(),
                                reason="engine log not built")
SENS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:          # runs the lifespan → model loaded once
        yield c


def _rows(n, start=60000):
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "ETTm1.parquet").iloc[start:start + n]
    return [{"timestamp": str(t), **{k: float(r[k]) for k in SENS}} for t, (_, r) in zip(df.timestamp, df.iterrows())]


def test_service_endpoints(client):
    assert client.get("/api/health").json()["ok"] is True
    r = client.get("/api/ready")
    assert r.status_code == 200 and r.json()["ready"] is True
    m = client.get("/api/model").json()
    assert m["parameters"] > 0 and m["calibration"]["fitted_on"].startswith("validation")
    assert len(m["input"]["features"]) == m["input"]["n_features"]
    assert "X-Request-ID" in r.headers and "X-Response-Time-ms" in r.headers
    assert client.get("/api/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"
    assert "routes" in client.get("/api/metrics").json()


def test_validation_errors(client):
    assert client.get("/api/log?asset=NOPE&start=1&end=2").status_code == 404
    assert client.get("/api/log?asset=ETTm1&start=60010&end=60000").status_code == 422
    assert client.get("/api/log?asset=ETTm1&start=abc&end=2").status_code == 422
    assert client.post("/api/sessions", json={"start_index": "x"}).status_code == 422
    assert client.post("/api/sessions", json={"assets": ["NOPE"]}).status_code in (400, 404)
    assert client.post("/api/sessions/doesnotexist/step", json={"n": 1}).status_code == 404
    assert client.post("/api/sessions/doesnotexist/step", json={"n": 10_000}).status_code == 422
    assert client.post("/api/incidents/ETTm1/1/replay", json={"hours_before": 1000}).status_code == 422
    assert client.post("/api/incidents/NOPE/1/replay").status_code == 404
    assert client.post("/api/alerts/not-an-id/ack").status_code == 422
    assert client.post("/api/alerts/log-999999/ack").status_code == 404


def test_predict_endpoint(client):
    rows = _rows(200)
    r = client.post("/api/predict", json={"asset": "ETTm1", "rows": rows}).json()
    assert r["n_scored"] == 200 - r["first_scored_row"]
    p = r["predictions"][-1]
    assert 0 <= p["raw_score"] <= 1 and 0 <= p["risk"] <= 1 and p["status"] in ("NORMAL", "WARNING", "CRITICAL")
    assert client.post("/api/predict", json={"asset": "ETTm1", "rows": rows[:20]}).status_code == 422      # too short
    assert client.post("/api/predict", json={"asset": "ETTm1", "rows": rows[::-1]}).status_code == 422    # not increasing
    bad = [dict(rows[0], OT="nan")] + rows[1:]
    assert client.post("/api/predict", json={"asset": "ETTm1", "rows": bad}).status_code == 422
    assert client.post("/api/predict", json={"asset": "NOPE", "rows": rows}).status_code == 404


def test_predict_matches_engine_log(client):
    """/predict on raw history must reproduce the engine's raw score at the same timestamp."""
    rows = _rows(200, 60000)
    r = client.post("/api/predict", json={"asset": "ETTm1", "rows": rows}).json()
    last = r["predictions"][-1]
    lg = client.get("/api/log?asset=ETTm1&start=60199&end=60200").json()[0]
    assert abs(lg["raw_score"] - last["raw_score"]) < 1e-4


@pytest.fixture
def built_dist(tmp_path, monkeypatch):
    """A minimal frontend build in a temp dir (dist/ is git-ignored, so tests must not depend on a local build),
    with a 'secret' file next to it that must never be reachable."""
    import backend.app as appmod
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>SmartGrid AI</title>")
    (dist / "robots.txt").write_text("ok")
    (tmp_path / "secret.txt").write_text("TOP-SECRET")
    monkeypatch.setattr(appmod, "DIST", dist.resolve())
    return dist


def test_path_traversal_blocked(client, built_dist):
    assert client.get("/robots.txt").text == "ok"                         # files inside dist/ are served
    for url in ["/..%2Fsecret.txt", "/%2e%2e/secret.txt", "/../secret.txt", "/..%2F..%2Fconfigs%2Fconfig.yaml",
                "/%2e%2e/%2e%2e/backend/app.py"]:
        r = client.get(url)
        assert "TOP-SECRET" not in r.text and "seed:" not in r.text and "FastAPI" not in r.text


def test_unknown_api_route_is_404_not_spa(client):
    assert client.get("/api/does-not-exist").status_code == 404


def test_model_unavailable_degrades_gracefully(client, monkeypatch):
    import backend.app as A
    monkeypatch.setitem(A._STATE, "dep", None)
    monkeypatch.setitem(A._STATE, "error", "FileNotFoundError: weights missing")
    assert client.get("/api/health").status_code == 200
    r = client.get("/api/ready")
    assert r.status_code == 503 and r.json()["checks"]["model_loaded"] is False
    s = client.get("/api/system")
    assert s.status_code == 503 and "model unavailable" in s.json()["detail"]
    assert client.post("/api/sessions", json={}).status_code == 503


def test_missing_engine_log(client, monkeypatch):
    import backend.app as A
    monkeypatch.setitem(A.STORE.__dict__, "log", {})
    assert client.get("/api/timeline").status_code == 503
    assert client.get("/api/snapshot?index=60000").status_code == 503


def test_spa_shell_is_not_cached(client, built_dist):
    r = client.get("/assets/ETTm1")                                       # a client-side route → SPA shell
    assert r.status_code == 200 and "SmartGrid AI" in r.text and "no-cache" in r.headers.get("cache-control", "")


def test_unbuilt_frontend_gives_clear_404(client, tmp_path, monkeypatch):
    import backend.app as appmod
    monkeypatch.setattr(appmod, "DIST", tmp_path / "missing")
    r = client.get("/")
    assert r.status_code == 404 and "npm run build" in r.json()["detail"]


def test_incident_list_is_consistent_with_final_evaluation(client):
    """Every lead time the UI can show for a test failure comes from the deployed operating point:
    the engine log and the offline evaluation (reports/final/test_per_failure.csv) agree for all 40."""
    inc = client.get("/api/incidents").json()
    assert len(inc) == 40
    for i in inc:
        assert i.get("lead_time_min") == i.get("engine_lead_time_min"), i


def test_confidence_bins_single_source(client):
    from src.phase2.calibration import CONFIDENCE_BINS
    assert client.get("/api/system").json()["thresholds"]["confidence_bins"] == CONFIDENCE_BINS
    assert client.get("/api/model").json()["thresholds"]["confidence_bins"] == CONFIDENCE_BINS


def test_cli_and_api_use_the_same_thresholds(client):
    """One source of truth: AlertPolicy.from_config (CLI tools) resolves the same thresholds as the API."""
    from src.streaming.alerts import AlertPolicy
    from src.streaming.artifacts import load_artifacts
    from src.utils.config import load_config
    cfg = load_config()
    pol = AlertPolicy.from_config(cfg, load_artifacts(cfg).meta)
    th = client.get("/api/model").json()["thresholds"]
    assert (pol.warning_threshold, pol.critical_threshold) == (th["warning_raw"], th["critical_raw"])
