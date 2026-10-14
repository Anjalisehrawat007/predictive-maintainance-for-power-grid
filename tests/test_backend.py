"""API / intelligence-engine tests (need the trained artefacts and the engine log)."""

import numpy as np
import pytest

from src.utils.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "reports" / "phase2" / "engine_log" / "ETTm1.parquet").exists(),
                                reason="engine log not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:          # run the lifespan: model loaded once at startup
        yield c


def test_system_contract(client):
    s = client.get("/api/system").json()
    assert s["status"] == "online" and "SIMULATED" in s["disclosure"]
    t = s["thresholds"]
    assert 0 < t["warning_risk"] <= t["critical_risk"] <= 1 and t["warning_raw"] <= t["critical_raw"]
    assert s["model"]["parameters"] > 0


def test_timeline_and_snapshot(client):
    tl = client.get("/api/timeline").json()
    a = next(iter(tl["assets"]))
    n = len(tl["timestamps"])
    assert all(len(v["risk"]) == n for v in tl["assets"].values())
    i = tl["assets"][a]["index"][500]
    snap = client.get(f"/api/snapshot?index={i}").json()
    assert set(snap["assets"]) == set(tl["assets"])
    assert client.get("/api/snapshot?index=1").status_code == 404


def test_session_records_and_bounds(client):
    tl = client.get("/api/timeline").json()
    a = next(iter(tl["assets"]))
    start = tl["assets"][a]["index"][300]
    s = client.post("/api/sessions", json={"start_index": start}).json()
    r = client.post(f"/api/sessions/{s['id']}/step", json={"n": 3}).json()
    rec = r["records"][a][-1]
    for k in ("raw_score", "risk", "confidence", "status", "signals", "health", "timings_ms"):
        assert k in rec
    assert 0 <= rec["risk"] <= 1 and 0 <= rec["confidence"] <= 1 and 0 <= rec["health"]["score"] <= 100
    # live engine must agree with the engine log at the same index
    log = client.get(f"/api/log?asset={a}&start={rec['index']}&end={rec['index'] + 1}").json()[0]
    assert abs(log["raw_score"] - rec["raw_score"]) < 1e-5 and log["status"] == rec["status"]
    assert client.post("/api/sessions", json={"start_index": 5}).status_code == 400
    assert client.post("/api/sessions/nope/step", json={"n": 1}).status_code == 404


def test_incident_replay_lead_time_matches_offline(client):
    inc = client.get("/api/incidents").json()
    i = next(x for x in inc if x.get("engine_lead_time_min") and x.get("lead_time_min"))
    s = client.post(f"/api/incidents/{i['asset']}/{i['event_id']}/replay", json={"hours_before": 12}).json()
    lead = None
    while True:
        r = client.post(f"/api/sessions/{s['id']}/step", json={"n": 64}).json()
        for rec in r["records"][i["asset"]]:
            if rec["ground_truth"]["failure_event"]:
                lead = rec.get("lead_time_min")
        if r["done"]:
            break
    assert lead is not None and abs(lead - i["engine_lead_time_min"]) < 1e-6


def test_alert_ack_roundtrip(client):
    al = client.get("/api/alerts").json()
    aid = al[0]["id"]
    assert client.post(f"/api/alerts/{aid}/ack", json={"by": "test"}).json()["status"] == "acknowledged"
    assert next(a for a in client.get("/api/alerts").json() if a["id"] == aid)["status"] == "acknowledged"
    client.delete(f"/api/alerts/{aid}/ack")
    assert next(a for a in client.get("/api/alerts").json() if a["id"] == aid)["status"] != "acknowledged"


def test_prime_equals_full_warmup():
    from backend.intelligence import AssetStream, load_deployment
    from backend.sessions import Session
    from src.data.dataset import load_processed
    from src.utils.config import load_config
    cfg = load_config(); dep = load_deployment(cfg); dfs, _ = load_processed(cfg)
    start = 61000
    s = Session(dep, dfs, ["ETTm2"], start, start + 20)
    a = s.step(20)["records"]["ETTm2"]
    ref = AssetStream(dep, "ETTm2"); df = dfs["ETTm2"]
    for i in range(start - dep.art.fe.lookback - dep.art.seq_len + 1, start):
        ref.update(df.timestamp.iloc[i], df.iloc[i], bool(df.offline.iloc[i]), i, emit=False)
    b = [ref.update(df.timestamp.iloc[i], df.iloc[i], bool(df.offline.iloc[i]), i)[0] for i in range(start, start + 20)]
    assert np.allclose([x["raw_score"] for x in a], [x["raw_score"] for x in b], atol=1e-6)
    assert [x["status"] for x in a] == [x["status"] for x in b]
