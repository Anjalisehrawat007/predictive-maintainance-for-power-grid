"""Scenario Lab, visitor uploads and the public (no-dataset-rows) deployment mode."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.utils.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "best_attention_model.pt").exists(),
                                reason="trained model not present (run `make data`)")
SENS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:
        yield c


def _run(client, sid, n=64):
    recs, events = [], []
    while True:
        r = client.post(f"/api/sessions/{sid}/step", json={"n": n}).json()
        recs += next(iter(r["records"].values()))
        events += r["events"]
        if r["done"]:
            return recs, events


def test_scenario_injected_fault_is_streamed_through_the_engine(client):
    s = client.post("/api/scenarios", json={"asset": "ETTm2", "month": 4, "fault": "thermal_runaway",
                                            "severity": 1.5, "precursor_hours": 8, "seed": 3}).json()
    assert s["mode"] == "scenario" and s["incident"]["failure_time"]
    recs, events = _run(client, s["id"])
    fail = [r for r in recs if r["ground_truth"]["failure_event"]]
    assert len(fail) == 1 and fail[0]["timestamp"] == s["incident"]["failure_time"]
    assert any(e["kind"] in ("lead_time", "missed") for e in events)   # outcome is measured, not scripted
    assert max(r["risk"] for r in recs if r["risk"] is not None) > 0


def test_scenario_is_deterministic_and_fault_free_option(client):
    body = {"asset": "ETTm1", "month": 1, "fault": "none", "seed": 7}
    a, b = client.post("/api/scenarios", json=body).json(), client.post("/api/scenarios", json=body).json()
    ra, _ = _run(client, a["id"])
    rb, _ = _run(client, b["id"])
    assert [r["raw_score"] for r in ra] == [r["raw_score"] for r in rb]
    assert not any(r["ground_truth"]["failure_event"] for r in ra)


def test_scenario_input_validation(client):
    assert client.post("/api/scenarios", json={"fault": "meltdown"}).status_code == 422
    assert client.post("/api/scenarios", json={"severity": 9}).status_code == 422
    assert client.post("/api/scenarios", json={"asset": "nope"}).status_code == 404


def test_upload_streams_visitor_rows_and_validates(client):
    from src.scenario.baseline import generate
    df = generate("ETTm1", "2018-05-01", 250, seed=1)
    rows = [{"timestamp": str(t), **{c: float(v[c]) for c in SENS}} for t, (_, v) in zip(df.timestamp, df.iterrows())]
    s = client.post("/api/uploads", json={"asset": "ETTm1", "rows": rows}).json()
    assert s["mode"] == "upload"
    recs, _ = _run(client, s["id"])
    assert len(recs) == 250 - s["start"] and all(r["risk"] is not None for r in recs)
    assert client.post("/api/uploads", json={"asset": "ETTm1", "rows": rows[:100]}).status_code == 422   # too short
    gap = rows[:120] + rows[121:]
    assert client.post("/api/uploads", json={"asset": "ETTm1", "rows": gap}).status_code == 422          # not 15-min


PUBLIC_SCRIPT = r"""
import json
from fastapi.testclient import TestClient
from backend.app import app
from backend.app import STORE
out = {}
with TestClient(app) as c:
    out["ready"] = c.get("/api/ready").status_code
    out["system_public"] = c.get("/api/system").json()["public"]
    for u in ["/api/timeline", "/api/log?asset=ETTm1&start=60000&end=60010", "/api/snapshot?index=60000"]:
        out[u] = c.get(u).status_code
    out["sessions"] = c.post("/api/sessions", json={}).status_code
    out["replay"] = c.post("/api/incidents/ETTm1/109/replay", json={}).status_code
    out["scenario"] = c.post("/api/scenarios", json={"fault": "load_oscillation"}).status_code
    out["failures"] = c.get("/api/analytics/failures").status_code
    out["incidents_with_lead"] = sum(i.get("engine_lead_time_min") is not None for i in c.get("/api/incidents").json())
    out["dfs_loaded"] = len(STORE.dfs)
print(json.dumps(out))
"""


def test_public_mode_serves_no_dataset_rows():
    r = subprocess.run([sys.executable, "-c", PUBLIC_SCRIPT], capture_output=True, text=True, cwd=PROJECT_ROOT,
                       env=os.environ | {"SGPM_PUBLIC": "1"}, timeout=300)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ready"] == 200 and out["system_public"] is True and out["dfs_loaded"] == 0
    assert all(out[u] == 404 for u in ["/api/timeline", "/api/log?asset=ETTm1&start=60000&end=60010",
                                       "/api/snapshot?index=60000"])
    assert out["sessions"] == 404 and out["replay"] == 404
    assert out["scenario"] == 200 and out["failures"] == 200 and out["incidents_with_lead"] == 25


def test_sample_csv_round_trips_into_upload(client):
    import io
    r = client.get("/api/samples/synthetic.csv?asset=ETTm2&fault=load_oscillation")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.StringIO(r.text))
    assert list(df.columns) == ["timestamp", *SENS]
    rows = [{"timestamp": t, **{c: float(v[c]) for c in SENS}} for t, (_, v) in zip(df.timestamp, df.iterrows())]
    assert client.post("/api/uploads", json={"asset": "ETTm2", "rows": rows}).status_code == 200


def test_stateless_runs_equal_stepped_sessions(client):
    body = {"asset": "ETTm1", "month": 10, "fault": "intermittent_arcing", "severity": 1.0, "precursor_hours": 6, "seed": 5}
    run = client.post("/api/scenarios/run", json=body).json()
    stepped, _ = _run(client, client.post("/api/scenarios", json=body).json()["id"])
    assert [r["raw_score"] for r in run["records"]] == [r["raw_score"] for r in stepped]
    assert [r["status"] for r in run["records"]] == [r["status"] for r in stepped]
    assert run["compute_ms"] > 0 and run["attribution_method"] in ("integrated gradients", "occlusion")
    assert client.post("/api/scenarios/run", json=body | {"precursor_hours": 40}).status_code == 422
