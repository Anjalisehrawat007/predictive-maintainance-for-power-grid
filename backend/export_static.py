"""Export a static snapshot of the API for hosting without a Python backend (e.g. Vercel).

Every file is produced by calling the *real* FastAPI app (TestClient), so the static site shows
exactly what the live backend serves.  Incident replays are recorded by running the real streaming
engine for each held-out failure (3 h, 6 h and 12 h look-back).

    python -m backend.export_static            # → frontend/public/data/
"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from backend.app import app
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "frontend" / "public" / "data"
HOURS = (3, 6, 12)


def _round(o, nd=5):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def dump(name: str, obj) -> int:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(_round(obj), separators=(",", ":"))
    p.write_text(s)
    return len(s)


def main() -> None:
    t0 = time.time()
    c = TestClient(app)
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = {}
    system = c.get("/api/system").json()
    system["static_export"] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "note": "Outputs recorded from the real Python engine; no model runs in the browser."}
    sizes["system"] = dump("system.json", system)
    for name, path in [("assets", "/api/assets"), ("timeline", "/api/timeline"), ("incidents", "/api/incidents"),
                       ("alerts", "/api/alerts"), ("failures", "/api/analytics/failures"), ("models", "/api/models"),
                       ("methodology", "/api/methodology")]:
        sizes[name] = dump(f"{name}.json", c.get(path).json())
    tl = c.get("/api/timeline").json()
    for a, v in tl["assets"].items():
        lo, hi = v["index"][0], v["index"][-1] + 1
        rows = []
        for s in range(lo, hi, 3000):
            rows += c.get(f"/api/log?asset={a}&start={s}&end={min(hi, s + 3000)}").json()
        sizes[f"log/{a}"] = dump(f"log/{a}.json", rows)
    from backend.store import Store
    ev = Store().log_events
    sizes["log/events"] = dump("log/events.json", json.loads(ev.to_json(orient="records")))
    # recorded incident replays through the real engine
    incidents = c.get("/api/incidents").json()
    n = 0
    for inc in incidents:
        for h in HOURS:
            s = c.post(f"/api/incidents/{inc['asset']}/{inc['event_id']}/replay", json={"hours_before": h}).json()
            recs, evs = [], []
            while True:
                r = c.post(f"/api/sessions/{s['id']}/step", json={"n": 64}).json()
                recs += r["records"][inc["asset"]]
                evs += r["events"]
                if r["done"]:
                    break
            c.delete(f"/api/sessions/{s['id']}")
            dump(f"replays/{inc['asset']}_{inc['event_id']}_{h}.json", {"session": s, "records": recs, "events": evs})
            n += 1
    total = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(json.dumps({"files": sum(1 for _ in OUT.rglob('*.json')), "replays": n, "total_mb": round(total / 1e6, 1),
                      "largest": {k: round(v / 1e6, 2) for k, v in sorted(sizes.items(), key=lambda x: -x[1])[:5]},
                      "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
