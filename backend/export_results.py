"""Export the static "results-only" website (e.g. for Vercel) — aggregate results, never sensor rows.

The live views need the PyTorch/ONNX engine and the ETDataset sensor rows. ETDataset is CC BY-ND 4.0, and the
benchmark injects simulated faults into it, so those rows are not redistributed. The static site therefore serves
only the results pages (Model Lab, Failure Intelligence, Methodology). Every JSON file comes from the real FastAPI
app (TestClient), and each one is scanned so that nothing containing per-step sensor readings can be written.

    make site            # VITE_STATIC=1 build → site/, then this script → site/data/*.json
"""
from __future__ import annotations

import json
import shutil

from fastapi.testclient import TestClient

from backend.app import app
from src.utils.config import PROJECT_ROOT

SITE = PROJECT_ROOT / "site"
ENDPOINTS = {"system": "/api/system", "models": "/api/models", "failures": "/api/analytics/failures",
             "methodology": "/api/methodology", "incidents": "/api/incidents", "patterns": "/api/patterns"}
SENSORS = {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"}


def sensor_rows(o, path="$") -> list[str]:
    """Paths of objects that look like sensor readings: a dict keyed by ≥ 3 sensor channels."""
    hits = []
    if isinstance(o, dict):
        if len(SENSORS & set(o)) >= 3 and all(isinstance(o[k], (int, float)) or o[k] is None for k in SENSORS & set(o)):
            hits.append(path)
        for k, v in o.items():
            hits += sensor_rows(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hits += sensor_rows(v, f"{path}[{i}]")
    return hits


def main() -> None:
    if not (SITE / "index.html").exists():
        raise SystemExit("site/ not built: run `make site` (VITE_STATIC=1 frontend build first)")
    out = SITE / "data"
    out.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        for name, url in ENDPOINTS.items():
            r = c.get(url)
            r.raise_for_status()
            data = r.json()
            hits = sensor_rows(data)
            # per-channel aggregate tables (e.g. outlier counts per sensor) are fine; per-step readings are not
            if len(hits) > 2:
                raise SystemExit(f"{name}: refusing to export — looks like sensor rows at {hits[:3]} …")
            (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    shutil.copytree(PROJECT_ROOT / "docs" / "screenshots", SITE / "screenshots", dirs_exist_ok=True)
    (SITE / "vercel.json").write_text(json.dumps({
        "rewrites": [{"source": "/((?!data/|_app/|screenshots/).*)", "destination": "/index.html"}],
        "headers": [{"source": "/index.html", "headers": [{"key": "Cache-Control", "value": "no-cache"}]}]}, indent=2))
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(json.dumps({"files": sorted(p.name for p in out.iterdir()), "site_mb": round(total / 1e6, 2)}))


if __name__ == "__main__":
    main()
