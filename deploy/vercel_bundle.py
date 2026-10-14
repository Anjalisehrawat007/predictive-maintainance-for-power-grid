"""Assemble the free Vercel deployment: static React app + ONNX-only FastAPI engine as a Python serverless function.

    python deploy/vercel_bundle.py        # → .deploy-vercel/   then:  cd .deploy-vercel && vercel deploy --prod

* public/            the frontend built with VITE_PUBLIC=1 (same-origin API) — the only statically served files
* api/index.py       the FastAPI app (SGPM_PUBLIC=1, SGPM_ENGINE=onnx): no PyTorch / scikit-learn / SciPy, no dataset rows
* engine files       code + committed metadata + aggregate JSON, plus the ONNX weights and fitted preprocessing, which
                     are bundled ONLY inside the function (not served as files; Vercel source protection keeps the
                     deployment source private). They are not in the Git repository (see DATASET.md).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".deploy-vercel"
ENGINE_DIRS = ("backend/", "src/", "configs/", "data/splits/", "data/processed/events.csv", "data/processed/dataset_card.json",
               "models/model_metadata.json", "models/phase2/deployment.json", "models/phase2/calibrator_final.json",
               "models/phase2/calibrator_phase1_selected.json", "models/onnx_manifest.json", "experiments/",
               "reports/", "LICENSE", "DATASET.md")
WEIGHTS = ["best_attention_model.onnx", "ensemble_member0.onnx", "ensemble_member1.onnx", "ensemble_member2.onnx",
           "preprocessor.pkl"]
# pinned to the versions in requirements.txt; nothing heavier is needed by the ONNX engine
REQUIREMENTS = ["fastapi==0.142.2", "pydantic==2.13.5", "numpy==2.4.6", "pandas==3.0.6", "onnxruntime==1.30.0",
                "pyyaml==6.0.3"]
API_INDEX = '''"""Vercel serverless entry point: the SmartGrid AI engine in public, ONNX-only mode."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SGPM_PUBLIC", "1")       # never serve dataset rows
os.environ.setdefault("SGPM_ENGINE", "onnx")    # no PyTorch in the function
# the web app is same-origin; the author's portfolio embeds a live demo that calls the engine cross-origin
os.environ.setdefault("SGPM_CORS_ORIGINS", "https://anjali-portfolio-hazel.vercel.app,http://localhost:3000")

from backend.app import app  # noqa: E402,F401
'''
VERCEL_JSON = {
    "outputDirectory": "public",
    "buildCommand": "",
    "functions": {"api/index.py": {"maxDuration": 60, "memory": 1024,
                                   "includeFiles": "{backend,src,configs,data,models,reports,experiments}/**"}},
    "rewrites": [{"source": "/api/(.*)", "destination": "/api/index"},
                 {"source": "/((?!api/|_app/).*)", "destination": "/index.html"}],
    "headers": [{"source": "/index.html", "headers": [{"key": "Cache-Control", "value": "no-cache"}]}],
}


def main() -> None:
    missing = [w for w in WEIGHTS if not (ROOT / "models" / w).exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run `make data` then `python -m src.streaming.export_onnx`")
    if OUT.exists():
        keep = OUT / ".vercel"
        tmp = ROOT / ".vercel-link-tmp"
        if keep.exists():
            shutil.move(str(keep), tmp)
        shutil.rmtree(OUT)
        OUT.mkdir()
        if tmp.exists():
            shutil.move(str(tmp), OUT / ".vercel")
    else:
        OUT.mkdir()
    # 1. frontend (same-origin API)
    env = os.environ | {"VITE_PUBLIC": "1", "VITE_API_BASE": ""}
    subprocess.run(["npx", "vite", "build", "--outDir", str(OUT / "public"), "--emptyOutDir"], cwd=ROOT / "frontend",
                   env=env, check=True, capture_output=True)
    # 2. engine files (tracked by git) + weights inside the function only
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    n = 0
    for f in files:
        if f.startswith(ENGINE_DIRS) and not f.endswith((".png", ".log")) and not (f.startswith("experiments/") and f.count("/") > 1):
            dst = OUT / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dst)
            n += 1
    for w in WEIGHTS:
        shutil.copy2(ROOT / "models" / w, OUT / "models" / w)
    (OUT / "api").mkdir(exist_ok=True)
    (OUT / "api" / "index.py").write_text(API_INDEX)
    (OUT / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    (OUT / "vercel.json").write_text(json.dumps(VERCEL_JSON, indent=2))
    static = [p for p in (OUT / "public").rglob("*") if p.is_file()]
    bad = [p for p in static if p.suffix in (".onnx", ".pkl", ".pt", ".parquet")]
    if bad:
        raise SystemExit(f"refusing: model/data files in public/: {bad}")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and ".vercel" not in p.parts) / 1e6
    print(json.dumps({"engine_files": n, "weights_in_function": WEIGHTS, "static_files": len(static), "bundle_mb": round(size, 2)}))


if __name__ == "__main__":
    main()
