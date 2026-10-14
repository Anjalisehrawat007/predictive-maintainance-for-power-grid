"""Publish the live engine to Hugging Face (needs `hf auth login` with a WRITE token first).

1. PRIVATE model repo  <user>/smartgrid-ai-models  ← trained weights (never public; see DATASET.md)
2. PUBLIC Docker Space <user>/smartgrid-ai-engine  ← code + committed metadata only (no dataset rows, no weights)

The Space downloads the weights at start-up. It needs ONE secret that you add yourself in the Space settings:
HF_TOKEN = a fine-grained READ token for the model repo.

    python deploy/publish_engine.py [--cors https://your-site.vercel.app]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ["best_attention_model.pt", "best_attention_model.onnx", "preprocessor.pkl",
           "ensemble_member0.pt", "ensemble_member1.pt", "ensemble_member2.pt"]
# what the engine needs at runtime (all tracked by git; no dataset rows, no weights)
INCLUDE = ("backend/", "src/", "configs/", "data/splits/", "data/processed/events.csv",
           "data/processed/dataset_card.json", "models/model_metadata.json", "models/phase2/", "experiments/",
           "reports/", "requirements.txt", "LICENSE", "DATASET.md")
EXCLUDE_SUFFIX = (".png", ".log", ".md")


def stage(tmp: Path) -> int:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    n = 0
    for f in files:
        if not f.startswith(INCLUDE) or (f.endswith(EXCLUDE_SUFFIX) and f not in ("DATASET.md",)):
            continue
        if f.startswith("experiments/") and f.count("/") > 1:     # only top-level experiment tables
            continue
        dst = tmp / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
        n += 1
    shutil.copy2(ROOT / "deploy/space/Dockerfile", tmp / "Dockerfile")
    shutil.copy2(ROOT / "deploy/space/README.md", tmp / "README.md")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smartgrid-ai-models")
    ap.add_argument("--space", default="smartgrid-ai-engine")
    ap.add_argument("--cors", default="https://predictive-maintenance-power-grid.vercel.app")
    a = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    mrepo, srepo = f"{user}/{a.models}", f"{user}/{a.space}"

    missing = [w for w in WEIGHTS if not (ROOT / "models" / w).exists()]
    if missing:
        raise SystemExit(f"missing weights {missing}: run `make data` first")
    api.create_repo(mrepo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(mrepo, repo_type="model").private:
        raise SystemExit(f"{mrepo} exists and is PUBLIC — refusing to upload weights there")
    for w in WEIGHTS:
        api.upload_file(path_or_fileobj=str(ROOT / "models" / w), path_in_repo=w, repo_id=mrepo, repo_type="model")
    print(f"weights → private model repo {mrepo}")

    api.create_repo(srepo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_variable(srepo, "SGPM_MODEL_REPO", mrepo)
    api.add_space_variable(srepo, "SGPM_CORS_ORIGINS", f"{a.cors},http://localhost:5173")
    with tempfile.TemporaryDirectory() as t:
        n = stage(Path(t))
        api.upload_folder(folder_path=t, repo_id=srepo, repo_type="space", commit_message="Deploy SmartGrid AI engine")
    host = f"https://{srepo.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print(f"{n} files → public Space {srepo}\nengine URL: {host}\n"
          f"NEXT: add the Space secret HF_TOKEN (fine-grained READ token for {mrepo}) at "
          f"https://huggingface.co/spaces/{srepo}/settings")


if __name__ == "__main__":
    main()
