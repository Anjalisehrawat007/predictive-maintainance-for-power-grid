"""Fetch the deployed model artefacts at container start (public deployment).

Trained weights are not published in the Git repository (see DATASET.md). The hosted engine downloads them from
a PRIVATE Hugging Face model repository at startup, so they never become part of a public repo or image layer.

    SGPM_MODEL_REPO=<user>/smartgrid-ai-models HF_TOKEN=<read token> python -m backend.fetch_models
"""
from __future__ import annotations

import os

from src.utils.config import PROJECT_ROOT

FILES = ["best_attention_model.pt", "best_attention_model.onnx", "preprocessor.pkl",
         "ensemble_member0.pt", "ensemble_member1.pt", "ensemble_member2.pt"]


def main() -> None:
    mdir = PROJECT_ROOT / "models"
    missing = [f for f in FILES if not (mdir / f).exists()]
    if not missing:
        print("model artefacts present")
        return
    repo = os.environ.get("SGPM_MODEL_REPO")
    if not repo:
        raise SystemExit(f"missing {missing} and SGPM_MODEL_REPO is not set")
    from huggingface_hub import hf_hub_download
    for f in missing:
        hf_hub_download(repo_id=repo, filename=f, local_dir=mdir, token=os.environ.get("HF_TOKEN"))
    print(f"downloaded {len(missing)} artefacts from {repo}")


if __name__ == "__main__":
    main()
