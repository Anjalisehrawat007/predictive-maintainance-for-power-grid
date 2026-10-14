"""Load the trained model + preprocessing artefacts for inference."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import resolve


@dataclass
class Artifacts:
    model: object | None          # torch.nn.Module; None for the ONNX-only serving build
    meta: dict
    fe: object
    scaler: object
    scaler_columns: list[str]
    feature_names: list[str]

    @property
    def seq_len(self) -> int:
        return int(self.meta["seq_len"])

    def features(self, df: pd.DataFrame, asset: str) -> np.ndarray:
        """Raw sensor frame -> scaled model features (N, F). Causal by construction."""
        F = self.fe.transform(df, asset)[self.scaler_columns].to_numpy(dtype=np.float64)
        Z = self.scaler.transform(F, asset)
        if self.feature_names != self.scaler_columns:
            Z = Z[:, [self.scaler_columns.index(c) for c in self.feature_names]]
        return Z


def deployment_manifest(cfg: dict) -> dict:
    """The deployment manifest (models/phase2/deployment.json): the single source of truth for the served
    model's operating thresholds and calibrator. Empty dict if it does not exist (Phase-1 fallback)."""
    path = resolve(cfg, "models_dir") / "phase2" / "deployment.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_artifacts_lite(cfg: dict) -> Artifacts:
    """Metadata + fitted preprocessing only (no PyTorch): for the ONNX-only serving build."""
    mdir = resolve(cfg, "models_dir")
    meta = json.loads((mdir / "model_metadata.json").read_text())
    with open(mdir / "preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)
    name = meta["selected_model"]
    meta = meta | {"active_model": name, "active_threshold": meta["family_models"][name]["threshold"]}
    return Artifacts(None, meta, pre["fe"], pre["scaler"], pre["scaler_columns"], pre["feature_names"])


def load_artifacts(cfg: dict, model_file: str | None = None) -> Artifacts:
    import torch

    from src.models.recurrent import build_model
    mdir = resolve(cfg, "models_dir")
    meta = json.loads((mdir / "model_metadata.json").read_text())
    with open(mdir / "preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)
    name = model_file or meta["selected_model"]
    spec = meta["family_models"][name]
    if name == "ensemble":
        from src.models.recurrent import SeedEnsemble

        members = []
        for i in range(spec["n_members"]):
            m = build_model(spec["model_type"], len(pre["feature_names"]), spec["hidden_size"],
                            spec["num_layers"], meta["dropout"])
            m.load_state_dict(torch.load(mdir / f"ensemble_member{i}.pt", map_location="cpu"))
            members.append(m.eval())
        meta = meta | {"active_model": name, "active_threshold": spec["threshold"]}
        return Artifacts(SeedEnsemble(members).eval(), meta, pre["fe"], pre["scaler"], pre["scaler_columns"],
                         pre["feature_names"])
    model = build_model(spec["model_type"], len(pre["feature_names"]), spec["hidden_size"],
                        spec["num_layers"], meta["dropout"])
    if spec["quantized"]:
        from src.training.final import quantize_dynamic

        model = quantize_dynamic(model)
    # int8 checkpoints hold packed params that the weights-only loader rejects; they are
    # produced locally by src.training.final, never downloaded.
    model.load_state_dict(torch.load(mdir / f"{name}.pt", map_location="cpu",
                                     weights_only=not spec["quantized"]))
    model.eval()
    meta = meta | {"active_model": name, "active_threshold": spec["threshold"]}
    return Artifacts(model, meta, pre["fe"], pre["scaler"], pre["scaler_columns"], pre["feature_names"])
