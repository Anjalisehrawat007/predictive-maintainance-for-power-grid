"""Export the deployed model and its 3 seed-ensemble members to ONNX (dynamic batch) and verify each against
PyTorch. The ONNX files let the engine run without PyTorch (SGPM_ENGINE=onnx, e.g. the serverless deployment).

    python -m src.streaming.export_onnx
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.models.recurrent import build_model
from src.streaming.artifacts import load_artifacts
from src.utils.config import load_config, resolve


def _export(model, L: int, F: int, path) -> float:
    import onnxruntime as ort
    model.eval()
    x = torch.randn(1, L, F)
    torch.onnx.export(model, x, str(path), input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
    xb = np.random.randn(64, L, F).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xb)).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": xb})[0]
    err = float(np.abs(ref - got).max())
    if err > 1e-4:
        raise SystemExit(f"{path.name}: ONNX output differs from PyTorch (max |Δlogit| {err:.2e})")
    return err


def main() -> None:
    cfg = load_config()
    art = load_artifacts(cfg)
    mdir = resolve(cfg, "models_dir")
    L, F, meta = art.seq_len, len(art.feature_names), art.meta
    out = {f"{meta['active_model']}.onnx": _export(art.model, L, F, mdir / f"{meta['active_model']}.onnx")}
    for i in range(3):
        f = mdir / f"ensemble_member{i}.pt"
        if f.exists():
            m = build_model(meta["model_type"], F, meta["hidden_size"], meta["num_layers"], meta["dropout"])
            m.load_state_dict(torch.load(f, map_location="cpu"))
            out[f"ensemble_member{i}.onnx"] = _export(m, L, F, mdir / f"ensemble_member{i}.onnx")
    (mdir / "onnx_manifest.json").write_text(json.dumps({
        "model": meta["active_model"], "parameters": int(sum(p.numel() for p in art.model.parameters())),
        "max_abs_logit_diff_vs_pytorch": out}, indent=1))
    print({k: f"max |Δlogit| {v:.1e}" for k, v in out.items()})


if __name__ == "__main__":
    main()
