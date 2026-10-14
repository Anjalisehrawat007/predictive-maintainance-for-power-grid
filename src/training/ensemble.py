"""Post-selection experiment: does a seed ensemble of the selected config help?

Trains ``n`` seeds of the selected configuration (deterministic seeds, so the
members equal the final-stage runs), evaluates the probability-averaged
ensemble with a threshold re-selected on VALIDATION, and re-runs the
multi-objective selection with the ensemble as an extra candidate.  The
ensemble replaces the selected model only if it wins that selection.
"""
from __future__ import annotations

import json

import pandas as pd
import torch

from src.evaluation.efficiency import torch_efficiency
from src.models.recurrent import SeedEnsemble
from src.training.experiment import _evaluate, _log_row, _row
from src.training.final import selection_score
from src.training.stages import SEEDS_FULL, _run, load_state
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("ensemble")


def run_ensemble(cfg: dict) -> dict:
    mdir, edir = resolve(cfg, "models_dir"), resolve(cfg, "experiments_dir")
    meta = json.loads((mdir / "model_metadata.json").read_text())
    st = load_state(cfg) | {"model_type": meta["model_type"], "hidden_size": meta["hidden_size"],
                            "num_layers": meta["num_layers"]}
    members, p = [], None
    for i, seed in enumerate(SEEDS_FULL):
        _, m, p = _run(cfg, "ensemble:member", st, seed, return_model=True)
        torch.save(m.state_dict(), mdir / f"ensemble_member{i}.pt")
        members.append(m.eval())
    ens = SeedEnsemble(members).eval()
    b = WindowBatcher(p.X, p.y, st["seq_len"], torch.device("cpu"))
    pv, pt = predict_proba(ens, b, p.idx["val"]), predict_proba(ens, b, p.idx["test"])
    res, _ = _evaluate(cfg, p, pv, pt)
    eff = torch_efficiency(ens, st["seq_len"], p.X.shape[1]) | {"train_time_s": float("nan")}
    row = _row(cfg, f"ensemble{len(members)}-{meta['model_type']}", "ensemble", f"{meta['model_type']}_x{len(members)}",
               "deep", st["feature_set"], st["horizon"], st["seq_len"], st["loss"], res, eff,
               {"seed": -1, "notes": f"probability average of seeds {SEEDS_FULL}"})
    _log_row(cfg, row, {"val": pv, "test": pt})

    fc = pd.read_csv(edir / "final_candidates.csv")
    fc = fc[fc.file != "ensemble"]
    fc = pd.concat([fc, pd.DataFrame([row | {"family": "ensemble", "file": "ensemble", "quantized": False}])],
                   ignore_index=True)
    fc["selection_score"] = selection_score(fc, cfg["selection"]["weights"])
    fc = fc.sort_values("selection_score", ascending=False)
    fc.to_csv(edir / "final_candidates.csv", index=False)
    log.info("Ensemble val PR-AUC %.4f vs selected %.4f | selection winner: %s", row["val_pr_auc"],
             meta["val_metrics"]["val_pr_auc"], fc.iloc[0]["file"])
    meta["family_models"]["ensemble"] = {"model_type": meta["model_type"], "hidden_size": meta["hidden_size"],
                                         "num_layers": meta["num_layers"], "seed": -1, "quantized": False,
                                         "threshold": float(row["threshold"]), "n_members": len(members)}
    if fc.iloc[0]["file"] == "ensemble":
        from src.evaluation.metrics import threshold_for_precision

        meta.update(selected_model="ensemble", family="ensemble",
                    warning_threshold=float(row["threshold"]),
                    critical_threshold=max(float(row["threshold"]), threshold_for_precision(
                        p.y[p.idx["val"]], pv, cfg["alerts"]["critical_precision"])),
                    val_metrics={k: row[k] for k in row if k.startswith("val_")},
                    efficiency={k: row[k] for k in ("parameters", "model_size_kb", "inference_latency_ms")})
        log.info("Ensemble adopted as the selected model.")
    else:
        log.info("Ensemble NOT adopted (kept %s).", meta["selected_model"])
    (mdir / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    return row
