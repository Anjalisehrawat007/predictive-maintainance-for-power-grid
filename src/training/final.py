"""Final stage: per-family best models, quantisation, multi-objective selection."""
from __future__ import annotations

import copy
import json
import pickle
import time

import pandas as pd
import torch

from src.evaluation.efficiency import torch_efficiency
from src.evaluation.metrics import threshold_for_precision
from src.training.experiment import _evaluate, _log_row, _row, get_prepared
from src.training.stages import SEEDS_FULL, _run, _variant, load_state
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("final")

FAMILY_FILES = {"lstm": "best_lstm", "gru": "best_gru", "attention": "best_attention_model",
                "lite": "best_lite_model"}


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """int8 dynamic quantisation of recurrent + linear layers (CPU inference)."""
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else engines[-1]
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {torch.nn.GRU, torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)


def _attention_family(cfg) -> str:
    p = resolve(cfg, "experiments_dir") / "stage_log.json"
    if p.exists():
        for e in json.loads(p.read_text()):
            if e["stage"] == "architectures":
                tbl = {r["candidate"]: r["score"] for r in e["table"]}
                return max(("lstm_attention", "gru_attention"), key=lambda k: tbl.get(k, -1))
    return "gru_attention"


def selection_score(df: pd.DataFrame, w: dict) -> pd.Series:
    """Weighted sum of min-max normalised VALIDATION metrics (higher = better)."""
    def norm(s, higher=True):
        s = s.astype(float)
        r = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return r if higher else 1 - r
    return (w["pr_auc"] * norm(df.val_pr_auc) + w["recall"] * norm(df.val_event_recall)
            + w["f1"] * norm(df.val_f1) + w["false_alarms"] * norm(df.val_false_alarms_per_day, False)
            + w["lead_time"] * norm(df.val_lead_time_median_min) + w["latency"] * norm(df.inference_latency_ms, False)
            + w["size"] * norm(df.model_size_kb, False))


def run_final(cfg: dict) -> dict:
    st = load_state(cfg)
    mdir = resolve(cfg, "models_dir")
    fams = {"lstm": _variant(st, model_type="lstm"), "gru": _variant(st, model_type="gru"),
            "attention": _variant(st, model_type=_attention_family(cfg)),
            "lite": _variant(st, model_type="gru_lite", hidden_size=min(32, st["hidden_size"]), num_layers=1)}
    cands = []
    for fam, fst in fams.items():
        best = None
        for seed in SEEDS_FULL:
            row, model, p = _run(cfg, f"final:{fam}", fst, seed, return_model=True)
            if best is None or row["val_pr_auc"] > best[0]["val_pr_auc"]:
                best = (row, model, p, seed)
        row, model, p, seed = best
        torch.save(model.state_dict(), mdir / f"{FAMILY_FILES[fam]}.pt")
        cands.append({"family": fam, "file": FAMILY_FILES[fam], "state": fst | {"seed": seed},
                      "row": row, "model": model, "quantized": False})
        # dynamic int8 quantisation of the same weights
        qm = quantize_dynamic(model)
        b = WindowBatcher(p.X, p.y, fst["seq_len"], torch.device("cpu"))
        pv, pt = predict_proba(qm, b, p.idx["val"]), predict_proba(qm, b, p.idx["test"])
        res, _ = _evaluate(cfg, p, pv, pt)
        eff = torch_efficiency(qm, fst["seq_len"], p.X.shape[1])
        eff["parameters"] = row["parameters"]
        qrow = _row(cfg, row["run_id"] + "-int8", f"final:{fam}", row["model"] + "_int8", "deep",
                    fst["feature_set"], fst["horizon"], fst["seq_len"], fst["loss"], res,
                    eff | {"train_time_s": row["train_time_s"]}, {"seed": seed, "notes": "dynamic int8"})
        _log_row(cfg, qrow, {"val": pv, "test": pt})
        torch.save(qm.state_dict(), mdir / f"{FAMILY_FILES[fam]}_int8.pt")
        cands.append({"family": fam, "file": FAMILY_FILES[fam] + "_int8", "state": fst | {"seed": seed},
                      "row": qrow, "model": qm, "quantized": True})
        log.info("%s fp32 val PR-AUC %.4f | int8 val PR-AUC %.4f (latency %.3f -> %.3f ms)", fam,
                 row["val_pr_auc"], qrow["val_pr_auc"], row["inference_latency_ms"], qrow["inference_latency_ms"])

    table = pd.DataFrame([c["row"] | {"family": c["family"], "file": c["file"], "quantized": c["quantized"]}
                          for c in cands])
    table["selection_score"] = selection_score(table, cfg["selection"]["weights"])
    table = table.sort_values("selection_score", ascending=False)
    table.to_csv(resolve(cfg, "experiments_dir") / "final_candidates.csv", index=False)
    win = next(c for c in cands if c["file"] == table.iloc[0]["file"])
    log.info("Selected final model: %s (score %.3f)", win["file"], table.iloc[0]["selection_score"])

    # thresholds (validation only)
    fst = win["state"]
    p = get_prepared(cfg, fst["feature_set"], fst["horizon"])
    b = WindowBatcher(p.X, p.y, fst["seq_len"], torch.device("cpu"))
    pv = predict_proba(win["model"], b, p.idx["val"])
    warn = float(win["row"]["threshold"])
    crit = max(warn, threshold_for_precision(p.y[p.idx["val"]], pv, cfg["alerts"]["critical_precision"]))
    meta = {
        "selected_model": win["file"], "family": win["family"], "quantized": win["quantized"],
        "model_type": fst["model_type"], "hidden_size": fst["hidden_size"], "num_layers": fst["num_layers"],
        "dropout": fst["dropout"], "seq_len": fst["seq_len"], "horizon": fst["horizon"],
        "feature_set": fst["feature_set"], "feature_names": p.feature_names, "n_features": len(p.feature_names),
        "loss": fst["loss"], "seed": fst["seed"], "warning_threshold": warn, "critical_threshold": crit,
        "threshold_objective": cfg["threshold"], "selection_weights": cfg["selection"]["weights"],
        "val_metrics": {k: win["row"][k] for k in win["row"] if k.startswith("val_")},
        "test_metrics": {k: win["row"][k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc",
                                                     "event_recall", "lead_time_median_min",
                                                     "false_alarms_per_day")},
        "efficiency": {k: win["row"][k] for k in ("parameters", "model_size_kb", "inference_latency_ms")},
        "family_models": {c["file"]: {"model_type": c["state"]["model_type"], "hidden_size": c["state"]["hidden_size"],
                                      "num_layers": c["state"]["num_layers"], "seed": c["state"]["seed"],
                                      "quantized": c["quantized"], "threshold": float(c["row"]["threshold"])}
                          for c in cands},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (mdir / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    with open(mdir / "preprocessor.pkl", "wb") as f:
        pickle.dump({"fe": p.fe, "scaler": p.scaler, "feature_names": p.feature_names,
                     "scaler_columns": p.extras["scaler_columns"]}, f)
    return meta
