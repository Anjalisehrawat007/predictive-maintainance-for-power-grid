"""Small, pruned Optuna search on the VALIDATION set (test data never touched)."""
from __future__ import annotations

import copy

import optuna

from src.training.experiment import run_dl
from src.utils.config import resolve
from src.utils.logging import get_logger

log = get_logger("hpo")


def run_hpo(cfg: dict, state: dict) -> dict:
    h = cfg["hpo"]
    storage = f"sqlite:///{resolve(cfg, 'experiments_dir') / 'optuna.db'}"

    def objective(trial: optuna.Trial) -> float:
        st = copy.deepcopy(state)
        st["hidden_size"] = trial.suggest_categorical("hidden_size", [16, 32, 64, 96])
        st["num_layers"] = trial.suggest_int("num_layers", 1, 2)
        st["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        st["lr"] = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
        st["train_overrides"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        st["train_overrides"]["input_noise"] = trial.suggest_categorical("input_noise", [0.0, 0.05, 0.1, 0.2])
        trial.set_user_attr("state", st)
        row = run_dl(cfg, "hpo", model_type=st["model_type"], feature_set=st["feature_set"],
                     horizon=st["horizon"], seq_len=st["seq_len"], loss=st["loss"],
                     hidden_size=st["hidden_size"], num_layers=st["num_layers"], dropout=st["dropout"],
                     lr=st["lr"], negative_ratio=st["negative_ratio"], seed=cfg["seed"],
                     max_epochs=h["max_epochs"], train_overrides=st["train_overrides"], trial=trial)
        return row["val_pr_auc"]

    study = optuna.create_study(
        study_name=f"sgpm_{state['model_type']}_{state['feature_set']}_H{state['horizon']}",
        direction="maximize", storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    remaining = max(0, h["n_trials"] - len(study.trials))
    study.optimize(objective, n_trials=remaining, timeout=h["timeout_minutes"] * 60)
    study.trials_dataframe().to_csv(resolve(cfg, "experiments_dir") / "optuna_trials.csv", index=False)
    log.info("HPO best val PR-AUC %.4f with %s", study.best_value, study.best_params)
    return study.best_trial.user_attrs["state"]
