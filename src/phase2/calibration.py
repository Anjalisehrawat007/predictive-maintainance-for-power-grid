"""Probability calibration + model-confidence layer (Phase 2).

* Calibrators are fitted on VALIDATION data only (never test).
* Method choice uses 2-fold cross-fitting *inside* validation (fit on one half of the
  validation period, score ECE/Brier on the other half, and vice versa).
* Calibration is monotone, so alarms/thresholds are unaffected; it only changes the
  number shown to operators ("calibrated risk"), which is kept distinct from the
  "raw model score".
* Confidence = agreement of the 3 seed-ensemble members (1 − 2·σ of their probabilities).
  It is a statement about the MODEL, not about the physical asset.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.models.recurrent import build_model
from src.phase2 import resim
from src.phase2.analysis import ece
from src.phase2.common import P2_MODELS, P2_REPORTS, write_md
from src.training.trainer import WindowBatcher, predict_proba
from src.utils.plotting import INK2, SERIES, apply_style, plt, save

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Monotone map raw score → calibrated probability."""

    def __init__(self, method: str = "identity"):
        self.method, self.T, self.b, self.iso = method, 1.0, 0.0, None

    def fit(self, p, y):
        if self.method == "temperature":
            z = logit(p)

            def nll(T):
                q = 1 / (1 + np.exp(-z / T))
                return -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
            self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e4).fit(logit(p)[:, None], y)
            self.T, self.b = float(1 / lr.coef_[0, 0]), float(lr.intercept_[0])
        elif self.method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if self.method in ("temperature", "platt"):
            return 1 / (1 + np.exp(-(logit(p) / self.T + self.b)))
        if self.method == "isotonic":
            return self.iso.predict(p)
        return p

    def to_dict(self):
        d = {"method": self.method, "T": self.T, "b": self.b}
        if self.iso is not None:
            d["iso_x"] = self.iso.X_thresholds_.tolist()
            d["iso_y"] = self.iso.y_thresholds_.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        c = cls(d["method"]); c.T, c.b = d["T"], d.get("b", 0.0)
        if d["method"] == "isotonic":
            c.iso = IsotonicRegression(out_of_bounds="clip").fit(d["iso_x"], d["iso_y"])
        return c


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def scores(y, p):
    return {"brier": brier(y, p), "ece": ece(y, p), "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}


def choose_and_fit(pv, yv, asset_v, t_v) -> tuple[Calibrator, pd.DataFrame]:
    """2-fold cross-fitting inside validation (split by time: first half / second half)."""
    order = np.argsort(t_v)
    half = np.zeros(len(pv), bool)
    half[order[: len(order) // 2]] = True
    rows = []
    for m in ("identity", "temperature", "platt", "isotonic"):
        e, b = [], []
        for fit_mask in (half, ~half):
            c = Calibrator(m).fit(pv[fit_mask], yv[fit_mask])
            q = c(pv[~fit_mask])
            e.append(ece(yv[~fit_mask], q)); b.append(brier(yv[~fit_mask], q))
        rows.append({"method": m, "cv_ece": float(np.mean(e)), "cv_brier": float(np.mean(b))})
    cv = pd.DataFrame(rows).sort_values("cv_brier")
    best = cv.iloc[0]["method"]
    return Calibrator(best).fit(pv, yv), cv


def load_members(cfg, p, meta, root="models"):
    from src.utils.config import PROJECT_ROOT
    ms = []
    for i in range(3):
        m = build_model(meta["model_type"], p.X.shape[1], meta["hidden_size"], meta["num_layers"], meta["dropout"])
        m.load_state_dict(torch.load(PROJECT_ROOT / root / f"ensemble_member{i}.pt", map_location="cpu"))
        ms.append(m.eval())
    return ms


@torch.no_grad()
def mc_dropout(model, W, n=20):
    model.train()                     # enables dropout (GRU has 1 layer → dropout acts on the head)
    out = np.stack([torch.sigmoid(model(W)).numpy() for _ in range(n)])
    model.eval()
    return out.std(0)


# Model-confidence bins (lower bounds). Single definition used by evaluation, the API and the UI.
CONFIDENCE_BINS = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.0}


def confidence_bin(conf: np.ndarray) -> np.ndarray:
    """Label each confidence value HIGH / MEDIUM / LOW using CONFIDENCE_BINS."""
    c = np.asarray(conf, dtype=float)
    return np.where(c >= CONFIDENCE_BINS["HIGH"], "HIGH", np.where(c >= CONFIDENCE_BINS["MEDIUM"], "MEDIUM", "LOW"))


def confidence_from_members(member_probs: np.ndarray) -> np.ndarray:
    """MODEL CONFIDENCE (seed agreement) = 1 − 2·σ(member probabilities), in [0, 1]."""
    return np.clip(1 - 2 * member_probs.std(0), 0, 1)


def uncertainty_eval(y, p, conf, thr) -> dict:
    """Is low confidence predictive of errors?  (alarms and all samples)"""
    alarm = p >= thr
    out = {}
    err = (alarm != (y == 1))
    out["auroc_conf_vs_error_all"] = float(roc_auc_score(err, 1 - conf)) if 0 < err.sum() < len(err) else np.nan
    if alarm.sum() > 10:
        fp = (y[alarm] == 0)
        out["auroc_conf_vs_false_alarm"] = float(roc_auc_score(fp, 1 - conf[alarm])) if 0 < fp.sum() < alarm.sum() else np.nan
        hi = conf[alarm] >= np.median(conf[alarm])
        out["alarm_precision_high_conf"] = float((y[alarm][hi] == 1).mean())
        out["alarm_precision_low_conf"] = float((y[alarm][~hi] == 1).mean())
    return out


def run(cfg, art, p, events, thr, tag="phase1_selected", n_real=5) -> dict:
    from src.phase2.experiments import RS_SEED0
    apply_style()
    vb = WindowBatcher(p.X, p.y, art.seq_len, torch.device("cpu"))
    pv = predict_proba(art.model, vb, p.idx["val"])
    yv = p.y[p.idx["val"]]
    cal, cv = choose_and_fit(pv, yv, p.asset[p.idx["val"]], p.timestamps[p.idx["val"]])
    members = load_members(cfg, p, art.meta)
    res = {"cv_selection": cv.to_dict(orient="records"), "chosen": cal.method, "calibrator": cal.to_dict()}
    evals = []
    rel = {}
    unc_rows = []
    sets = [("test (original)", p, p.idx["test"])]
    for k in range(n_real):
        dfs, ev = resim.simulate(cfg, RS_SEED0 + 10 * k)
        q = resim.prepared_from(cfg, art, dfs, art.meta["horizon"])
        sets.append((f"test (re-sim {k})", q, q.idx["test"]))
        sets.append((f"val (re-sim {k})", q, q.idx["val"]))
    for name, q, rows in sets:
        b = WindowBatcher(q.X, q.y, art.seq_len, torch.device("cpu"))
        raw = predict_proba(art.model, b, rows)
        y = q.y[rows]
        for mname in ("identity", "temperature", "platt", "isotonic"):
            c = Calibrator(mname).fit(pv, yv)
            evals.append({"set": name, "method": mname} | scores(y, c(raw)))
        if name == "test (original)":
            rel = {"raw": (y, raw), "calibrated": (y, cal(raw))}
            mp = np.stack([predict_proba(m, b, rows) for m in members])
            conf = confidence_from_members(mp)
            sel = np.random.default_rng(0).choice(len(rows), min(6000, len(rows)), replace=False)
            W = torch.from_numpy(q.X[rows[sel][:, None] + np.arange(-art.seq_len + 1, 1)])
            mc = mc_dropout(art.model, W)
            unc_rows.append({"method": "seed ensemble (3 members)"} | uncertainty_eval(y, raw, conf, thr))
            unc_rows.append({"method": "MC dropout (20 passes)"} |
                            uncertainty_eval(y[sel], raw[sel], np.clip(1 - 2 * mc, 0, 1), thr))
            res["confidence_distribution"] = {"median": float(np.median(conf)), "p05": float(np.quantile(conf, .05)),
                                              "median_in_alarms": float(np.median(conf[raw >= thr]))}
    ev_df = pd.DataFrame(evals)
    agg = ev_df.assign(group=ev_df.set.str.extract(r"^(\w+ \(\w+)")[0]).groupby(["group", "method"])[["brier", "ece"]].mean().round(4)
    res["evaluation"] = ev_df.to_dict(orient="records")
    unc = pd.DataFrame(unc_rows)
    res["uncertainty"] = unc.to_dict(orient="records")
    # reliability plot
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], ls=":", color=INK2)
    for k, (lab, (y, pp)) in enumerate(rel.items()):
        qs = np.unique(np.quantile(pp, np.linspace(0, 1, 16)))
        idx = np.clip(np.searchsorted(qs, pp, side="right") - 1, 0, len(qs) - 2)
        xs = [pp[idx == i].mean() for i in range(len(qs) - 1) if (idx == i).any()]
        ys = [y[idx == i].mean() for i in range(len(qs) - 1) if (idx == i).any()]
        ax.plot(xs, ys, marker="o", ms=4, color=SERIES[k], label=f"{lab} (ECE {ece(y, pp):.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed failure-within-2h rate")
    ax.set_title(f"Reliability on test — {cal.method} fitted on validation"); ax.legend()
    save(fig, P2_REPORTS / "figures" / "calibration_reliability.png")
    with open(P2_MODELS / f"calibrator_{tag}.json", "w") as f:
        json.dump(cal.to_dict(), f)
    L = ["# Calibration & model confidence (Phase 2)\n",
         f"Model: `{tag}`. Calibrators are fitted on the **validation** period only; the method is chosen by "
         "2-fold time-split cross-fitting *within* validation (Brier score). Test and re-simulated sets are used only to "
         "report.\n",
         "## Method selection (cross-fitted inside validation)\n", cv.round(4).to_markdown(index=False), "",
         f"**Chosen: `{cal.method}`** (T = {cal.T:.3f}, b = {cal.b:.3f}).\n",
         "## Out-of-sample calibration (mean over sets)\n", agg.to_markdown(), "",
         "Lower is better. `identity` = raw model score. Weighted-BCE training inflates raw scores "
         "(pos_weight ≈ 7.9), which is what calibration corrects.\n",
         "![](figures/calibration_reliability.png)\n",
         "## Model confidence (uncertainty) — does it flag mistakes?\n", unc.round(3).to_markdown(index=False), "",
         "`auroc_conf_vs_false_alarm` > 0.5 means lower confidence is associated with alarms that turn out false. "
         "Confidence is defined as **1 − 2·σ** of the three seed-ensemble members' probabilities "
         "(*model confidence*, not physical certainty).\n",
         f"Confidence distribution on test: {res.get('confidence_distribution')}\n"]
    write_md(P2_REPORTS / "calibration_uncertainty.md", "\n".join(L))
    (P2_REPORTS / "calibration_uncertainty.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from src.data.dataset import load_processed
    from src.data.pipeline import prepare
    from src.streaming.artifacts import load_artifacts
    from src.utils.config import load_config
    cfg = load_config()
    art = load_artifacts(cfg)
    p = prepare(cfg, art.meta["feature_set"], art.meta["horizon"])
    _, events = load_processed(cfg)
    print(json.dumps({k: v for k, v in run(cfg, art, p, events, art.meta["active_threshold"]).items()
                      if k in ("chosen", "uncertainty", "confidence_distribution")}, indent=1, default=float))
