"""Validate the Scenario Lab: how does the deployed model behave on SYNTHETIC baselines?

* false alarms per day on fault-free synthetic streams (vs 0.25/day on the real held-out period);
* detection rate and lead time of injected faults per type and severity.

Synthetic results are illustrative (the model was trained and evaluated on real data) and are reported next to
the scenario in the UI.     python -m src.scenario.validate  → reports/final/scenario_validation.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.scenario.scenario import build
from src.streaming.alerts import AlertPolicy
from src.streaming.artifacts import load_artifacts
from src.streaming.inference import min_rows, predict_frame
from src.utils.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "reports" / "final" / "scenario_validation.json"
MONTHS, SEEDS = (1, 4, 7, 10), range(5)


def alarms(raw: np.ndarray, pol: AlertPolicy) -> np.ndarray:
    p = AlertPolicy(pol.warning_threshold, pol.critical_threshold, pol.persistence_k)
    return np.array([p.update(float(x)) != "NORMAL" if np.isfinite(x) else False for x in raw])


def lead_minutes(al: np.ndarray, tf: int, gap: int) -> float | None:
    """Same definition as the benchmark: an alarm episode (gaps ≤ gap merged) still active at the trip."""
    idx = np.flatnonzero(al[:tf])
    if not len(idx) or tf - 1 - idx[-1] > gap:
        return None
    start = idx[-1]
    for i in idx[::-1][1:]:
        if start - i - 1 > gap:
            break
        start = i
    return float((tf - start) * 15)


def main() -> dict:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    art = load_artifacts(cfg)
    pol = AlertPolicy.from_config(cfg, art.meta)
    gap = cfg["threshold"]["merge_gap_steps"]
    warm = min_rows(art)
    out = {"about": __doc__.strip().split("\n\n")[0], "threshold_raw": pol.warning_threshold, "false_alarms": [], "detection": []}
    for asset in ("ETTm1", "ETTm2"):
        for m in MONTHS:
            fa = []
            for s in SEEDS:
                df, meta = build(asset, m, "none", 1.0, 8, 1000 + s, warm, lead_hours=7 * 24)
                al = alarms(predict_frame(art, df, asset), pol)[warm:]
                eps = int(al[0]) + int((al[1:] & ~al[:-1]).sum())
                fa.append(eps / (len(al) / 96))
            out["false_alarms"].append({"asset": asset, "month": m, "false_alarms_per_day": float(np.mean(fa))})
        for fault in ("thermal_runaway", "load_oscillation", "intermittent_arcing", "sudden"):
            for sev in (0.5, 1.0, 1.5):
                leads = []
                for m in MONTHS:
                    for s in SEEDS:
                        df, meta = build(asset, m, fault, sev, 8, 2000 + s, warm)
                        al = alarms(predict_frame(art, df, asset), pol)
                        leads.append(lead_minutes(al, meta["failure_idx"], gap))
                det = [x for x in leads if x is not None]
                out["detection"].append({"asset": asset, "fault": fault, "severity": sev, "precursor_hours": 8,
                                         "n": len(leads), "detected": len(det), "detection_rate": len(det) / len(leads),
                                         "lead_median_min": float(np.median(det)) if det else None})
    fa = pd.DataFrame(out["false_alarms"])
    out["false_alarms_per_day_overall"] = float(fa.false_alarms_per_day.mean())
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    print("FA/day overall", round(o["false_alarms_per_day_overall"], 2))
    print(pd.DataFrame(o["false_alarms"]).pivot(index="month", columns="asset", values="false_alarms_per_day").round(2))
    d = pd.DataFrame(o["detection"])
    print(d.pivot_table(index=["fault", "severity"], columns="asset", values=["detection_rate", "lead_median_min"]).round(2))
