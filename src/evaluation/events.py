"""Event-level (operational) evaluation: alarms, lead time, false alarms.

Definitions (all in 15-min steps, configurable via ``threshold`` config):

* **alarm**     — probability >= threshold for ``k`` consecutive scored steps.
* **episode**   — contiguous alarm steps; gaps <= ``merge_gap`` are merged
                  (never across offline/outage rows).
* **reliable warning** for failure at t_f — an episode that overlaps the event
  window [t_f - W, t_f) and is still active at most ``merge_gap`` steps before
  t_f.  Lead time = t_f - max(episode start, t_f - W).
* **false alarm** — an episode that overlaps no event window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP_MIN = 15


def persistence_alarm(prob: np.ndarray, thr: float, k: int) -> np.ndarray:
    above = np.nan_to_num(prob, nan=-1.0) >= thr
    if k <= 1:
        return above
    run = np.zeros(len(above), dtype=int)
    for i, a in enumerate(above):
        run[i] = run[i - 1] + 1 if (a and i > 0) else int(a)
    return run >= k


def alarm_episodes(alarm: np.ndarray, valid: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Return [start, end] (inclusive) episodes; never merge across invalid rows."""
    eps: list[list[int]] = []
    idx = np.where(alarm)[0]
    for i in idx:
        if eps and i - eps[-1][1] - 1 <= merge_gap and valid[eps[-1][1]: i + 1].all():
            eps[-1][1] = i
        else:
            eps.append([i, i])
    return [(a, b) for a, b in eps]


def evaluate_events(p, split: str, probs: np.ndarray, thr: float, events: pd.DataFrame,
                    cfg: dict) -> dict:
    """Operational metrics for one split. ``probs`` aligns with ``p.idx[split]``."""
    k = cfg["threshold"]["persistence_k"]
    g = cfg["threshold"]["merge_gap_steps"]
    W = int(cfg["labels"]["event_window_hours"] * 60 / STEP_MIN)
    rows, n_episodes, false_eps, days, alarm_steps, valid_steps = [], 0, [], 0.0, 0, 0
    fev = events[events.kind == "failure"]
    for a, off in p.offsets.items():
        lo, hi = p.bounds[a].ranges()[split]
        m = (p.idx[split] >= off + lo) & (p.idx[split] < off + hi)
        full = np.full(hi - lo, np.nan)
        full[p.idx[split][m] - off - lo] = probs[m]
        valid = ~np.isnan(full)
        days += valid.sum() * STEP_MIN / 1440
        alarm = persistence_alarm(full, thr, k)
        alarm_steps += int(alarm[valid].sum()); valid_steps += int(valid.sum())
        eps = alarm_episodes(alarm, valid, g)
        n_episodes += len(eps)
        fa = fev[(fev.asset == a) & (fev.failure_idx >= lo) & (fev.failure_idx < hi)]
        windows = [(int(r.failure_idx) - lo - W, int(r.failure_idx) - lo) for r in fa.itertuples()]
        for s, e in eps:
            if not any(s < wf and e >= ws for ws, wf in windows):
                false_eps.append({"asset": a, "start": p.timestamps[off + lo + s],
                                  "end": p.timestamps[off + lo + e], "duration_steps": e - s + 1})
        for r, (ws, wf) in zip(fa.itertuples(), windows):
            over = [(s, e) for s, e in eps if s < wf and e >= ws]
            reliable = [(s, e) for s, e in over if e >= wf - 1 - g]
            lead = (wf - max(reliable[-1][0], ws)) * STEP_MIN if reliable else np.nan
            first_any = (wf - max(over[0][0], ws)) * STEP_MIN if over else np.nan
            onset_lead = (r.failure_idx - r.onset_idx) * STEP_MIN
            rows.append({"asset": a, "event_id": r.event_id, "type": r.type,
                         "failure_time": p.timestamps[off + r.failure_idx],
                         "onset_time": p.timestamps[off + r.onset_idx],
                         "precursor_min": onset_lead, "detected": bool(reliable),
                         "lead_time_min": lead, "first_alarm_lead_min": first_any,
                         "warning_time": (p.timestamps[off + lo + max(reliable[-1][0], ws)]
                                          if reliable else pd.NaT)})
    per = pd.DataFrame(rows)
    det = per[per.detected] if len(per) else per
    detectable = per[per.type != "sudden"] if len(per) else per
    lt = det["lead_time_min"].to_numpy(float) if len(det) else np.array([])
    out = {
        "n_failures": int(len(per)),
        "n_detectable": int(len(detectable)),
        "n_detected": int(per.detected.sum()) if len(per) else 0,
        "event_recall": float(per.detected.mean()) if len(per) else float("nan"),
        "event_recall_detectable": float(detectable.detected.mean()) if len(detectable) else float("nan"),
        "lead_time_mean_min": float(np.mean(lt)) if len(lt) else float("nan"),
        "lead_time_median_min": float(np.median(lt)) if len(lt) else float("nan"),
        "lead_time_min_min": float(np.min(lt)) if len(lt) else float("nan"),
        "lead_time_max_min": float(np.max(lt)) if len(lt) else float("nan"),
        "lead_time_p25_min": float(np.percentile(lt, 25)) if len(lt) else float("nan"),
        "alarm_episodes": int(n_episodes),
        "false_alarm_episodes": int(len(false_eps)),
        "false_alarms_per_day": len(false_eps) / days if days else float("nan"),
        "false_alarms_per_week": 7 * len(false_eps) / days if days else float("nan"),
        "episode_precision": 1 - len(false_eps) / n_episodes if n_episodes else float("nan"),
        "scored_days": days,
        # share of operating time spent in alarm: guards against "always-on" alarms that
        # would otherwise look good on event recall with few (long) episodes
        "alarm_time_fraction": alarm_steps / valid_steps if valid_steps else float("nan"),
    }
    if len(per):
        for t, gdf in per.groupby("type"):
            out[f"event_recall_{t}"] = float(gdf.detected.mean())
    return {"summary": out, "per_failure": per, "false_alarms": pd.DataFrame(false_eps)}
