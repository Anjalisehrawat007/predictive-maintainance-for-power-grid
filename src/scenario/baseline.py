"""Synthetic transformer baseline for the public Scenario Lab.

The public website must not serve ETDataset rows (CC BY-ND 4.0), so visitors build fault scenarios on a
**synthetic** normal-operation stream instead. This module fits a compact statistical model of normal operation
on the *training period* of each transformer and stores only aggregate parameters (no rows):

* a mean 15-minute-of-day profile per channel and calendar month (12 × 96 values), plus a day-of-week offset;
* a slow day-to-day level drift (AR(1) on daily means, correlated across channels);
* fast 15-minute fluctuations (AR(1) per channel with cross-channel correlated innovations).

    python -m src.scenario.baseline          # → configs/scenario_baseline.json

The generator samples new, never-observed series from these parameters. Synthetic series are an approximation
of normal operation; detection results on them are illustrative and are not the benchmark.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config

SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
OUT = PROJECT_ROOT / "configs" / "scenario_baseline.json"
SLOTS = 96


def _ar1(x: np.ndarray) -> tuple[float, np.ndarray]:
    """AR(1) coefficient and innovations of a mean-zero series (NaN-safe)."""
    a, b = x[:-1], x[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    phi = float(np.clip(np.dot(a[ok], b[ok]) / max(np.dot(a[ok], a[ok]), 1e-12), 0.0, 0.999))
    inn = b - phi * a
    return phi, inn


def fit(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    bounds = json.loads((PROJECT_ROOT / "data" / "splits" / "split_bounds.json").read_text())
    out = {"about": "Aggregate parameters of normal operation fitted on each transformer's TRAINING period "
                    "(no data rows). Used to synthesise baseline streams for the Scenario Lab.",
           "sensors": SENSORS, "slots_per_day": SLOTS, "assets": {}}
    for asset in ("ETTm1", "ETTm2"):
        df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / f"{asset}.parquet")
        df = df.iloc[: bounds[asset]["train_end"]].copy()
        # real readings only: undo injected deltas and drop simulated-event windows and outages
        for c in SENSORS:
            df[c] = df[c] - df[f"sim_delta_{c}"]
        df.loc[(df.sim_phase != "normal") | df.offline, SENSORS] = np.nan
        ts = pd.to_datetime(df.timestamp)
        slot = (ts.dt.hour * 4 + ts.dt.minute // 15).to_numpy()
        month, dow = ts.dt.month.to_numpy(), ts.dt.dayofweek.to_numpy()
        X = df[SENSORS].to_numpy(dtype=float)

        prof = np.zeros((12, SLOTS, len(SENSORS)))
        for m in range(1, 13):
            for s in range(SLOTS):
                sel = (month == m) & (slot == s)
                prof[m - 1, s] = np.nanmean(X[sel], axis=0)
        R = X - prof[month - 1, slot]
        dow_off = np.stack([np.nanmean(R[dow == d], axis=0) for d in range(7)])
        R = R - dow_off[dow]

        # slow component: daily means of the residual → AR(1) with correlated innovations
        day = (ts - ts.iloc[0]).dt.days.to_numpy()
        dly = pd.DataFrame(R).groupby(day).mean()
        daily = dly.to_numpy()
        phi_d, inn_d = zip(*[_ar1(daily[:, j]) for j in range(len(SENSORS))])
        inn_d = np.stack(inn_d, 1)
        cov_d = pd.DataFrame(inn_d).cov().to_numpy()
        # fast component: 15-min residual around the (interpolated) daily level
        level = dly.reindex(day).to_numpy()
        F = R - level
        phi_f, inn_f = zip(*[_ar1(F[:, j]) for j in range(len(SENSORS))])
        inn_f = np.stack(inn_f, 1)
        # volatility clustering: log of the 4-h rolling std of the innovations follows an AR(1)
        rs = pd.DataFrame(inn_f).rolling(16, min_periods=8).std().to_numpy()
        lv = np.log(np.clip(rs, 1e-6, None))
        lv = lv - np.nanmean(lv, 0)
        phi_v, sig_v = [], []
        for j in range(len(SENSORS)):
            ph, iv = _ar1(lv[:, j])
            phi_v.append(ph); sig_v.append(float(np.nanstd(iv)))
        # heavy tails: excess kurtosis of volatility-normalised innovations → Student-t degrees of freedom
        u = inn_f / np.clip(rs, 1e-6, None)
        u = u[1:]
        kurt = pd.DataFrame(u).kurt().to_numpy()
        dof = np.where(kurt > 0.05, np.clip(4 + 6 / np.maximum(kurt, 1e-3), 4.2, 60), 60)
        # month-specific scale and cross-channel correlation of the innovations
        corr = pd.DataFrame(inn_f).corr().to_numpy()
        sd_m = np.stack([np.nanstd(inn_f[month[1:] == m], 0) for m in range(1, 13)])
        ref = _shape_stats(pd.DataFrame(X, columns=SENSORS), month)
        from src.data.simulation import local_sigma
        sigma_real = [local_sigma(np.nan_to_num(X[:, j], nan=float(np.nanmean(X[:, j])))) for j in range(len(SENSORS))]
        out["assets"][asset] = {
            "ref_stats": ref,
            "local_sigma_real": np.round(sigma_real, 6).tolist(),
            "profile": np.round(prof, 4).tolist(), "dow_offset": np.round(dow_off, 4).tolist(),
            "phi_daily": np.round(phi_d, 4).tolist(), "cov_daily": np.round(cov_d, 6).tolist(),
            "phi_fast": np.round(phi_f, 4).tolist(), "corr_fast": np.round(corr, 4).tolist(),
            "sd_fast_month": np.round(sd_m, 5).tolist(), "phi_vol": np.round(phi_v, 4).tolist(),
            "sigma_vol": np.round(sig_v, 4).tolist(), "t_dof": np.round(dof, 2).tolist(),
        }
    OUT.write_text(json.dumps(out))
    for asset in out["assets"]:
        _calibrate(out, asset)
    OUT.write_text(json.dumps(out))
    return out


DOF_GRID = [4.5, 6.0, 8.0, 12.0, 20.0, 60.0]


def _shape_stats(df: pd.DataFrame, month) -> dict:
    """Aggregate shape statistics per channel: median 4-h rolling std of 15-min changes per month, and the
    tail ratio q99.5(|change|) / median(|change|) (how often big jumps happen)."""
    d = df[SENSORS].diff()
    rs = d.rolling(16, min_periods=8).std()
    m = np.asarray(month)
    with warnings.catch_warnings():                 # months absent from a sample give NaN (expected)
        warnings.simplefilter("ignore", RuntimeWarning)
        vol = [[float(np.nanmedian(rs[c].to_numpy()[m == mm])) for c in SENSORS] for mm in range(1, 13)]
    a = d.abs()
    tail = [float(np.nanquantile(a[c], 0.995) / max(np.nanmedian(a[c]), 1e-9)) for c in SENSORS]
    lvl = df[SENSORS].rolling(16, min_periods=8).std()
    smooth = [float(np.nanmedian(lvl[c]) / max(np.nanmedian(rs[c]), 1e-9)) for c in SENSORS]
    return {"vol_month": vol, "tail": tail, "smooth": smooth}


def _calibrate(out: dict, asset: str, iters: int = 3) -> None:
    """Moment-match the generator to the training-period shape statistics (scale per month & channel,
    Student-t degrees of freedom per channel). Only aggregate statistics are compared."""
    P = out["assets"][asset]
    ref_vol = np.asarray(P["ref_stats"]["vol_month"])
    ref_tail = np.asarray(P["ref_stats"]["tail"])
    for _ in range(iters):
        OUT.write_text(json.dumps(out))
        sims = [generate(asset, f"2017-{m:02d}-01", 96 * 20, seed=100 + m) for m in range(1, 13)]
        syn_vol = np.stack([_shape_stats(g, g.timestamp.dt.month)["vol_month"][m - 1] for m, g in zip(range(1, 13), sims)])
        P["sd_fast_month"] = np.round(np.asarray(P["sd_fast_month"]) * np.clip(ref_vol / np.clip(syn_vol, 1e-9, None), 0.2, 5), 6).tolist()
    # tail weight: pick the dof whose tail ratio is closest to the real one (per channel)
    best = []
    base = list(P["t_dof"])
    for j in range(len(SENSORS)):
        errs = []
        for dof in DOF_GRID:
            P["t_dof"] = [dof if k == j else base[k] for k in range(len(SENSORS))]
            OUT.write_text(json.dumps(out))
            g = pd.concat([generate(asset, f"2017-{m:02d}-01", 96 * 10, seed=200 + m) for m in (1, 4, 7, 10)])
            errs.append(abs(np.log(_shape_stats(g, g.timestamp.dt.month)["tail"][j] / ref_tail[j])))
        best.append(DOF_GRID[int(np.argmin(errs))])
    P["t_dof"] = best


def _params(asset: str) -> dict:
    p = json.loads(OUT.read_text())["assets"][asset]
    return {k: np.asarray(v) for k, v in p.items()}


def generate(asset: str, start: str, n: int, seed: int) -> pd.DataFrame:
    """Sample ``n`` 15-minute rows of synthetic normal operation starting at ``start``."""
    p = _params(asset)
    rng = np.random.default_rng(seed)
    ts = pd.date_range(pd.Timestamp(start).floor("15min"), periods=n, freq="15min")
    slot = (ts.hour * 4 + ts.minute // 15).to_numpy()
    k = len(SENSORS)
    base = p["profile"][ts.month.to_numpy() - 1, slot] + p["dow_offset"][ts.dayofweek.to_numpy()]
    # daily level drift
    days = (ts.normalize() - ts[0].normalize()).days.to_numpy()
    nd = days.max() + 1
    Ld = np.linalg.cholesky(p["cov_daily"] + 1e-9 * np.eye(k))
    lv = np.zeros((nd, k))
    lv[0] = rng.standard_normal(k) @ Ld.T / np.sqrt(np.maximum(1 - p["phi_daily"] ** 2, 1e-3))
    for d in range(1, nd):
        lv[d] = p["phi_daily"] * lv[d - 1] + rng.standard_normal(k) @ Ld.T
    # smooth the day-to-day level so there are no midnight steps
    lvl = pd.DataFrame(lv[days]).rolling(96, min_periods=1, center=False).mean().to_numpy()
    # fast fluctuations: AR(1), Student-t innovations, correlated across channels, with stochastic volatility
    Lc = np.linalg.cholesky(p["corr_fast"] + 1e-6 * np.eye(k))
    dof = p["t_dof"]
    g = rng.standard_normal((n, k)) @ Lc.T                         # correlated Gaussian
    chi = rng.chisquare(dof, size=(n, k))
    t = g / np.sqrt(chi / dof) * np.sqrt((dof - 2) / dof)           # unit-variance Student-t
    v = np.zeros((n, k))
    for i in range(1, n):
        v[i] = p["phi_vol"] * v[i - 1] + p["sigma_vol"] * rng.standard_normal(k)
    var_v = p["sigma_vol"] ** 2 / np.maximum(1 - p["phi_vol"] ** 2, 1e-3)
    sd = p["sd_fast_month"][ts.month.to_numpy() - 1] * np.exp(v - var_v / 2)
    z = t * sd
    e = np.zeros((n, k))
    e[0] = z[0] / np.sqrt(np.maximum(1 - p["phi_fast"] ** 2, 1e-3))
    for i in range(1, n):
        e[i] = p["phi_fast"] * e[i - 1] + z[i]
    X = base + lvl + e
    return pd.DataFrame({"timestamp": ts, **{c: X[:, j] for j, c in enumerate(SENSORS)}})


if __name__ == "__main__":
    o = fit()
    print({a: {"t_dof": v["t_dof"], "phi_vol": v["phi_vol"]} for a, v in o["assets"].items()})
