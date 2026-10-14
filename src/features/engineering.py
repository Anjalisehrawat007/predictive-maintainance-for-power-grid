"""Strictly causal feature engineering.

Every feature at timestamp ``t`` is a function of observations at
``t' <= t`` only (backward-looking rolling windows, differences, lags).
Quantities that need to be *estimated* (the thermal regression model, spike
thresholds) are fitted on the training period only via :meth:`fit`.

``FeatureEngineer.lookback`` is the number of past rows needed to compute an
exact feature row; the streaming engine keeps exactly that many rows in its
buffer and ``tests/test_streaming.py`` asserts streaming == offline features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
USEFUL_LOAD = ["HUFL", "MUFL", "LUFL"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


@dataclass
class _AssetFit:
    thermal_coef: np.ndarray
    thermal_cols: list[str]
    diff_sigma: dict[str, float]


@dataclass
class FeatureEngineer:
    sensor_cols: list[str]
    short_window: int = 4
    long_window: int = 16
    baseline_window: int = 96
    diff_lags: tuple[int, ...] = (1, 4)
    spike_sigma: float = 3.0
    time_features: bool = True
    adaptive_window: int = 0          # >0 adds regime-relative ("_a") features, see transform()
    robust: bool = False              # Phase 2: adds 24 h-relative ("_r") variability features
    fits: dict[str, _AssetFit] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, adaptive: bool = False, robust: bool = False) -> FeatureEngineer:
        f = cfg["features"]
        return cls(sensor_cols=cfg["dataset"]["sensor_cols"], short_window=f["short_window"],
                   long_window=f["long_window"], baseline_window=f["baseline_window"],
                   diff_lags=tuple(f["diff_lags"]), spike_sigma=f["spike_sigma"],
                   time_features=f["time_features"],
                   adaptive_window=int(f.get("adaptive_window", 672)) if adaptive else 0, robust=robust)

    # ------------------------------------------------------------------
    @property
    def lookback(self) -> int:
        """Rows of history required for an exact feature row (incl. current row)."""
        # robust features need a 24 h window over the residual, which itself needs 24 h of history
        return self.baseline_window + self.long_window + 2 + self.adaptive_window + (self.baseline_window if self.robust else 0)

    def _thermal_design(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal regressors for the 'expected' oil temperature given recent load."""
        X = pd.DataFrame(index=df.index)
        for c in USEFUL_LOAD:
            X[f"{c}_m{self.long_window}"] = df[c].rolling(self.long_window, min_periods=self.long_window).mean()
            X[f"{c}_m{self.baseline_window}"] = df[c].rolling(self.baseline_window, min_periods=self.baseline_window).mean()
        X["OT_m96_lag16"] = df["OT"].shift(self.long_window).rolling(
            self.baseline_window - self.long_window, min_periods=self.baseline_window - self.long_window).mean()
        return X

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, asset: str, train_mask: np.ndarray) -> FeatureEngineer:
        """Fit per-asset quantities using *training rows only*."""
        d = df.reset_index(drop=True)
        tm = np.asarray(train_mask, dtype=bool)
        usable = tm & ~d["offline"].to_numpy(dtype=bool) if "offline" in d else tm
        X = self._thermal_design(d)
        ok = usable & X.notna().all(axis=1).to_numpy()
        A = np.c_[X.to_numpy()[ok], np.ones(ok.sum())]
        y = d["OT"].to_numpy()[ok]
        lam = 1e-3 * np.eye(A.shape[1])
        coef = np.linalg.solve(A.T @ A + lam, A.T @ y)
        sig = {c: float(np.nanstd(d[c].diff().to_numpy()[usable])) + 1e-6 for c in self.sensor_cols}
        self.fits[asset] = _AssetFit(coef, list(X.columns), sig)
        return self

    def transform(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Return the full feature frame (raw + engineered + time) for one asset."""
        if asset not in self.fits:
            raise RuntimeError(f"FeatureEngineer not fitted for asset {asset}")
        fit = self.fits[asset]
        d = df.reset_index(drop=True)
        S, L, B = self.short_window, self.long_window, self.baseline_window
        out: dict[str, pd.Series] = {}
        for c in self.sensor_cols:
            x = d[c]
            out[c] = x
            d1 = x.diff(1)
            for lag in self.diff_lags:
                out[f"{c}_diff{lag}"] = x.diff(lag)
            out[f"{c}_rmean{S}"] = x.rolling(S, min_periods=S).mean()
            out[f"{c}_rstd{S}"] = x.rolling(S, min_periods=S).std()
            out[f"{c}_rstd{L}"] = x.rolling(L, min_periods=L).std()
            out[f"{c}_range{L}"] = x.rolling(L, min_periods=L).max() - x.rolling(L, min_periods=L).min()
            bm = x.rolling(B, min_periods=B).mean()
            bs = x.rolling(B, min_periods=B).std()
            out[f"{c}_zbase"] = (x - bm) / (bs + 1e-6)
            out[f"{c}_volratio"] = d1.rolling(L, min_periods=L).std() / (d1.rolling(B, min_periods=B).std() + 1e-6)
            spikes = (d1.abs() > self.spike_sigma * fit.diff_sigma[c]).astype(float)
            spikes[d1.isna()] = np.nan
            out[f"{c}_spikes{L}"] = spikes.rolling(L, min_periods=L).sum()
        # physics-informed: oil temperature not explained by recent load
        X = self._thermal_design(d)
        pred = np.c_[X.to_numpy(), np.ones(len(X))] @ fit.thermal_coef
        resid = d["OT"] - pred
        out["OT_resid"] = resid
        out[f"OT_resid_diff{S}"] = resid.diff(S)
        out[f"OT_resid_rmean{L}"] = resid.rolling(L, min_periods=L).mean()
        out[f"OT_diff{L}"] = d["OT"].diff(L)
        total = d[USEFUL_LOAD].sum(axis=1)
        out["load_total"] = total
        out[f"load_total_diff{S}"] = total.diff(S)
        if self.adaptive_window:
            out.update(self._adaptive(d, resid, total))
        if self.robust:
            out.update(self._robust(d, resid))
        if self.time_features:
            ts = pd.to_datetime(d["timestamp"])
            hr = ts.dt.hour + ts.dt.minute / 60.0
            out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
            out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7), np.cos(2 * np.pi * ts.dt.dayofweek / 7)
            out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25), np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
        return pd.DataFrame(out)

    def transform_last(self, raw: np.ndarray, timestamp, asset: str) -> dict[str, float]:
        """Features of the NEWEST row only, from a (lookback, n_sensors) array in ``sensor_cols`` order.

        Numerically equivalent to ``transform(df).iloc[-1]`` for the base feature sets (verified by
        tests/test_streaming.py) but ~10x cheaper: the streaming engine calls it once per observation
        instead of recomputing every rolling window over the whole buffer with pandas.
        """
        if self.robust or self.adaptive_window:
            raise NotImplementedError("transform_last supports the base feature sets only")
        fit = self.fits[asset]
        S, L, B = self.short_window, self.long_window, self.baseline_window
        col = {c: raw[:, j].astype(float) for j, c in enumerate(self.sensor_cols)}
        out: dict[str, float] = {}

        def std(v):
            return float(np.std(v, ddof=1))
        for c in self.sensor_cols:
            x = col[c]
            d1 = np.diff(x)
            out[c] = x[-1]
            for lag in self.diff_lags:
                out[f"{c}_diff{lag}"] = x[-1] - x[-1 - lag]
            out[f"{c}_rmean{S}"] = float(x[-S:].mean())
            out[f"{c}_rstd{S}"] = std(x[-S:])
            out[f"{c}_rstd{L}"] = std(x[-L:])
            out[f"{c}_range{L}"] = float(x[-L:].max() - x[-L:].min())
            out[f"{c}_zbase"] = (x[-1] - x[-B:].mean()) / (std(x[-B:]) + 1e-6)
            out[f"{c}_volratio"] = std(d1[-L:]) / (std(d1[-B:]) + 1e-6)
            out[f"{c}_spikes{L}"] = float((np.abs(d1[-L:]) > self.spike_sigma * fit.diff_sigma[c]).sum())

        def resid_at(k: int) -> float:                     # k = steps back from the newest row
            e = len(raw) - k                               # exclusive end index of the window ending at t-k
            v = [col[c][e - L:e].mean() for c in USEFUL_LOAD]
            v = [x for pair in zip(v, [col[c][e - B:e].mean() for c in USEFUL_LOAD]) for x in pair]
            v.append(col["OT"][e - L - (B - L): e - L].mean())
            return col["OT"][e - 1] - float(np.dot(np.r_[v, 1.0], fit.thermal_coef))
        r = np.array([resid_at(k) for k in range(L)])[::-1]   # resid at t-L+1 .. t
        out["OT_resid"] = r[-1]
        out[f"OT_resid_diff{S}"] = r[-1] - r[-1 - S]
        out[f"OT_resid_rmean{L}"] = float(r.mean())
        out[f"OT_diff{L}"] = col["OT"][-1] - col["OT"][-1 - L]
        tot = sum(col[c] for c in USEFUL_LOAD)
        out["load_total"] = float(tot[-1])
        out[f"load_total_diff{S}"] = float(tot[-1] - tot[-1 - S])
        if self.time_features:
            ts = pd.Timestamp(timestamp)
            hr = ts.hour + ts.minute / 60.0
            out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
            out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * ts.dayofweek / 7), np.cos(2 * np.pi * ts.dayofweek / 7)
            out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * ts.dayofyear / 365.25), np.cos(2 * np.pi * ts.dayofyear / 365.25)
        return out

    def _robust(self, d: pd.DataFrame, resid: pd.Series) -> dict[str, pd.Series]:
        """Variability features relative to the channel's own *last 24 h* step-to-step std
        (rolling std, not median, over the baseline window). Unlike the 7-day median scale this
        follows daily regime changes quickly and is not degenerate for slowly varying channels."""
        S, L, B = self.short_window, self.long_window, self.baseline_window
        out: dict[str, pd.Series] = {}
        for c in self.sensor_cols:
            x = d[c]
            sc = x.diff(1).rolling(B, min_periods=B).std() + 1e-3 * (x.abs().rolling(B, min_periods=B).mean() + 1e-3)
            for lag in self.diff_lags:
                out[f"{c}_diff{lag}_r"] = (x.diff(lag) / sc).clip(-50, 50)
            out[f"{c}_rstd{S}_r"] = (x.rolling(S, min_periods=S).std() / sc).clip(0, 50)
            out[f"{c}_range{L}_r"] = ((x.rolling(L, min_periods=L).max() - x.rolling(L, min_periods=L).min()) / sc).clip(0, 100)
        rsc = resid.rolling(B, min_periods=B).std() + 1e-3
        out["OT_resid_z24_r"] = ((resid - resid.rolling(B, min_periods=B).mean()) / rsc).clip(-20, 20)
        return out

    def _adaptive(self, d: pd.DataFrame, resid: pd.Series, total: pd.Series) -> dict[str, pd.Series]:
        """Regime-relative features: scale-dependent quantities divided by a *robust, causal*
        trailing scale = 1.4826 * rolling median(|Δx|) over ``adaptive_window`` steps.

        The median (not the std) keeps the scale insensitive to the huge jumps of
        outages, so sensitivity does not collapse for a week after every trip.
        """
        A, S, L = self.adaptive_window, self.short_window, self.long_window
        out: dict[str, pd.Series] = {}

        def scale(x: pd.Series) -> pd.Series:
            return 1.4826 * x.diff(1).abs().rolling(A, min_periods=A).median() + 1e-6

        for c in self.sensor_cols:
            x = d[c]
            sc = scale(x)
            d1 = x.diff(1)
            for lag in self.diff_lags:
                out[f"{c}_diff{lag}_a"] = x.diff(lag) / sc
            out[f"{c}_rstd{S}_a"] = x.rolling(S, min_periods=S).std() / sc
            out[f"{c}_rstd{L}_a"] = x.rolling(L, min_periods=L).std() / sc
            out[f"{c}_range{L}_a"] = (x.rolling(L, min_periods=L).max() - x.rolling(L, min_periods=L).min()) / sc
            sp = (d1.abs() > self.spike_sigma * sc).astype(float)
            sp[d1.isna() | sc.isna()] = np.nan
            out[f"{c}_spikes{L}_a"] = sp.rolling(L, min_periods=L).sum()
        rs = scale(resid)
        dev = resid - resid.rolling(A, min_periods=A).median()
        out["OT_resid_a"] = dev / rs
        out[f"OT_resid_diff{S}_a"] = resid.diff(S) / rs
        out[f"OT_resid_rmean{L}_a"] = (dev / rs).rolling(L, min_periods=L).mean()
        out[f"OT_diff{L}_a"] = d["OT"].diff(L) / scale(d["OT"])
        out[f"load_total_diff{S}_a"] = total.diff(S) / scale(total)
        return out

    # ------------------------------------------------------------------
    def feature_sets(self) -> dict[str, list[str]]:
        """Column groups used for the raw / engineered / combined ablation."""
        dummy = pd.DataFrame({c: np.zeros(self.lookback + 2) for c in self.sensor_cols})
        dummy["timestamp"] = pd.date_range("2020-01-01", periods=len(dummy), freq="15min")
        tmp = FeatureEngineer(**{k: getattr(self, k) for k in (
            "sensor_cols", "short_window", "long_window", "baseline_window", "diff_lags",
            "spike_sigma", "time_features", "adaptive_window", "robust")})
        tmp.fits["_"] = _AssetFit(np.zeros(len(self._thermal_design(dummy).columns) + 1),
                                  [], {c: 1.0 for c in self.sensor_cols})
        cols = list(tmp.transform(dummy, "_").columns)
        raw = list(self.sensor_cols)
        eng = [c for c in cols if c not in raw]
        # 'stationary': shape/deviation features only — drops absolute levels and the
        # annual cycle, which drift between periods and invite memorisation of
        # when (in the calendar) training failures happened.
        level = {f"{c}_rmean{self.short_window}" for c in raw} | {"load_total", "doy_sin", "doy_cos"}
        base = [c for c in eng if not c.endswith(("_a", "_r"))]
        stat = [c for c in base if c not in level]
        sets = {"raw": raw, "engineered": base, "raw_engineered": raw + base, "stationary": stat,
                "stationary_nocal": [c for c in stat if c not in TIME_FEATURES]}
        if self.robust:
            # robust = stationary with absolute-unit variability features replaced by 24 h-relative ones
            absvar = tuple(f"_{k}" for k in (f"diff{lag}" for lag in self.diff_lags)) + (
                f"_rstd{self.short_window}", f"_range{self.long_window}")
            keep = [c for c in stat if not (c.endswith(absvar) and c.split("_")[0] in self.sensor_cols
                                            and not c.startswith("OT_resid"))]
            sets["robust"] = keep + [c for c in eng if c.endswith("_r")]
        if self.adaptive_window:
            # adaptive = already-relative features (z-score vs 24 h baseline, volatility ratio,
            # hour/day) + every scale-dependent feature in its regime-relative "_a" form
            relative = [c for c in stat if c.endswith(("_zbase", "_volratio")) or c in TIME_FEATURES]
            sets["adaptive"] = relative + [c for c in eng if c.endswith("_a")]
        return sets
