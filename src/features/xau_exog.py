"""Exogenous / proxy driver features for XAUUSD — V5 Track 1.

The rejected 2026-07-05 meta-labeling attempt used ONLY XAUUSD-derived features
(|forecast|, ATR, returns, SMA distance) — saturated OHLC information, OOS AUC
~0.51. Gold's real drivers are exogenous: USD strength, risk sentiment, and
real yields. This module builds those as PROXIES from the FX CSVs already in
`data/` (no external data needed to start), with hooks to swap in real
DXY / US10Y-real-yield / VIX CSVs later (`real_paths`).

Causality contract (tested in tests/test_v5_xau_exog.py):
  Every feature at target bar t uses only source bars COMPLETED strictly before
  t. Each source series is computed on its own index, `.shift(1)`-ed (so row t
  carries bar t-1), then reindexed onto the XAUUSD index with forward-fill.
  Mutating any future source bar cannot change a past feature value.

Synthetic USD-strength index (higher = stronger USD):
  USD strengthens when USDJPY rises and EURUSD / GBPUSD fall, so the per-bar
  index return is  mean( +ret(USDJPY), -ret(EURUSD), -ret(GBPUSD) )  over
  whichever of those pairs are available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Sign of each pair's return in the USD-strength index (USD as the numeraire).
_USD_LEGS: dict[str, float] = {"USDJPY": +1.0, "EURUSD": -1.0, "GBPUSD": -1.0}
_DEFAULT_TF = "H4_long"


def _load_close(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = next((c for c in df.columns if "time" in c or c in {"date", "datetime"}), None)
    if tcol is None:
        return None
    df[tcol] = pd.to_datetime(df[tcol])
    s = df.set_index(tcol)["close"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _reindex_causal(series: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """shift(1) then align to target with ffill — past-only by construction."""
    return series.shift(1).reindex(target, method="ffill")


def _usd_strength_return(data_dir: Path, tf: str) -> pd.Series | None:
    """Per-bar synthetic USD-strength index return on its own (union) index."""
    legs = []
    for sym, sign in _USD_LEGS.items():
        close = _load_close(data_dir / f"{sym}_{tf}.csv")
        if close is not None:
            legs.append(sign * close.pct_change())
    if not legs:
        return None
    frame = pd.concat(legs, axis=1)
    return frame.mean(axis=1, skipna=True).dropna()


def _cross_pair_vol(data_dir: Path, tf: str, window: int) -> pd.Series | None:
    """Risk proxy: mean rolling abs-return across the FX legs (own index)."""
    vols = []
    for sym in _USD_LEGS:
        close = _load_close(data_dir / f"{sym}_{tf}.csv")
        if close is not None:
            vols.append(close.pct_change().abs().rolling(window, min_periods=window // 2).mean())
    if not vols:
        return None
    return pd.concat(vols, axis=1).mean(axis=1, skipna=True).dropna()


def _ewmac_sign(series: pd.Series, fast: int, slow: int) -> pd.Series:
    return np.sign(series.ewm(span=fast, min_periods=fast).mean()
                   - series.ewm(span=slow, min_periods=slow).mean())


def add_xau_exog_features(
    xau: pd.DataFrame,
    data_dir: str | Path = "data",
    timeframe: str = _DEFAULT_TF,
    *,
    vol_window: int = 20,
    trend_window: int = 20,
    corr_window: int = 60,
    real_paths: Mapping[str, str | Path] | None = None,
) -> pd.DataFrame:
    """Return an exogenous-feature frame indexed like `xau` (XAUUSD H4 OHLC).

    Columns (all completed-bar, shift-safe):
      usd_strength_ret_1   1-bar synthetic USD-index return
      usd_strength_z       z-score of the cumulative USD index over `trend_window`
      usd_strength_trend   EWMAC sign (+1/-1) of the cumulative USD index
      risk_vol             cross-pair mean abs-return (risk/vol proxy)
      xau_usd_corr         rolling corr(XAU ret, USD-index ret) — regime tell
      real_<name>_ret_1    1-bar return of each supplied real macro CSV (optional)

    Missing FX CSVs are skipped (columns filled 0.0) so the pipeline still runs;
    `available_exog(...)` reports what was actually loaded.
    """
    data_dir = Path(data_dir)
    idx = xau.index
    out = pd.DataFrame(index=idx)

    usd_ret = _usd_strength_return(data_dir, timeframe)
    if usd_ret is not None:
        out["usd_strength_ret_1"] = _reindex_causal(usd_ret, idx).fillna(0.0)
        usd_level = usd_ret.cumsum()  # synthetic index level (own grid)
        z = ((usd_level - usd_level.rolling(trend_window, min_periods=trend_window).mean())
             / usd_level.rolling(trend_window, min_periods=trend_window).std())
        out["usd_strength_z"] = _reindex_causal(z, idx).fillna(0.0)
        out["usd_strength_trend"] = _reindex_causal(
            _ewmac_sign(usd_level, trend_window, trend_window * 4), idx).fillna(0.0)
    else:
        out["usd_strength_ret_1"] = 0.0
        out["usd_strength_z"] = 0.0
        out["usd_strength_trend"] = 0.0

    risk = _cross_pair_vol(data_dir, timeframe, vol_window)
    out["risk_vol"] = (_reindex_causal(risk, idx).fillna(0.0)
                       if risk is not None else 0.0)

    # Rolling correlation of gold vs USD index — computed on the shared target
    # grid using already-past-shifted series (both lag by >=1 bar).
    xau_ret_lag = xau["close"].pct_change().shift(1)
    if usd_ret is not None:
        usd_on_xau = usd_ret.shift(1).reindex(idx, method="ffill")
        out["xau_usd_corr"] = (xau_ret_lag.rolling(corr_window, min_periods=corr_window // 2)
                               .corr(usd_on_xau).fillna(0.0))
    else:
        out["xau_usd_corr"] = 0.0

    if real_paths:
        for name, path in real_paths.items():
            close = _load_close(Path(path))
            col = f"real_{name.lower()}_ret_1"
            out[col] = (_reindex_causal(close.pct_change(), idx).fillna(0.0)
                        if close is not None else 0.0)

    return out


def available_exog(data_dir: str | Path = "data", timeframe: str = _DEFAULT_TF) -> list[str]:
    """FX legs with a CSV present, for logging what the proxy actually used."""
    data_dir = Path(data_dir)
    return [s for s in _USD_LEGS if (data_dir / f"{s}_{timeframe}.csv").exists()]
