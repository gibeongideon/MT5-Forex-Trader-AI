"""XAUUSD single-symbol research lab — continuous-position backtester.

Conventions match repo (scripts/v5_xau_backtest.py): net Sharpe from
daily-resampled equity pct-changes * sqrt(252), eval 2017+ primary.
Positions are exposure fractions of equity (leverage), decided on bar
close t, applied to return t+1 (next-bar, causal). Costs charged on
turnover using the bar's spread column + slippage.

All signals are lookahead-free: EWMAs/rolling stats on closes <= t.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

EVAL_START = "2017-01-01"
SLIP_USD = 0.10          # 1 pip slippage, matches live engine
ANN_H4 = int(252 * 6)    # H4 bars/year (bar count basis for vol ann.)
ANN_D1 = 252


# ---------------------------------------------------------------- data
def load_h4() -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H4_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df


def load_d1() -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}/data/GOLD_D1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # GOLD_D1 spread column is already in price units
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median())
    return df


# ------------------------------------------------------------- signals
def ewmac_fc(close: pd.Series, speeds, cap=2.0) -> pd.Series:
    """Carver EWMAC forecast, |1| ~ average strength (repo convention /target)."""
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in speeds:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(speeds) * 1.3).clip(-cap, cap)


def breakout_fc(close: pd.Series, windows, cap=2.0) -> pd.Series:
    """Carver breakout: (close - mid(N)) / range(N), smoothed span N/4."""
    combined = None
    for n in windows:
        hi = close.rolling(n, min_periods=n // 2).max()
        lo = close.rolling(n, min_periods=n // 2).min()
        mid = (hi + lo) / 2.0
        rng = (hi - lo).replace(0.0, np.nan)
        raw = ((close - mid) / rng * 4.0).ewm(span=max(2, n // 4)).mean()
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(windows) * 1.2).clip(-cap, cap)


def tsmom_fc(close: pd.Series, lookbacks, cap=2.0) -> pd.Series:
    sigs = [np.sign(close / close.shift(L) - 1.0) for L in lookbacks]
    return (sum(sigs) / len(sigs)).clip(-cap, cap)


def accel_fc(close: pd.Series, speeds, cap=2.0) -> pd.Series:
    """Trend acceleration: change in EWMAC forecast over fast span."""
    combined = None
    for fast, slow in speeds:
        base = ewmac_fc(close, [(fast, slow)], cap=cap * 2)
        raw = base - base.shift(fast)
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(speeds)).clip(-cap, cap)


def skew_fc(ret: pd.Series, window: int = 120, cap=2.0) -> pd.Series:
    """Negative-of-skew signal (sell positively skewed lottery moves)."""
    raw = -ret.rolling(window, min_periods=window // 2).skew()
    scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
    return (raw * scalar).clip(-cap, cap)


# ------------------------------------------------------------ engine
def run(df: pd.DataFrame, fc: pd.Series, *, ann: int, target_vol: float = 0.10,
        vol_hl: int = 42, buffer_frac: float = 0.0, long_tilt: float = 0.0,
        max_lev: float = 8.0, spread_mult: float = 1.0, delay: int = 1,
        vol_floor_q: float = 0.0) -> dict:
    """Continuous vol-targeted position from forecast; returns metrics dict."""
    close = df["close"]
    ret = close.pct_change()
    # per-bar realized vol -> annualized, causal
    vol = ret.ewm(halflife=vol_hl, min_periods=20).std() * np.sqrt(ann)
    if vol_floor_q > 0:
        floor = vol.expanding(min_periods=100).quantile(vol_floor_q).shift(1)
        vol = np.maximum(vol, floor)
    fc2 = (fc + long_tilt).clip(-2.0, 2.0)
    pos = (fc2 * (target_vol / vol)).clip(-max_lev, max_lev)

    if buffer_frac > 0:  # causal no-trade band around current holding
        avg = (target_vol / vol).clip(0, max_lev)
        band = buffer_frac * avg
        p = pos.values.copy()
        held = 0.0
        out = np.zeros_like(p)
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)

    pos = pos.shift(delay).fillna(0.0)
    cost_frac = ((df["spread_px"] * spread_mult / 2.0 + SLIP_USD) / close)
    gross = pos * ret
    costs = pos.diff().abs().fillna(0.0) * cost_frac
    net = (gross - costs).fillna(0.0)

    eq = (1.0 + net).cumprod()
    out = {}
    for tag, sl in (("full", slice(None, None)), ("eval", slice(EVAL_START, None))):
        e = eq.loc[sl]
        e = e / e.iloc[0]
        daily = e.resample("D").last().pct_change(fill_method=None).dropna()
        sh = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        dd = float((e / e.cummax() - 1.0).min() * 100)
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        cagr = float((e.iloc[-1]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        out[f"sharpe_{tag}"] = round(sh, 3)
        out[f"dd_{tag}"] = round(dd, 1)
        out[f"cagr_{tag}"] = round(cagr, 1)
    out["turnover_yr"] = round(float(pos.diff().abs().sum()
                                     / max((pos.index[-1] - pos.index[0]).days / 365.25, 1e-9)), 1)
    out["avg_abs_pos"] = round(float(pos.abs().mean()), 2)
    return out


def log_result(name: str, params: dict, metrics: dict) -> None:
    row = {"name": name, **metrics, "params": json.dumps(params)}
    hdr = not os.path.exists(LOG)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=hdr, index=False)
    print(f"{name:44s} eval {metrics['sharpe_eval']:+.3f} full {metrics['sharpe_full']:+.3f} "
          f"DD {metrics['dd_eval']:5.1f}% CAGR {metrics['cagr_eval']:+6.1f}% "
          f"turn {metrics['turnover_yr']:7.1f}")
