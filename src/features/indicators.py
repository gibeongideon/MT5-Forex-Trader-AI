"""
Composable technical indicator library — Phase 2.

Design rules:
  - Every function is pure: takes a pd.Series or pd.DataFrame, returns a Series.
  - No internal state. No lookahead (all rolling ops are shift-safe by default).
  - `compute(df, spec)` is the single entry point for feature engineering.

Usage:
    from src.features.indicators import compute, sma, rsi, atr

    # Add individual indicators to a DataFrame:
    df["sma_20"]  = sma(df["close"], 20)
    df["rsi_14"]  = rsi(df["close"], 14)
    df["atr_14"]  = atr(df, 14)

    # Or use the batch helper (returns enriched copy of df):
    df = compute(df, [
        ("sma_20",  sma,  {"period": 20}),
        ("ema_50",  ema,  {"period": 50}),
        ("rsi_14",  rsi,  {"period": 14}),
        ("atr_14",  atr,  {"period": 14}),
        # multi-output indicators: use a tuple of names
        (("macd", "macd_sig", "macd_hist"), macd,  {}),
        (("bb_u", "bb_m", "bb_l"),          bollinger_bands, {"period": 20}),
        (("stoch_k", "stoch_d"),             stochastic, {}),
    ])
"""

import numpy as np
import pandas as pd


# ─── Price-based indicators (take pd.Series of close prices) ─────────────────

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(
    series: pd.Series,
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist


def bollinger_bands(
    series:   pd.Series,
    period:   int   = 20,
    std_devs: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower)."""
    mid   = series.rolling(period, min_periods=period).mean()
    std   = series.rolling(period, min_periods=period).std()
    upper = mid + std_devs * std
    lower = mid - std_devs * std
    return upper, mid, lower


def bollinger_pct_b(series: pd.Series, period: int = 20, std_devs: float = 2.0) -> pd.Series:
    """Position of price within Bollinger Band: 0=lower, 0.5=mid, 1=upper."""
    upper, mid, lower = bollinger_bands(series, period, std_devs)
    return (series - lower) / (upper - lower).replace(0, np.nan)


# ─── OHLCV-based indicators (take full pd.DataFrame) ─────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename("atr")


def stochastic(
    df:       pd.DataFrame,
    k_period: int = 14,
    smooth_k: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Returns (%K smoothed, %D)."""
    low_min  = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k    = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k        = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d        = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (0–100)."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    overlap  = plus_dm < minus_dm
    plus_dm[overlap]  = 0
    minus_dm[~overlap] = 0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val  = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean()  / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().rename("adx")


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["tick_volume"]).cumsum().rename("obv")


# ─── Batch compute helper ─────────────────────────────────────────────────────

def compute(df: pd.DataFrame, spec: list) -> pd.DataFrame:
    """
    Add indicators to a copy of df in one call.

    spec items:
        (name,           fn, kwargs)   — single-output indicator
        ((n1, n2, ...),  fn, kwargs)   — multi-output indicator (returns tuple)

    Indicators that need only close prices receive df["close"].
    Indicators that need full OHLCV receive df directly.

    OHLCV indicators (detected automatically): atr, stochastic, adx, obv
    All others are treated as close-price indicators.
    """
    _OHLCV_FUNS = {atr, stochastic, adx, obv}

    result = df.copy()
    for item in spec:
        names, fn, kwargs = item
        arg = df if fn in _OHLCV_FUNS else df["close"]
        output = fn(arg, **kwargs)

        if isinstance(names, str):
            result[names] = output
        else:
            for col, series in zip(names, output):
                result[col] = series

    return result
