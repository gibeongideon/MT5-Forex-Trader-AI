"""
Cross-market feature generator — Phase 24.

Loads external instrument M15 CSVs and merges pre-shifted signals onto the
EURUSD DataFrame. All features are shifted 1 bar so at bar[t] we see
bar[t-1] cross-market data — zero lookahead.

Why this helps (saturation principle):
  enc8 has absorbed all information in EURUSD OHLCV. New signal must come
  from a different instrument. EURUSD moves are driven by:
    DXY / USDJPY  → USD strength (not visible in EURUSD candles alone)
    GBPUSD        → correlated institutional flow
    XAUUSD        → risk-off / safe-haven dynamic

Features added per instrument:
  {sym}_return_1  — 1-bar % return (directional momentum, shift=1)
  {sym}_rsi_14    — RSI(14), 0–100 normalised (overbought/oversold, shift=1)
  {sym}_atr_ratio — ATR / 100-bar rolling mean ATR (volatility regime, shift=1)

Usage:
    from src.features.cross_market import add_cross_market_cols

    # Add features in-place before passing df to FeaturePipeline:
    df_aug = add_cross_market_cols(df_eurusd, data_dir="data/",
                                   symbols=["GBPUSD", "USDJPY", "XAUUSD"])

    # FeaturePipeline keeps these columns because they're not in its
    # standard drop list, and they're already shifted so no extra shift needed.
    X, y = pipeline.build(df_aug, fit=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()
    return df


def _rsi14(series: pd.Series) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_ratio(df: pd.DataFrame, period: int = 14, roll: int = 100) -> pd.Series:
    if "high" not in df.columns or "low" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tr = pd.DataFrame({
        "hl":  df["high"] - df["low"],
        "hpc": (df["high"] - df["close"].shift(1)).abs(),
        "lpc": (df["low"]  - df["close"].shift(1)).abs(),
    }).max(axis=1)
    atr      = tr.rolling(period, min_periods=1).mean()
    roll_avg = atr.rolling(roll, min_periods=10).mean()
    return (atr / roll_avg.replace(0, np.nan)).clip(0, 5)


def add_cross_market_cols(
    df_eurusd: pd.DataFrame,
    data_dir:  str | Path = "data/",
    symbols:   Sequence[str] = ("GBPUSD", "USDJPY", "XAUUSD"),
    timeframe: str = "M15",
    features:  Sequence[str] = ("return_1", "rsi_14", "atr_ratio"),
) -> pd.DataFrame:
    """
    Merge cross-market features onto df_eurusd, pre-shifted by 1 bar.

    Parameters
    ----------
    df_eurusd : EURUSD M15 DataFrame (DatetimeIndex)
    data_dir  : directory containing {SYMBOL}_M15.csv files
    symbols   : list of instrument symbols to load
    timeframe : timeframe suffix of CSV files (default "M15")
    features  : which features to add per instrument

    Returns
    -------
    Augmented copy of df_eurusd with additional columns:
        {sym}_return_1, {sym}_rsi_14, {sym}_atr_ratio
    """
    result   = df_eurusd.copy()
    data_dir = Path(data_dir)
    loaded   = 0

    for sym in symbols:
        path = data_dir / f"{sym}_{timeframe}.csv"
        if not path.exists():
            print(f"  [CrossMarket] {path.name} not found — skipping {sym}. "
                  f"Provide the required cross-market CSV first.")
            continue

        try:
            cm = _load_csv(path)
        except Exception as e:
            print(f"  [CrossMarket] Failed to load {path.name}: {e}")
            continue

        close = cm["close"].astype(float)
        sym_l = sym.lower()

        if "return_1" in features:
            ret = close.pct_change(1).shift(1)   # shift=1: no lookahead
            result[f"{sym_l}_return_1"] = ret.reindex(result.index, method="ffill")

        if "rsi_14" in features:
            r14 = _rsi14(close).shift(1)
            result[f"{sym_l}_rsi_14"] = r14.reindex(result.index, method="ffill")

        if "atr_ratio" in features:
            ar = _atr_ratio(cm).shift(1)
            result[f"{sym_l}_atr_ratio"] = ar.reindex(result.index, method="ffill")

        loaded += 1
        n_feat = sum(1 for f in features)
        print(f"  [CrossMarket] {sym}: loaded {len(cm):,} bars "
              f"({cm.index[0].date()} → {cm.index[-1].date()}) "
              f"→ {n_feat} features added")

    if loaded == 0:
        print("  [CrossMarket] WARNING: No cross-market CSVs found. "
              "Provide the required cross-market CSV first.")
    else:
        added = [c for c in result.columns if c not in df_eurusd.columns]
        print(f"  [CrossMarket] Total new columns: {len(added)}  ({', '.join(added)})")

    return result


def available_symbols(data_dir: str | Path = "data/", timeframe: str = "M15") -> list[str]:
    """Return list of cross-market symbols with available CSV files."""
    data_dir = Path(data_dir)
    found = []
    for sym in ("GBPUSD", "USDJPY", "XAUUSD", "EURGBP", "USDCHF"):
        if (data_dir / f"{sym}_{timeframe}.csv").exists():
            found.append(sym)
    return found
