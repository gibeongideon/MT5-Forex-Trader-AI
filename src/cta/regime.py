"""Regime filters — lookahead-free gates applied to the SIGNAL panel before sizing.

A gate scales/zeroes signals based on market state; risk-budgeting then re-normalises across
the survivors and vol_target still pins portfolio vol. Every rolling stat reads close/returns
≤ t-1 (shift(1)) so the gate at day t never peeks at day t — the classic CTA backtest trap.

  trend : keep a position only if its sign agrees with the slow (200d SMA) trend.
  vol   : taper position size toward `floor` when an instrument's trailing vol is in an
          extreme upper percentile (crisis de-risk).
  none  : identity (regression guard — must reproduce the locked champion).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def trend_gate(close: pd.DataFrame, signals: pd.DataFrame, sma_window: int = 200) -> pd.DataFrame:
    """{0,1} gate: zero a signal whose sign fights the long trend (price vs 200d SMA).
    No gating during the SMA warmup. Past-only (close.shift(1))."""
    c = close.shift(1)
    sma = c.rolling(sma_window, min_periods=sma_window).mean()
    trend_sign = np.sign(c - sma)
    agree = (np.sign(signals) * trend_sign) >= 0            # aligned, or signal/trend flat
    mask = agree.astype(float).where(trend_sign.notna(), 1.0)
    return signals * mask


def vol_gate(returns: pd.DataFrame, halflife: int = 42, pct_window: int = 252,
             hi: float = 0.90, floor: float = 0.0) -> pd.DataFrame:
    """Multiplier in [floor,1]: 1 while trailing vol percentile < `hi`, then taper linearly to
    `floor` at the 100th percentile (crisis de-risk). Same EWM-std def as sizing. Past-only."""
    sigma = returns.shift(1).ewm(halflife=halflife, min_periods=20).std()
    pct = sigma.rolling(pct_window, min_periods=60).rank(pct=True)   # causal rank of current sigma
    taper = ((1.0 - pct) / (1.0 - hi)).clip(lower=floor, upper=1.0)
    mult = taper.where(pct >= hi, 1.0)                      # full size below the hi percentile
    return mult.where(pct.notna(), 1.0)


def regime_gate(close: pd.DataFrame, returns: pd.DataFrame, signals: pd.DataFrame,
                mode: str = "none") -> pd.DataFrame:
    """Dispatch the regime filter. mode='none' returns signals unchanged (byte-identical)."""
    if mode == "none":
        return signals
    if mode == "trend":
        return trend_gate(close, signals)
    if mode == "vol":
        return signals * vol_gate(returns)
    if mode == "trend_vol":
        return trend_gate(close, signals) * vol_gate(returns)
    raise ValueError(f"unknown regime mode '{mode}' (none|trend|vol|trend_vol)")
