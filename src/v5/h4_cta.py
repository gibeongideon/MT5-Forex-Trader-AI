"""V5 H4 CTA strategy — vol-targeted EWMAC trend portfolio on 4H bars.

Pre-registered configuration (declared before any backtest was run; do NOT
sweep-then-best-report):

  universe      : EURUSD, GBPUSD, USDJPY, XAUUSD (H4 close-to-close)
  signal        : EWMAC, module-default Carver speeds scaled to H4 bars
                  (daily (8,32),(16,64),(32,128),(64,256) x 6 bars/day)
  sizing        : cluster-equal risk budget across {fx, metal}, then
                  inverse-vol within class; portfolio vol target 10%/yr
  execution     : position formed on close of bar t earns bar t+1 return
                  (positions.shift(1)); no calendar rebalance; causal
                  no-trade buffer band 0.10
  costs         : FULL per-bar spread (floored at symbol median) per unit
                  turnover — conservative, ~2x a half-spread fill

Everything is past-only: EWMAC uses shift(1) expanding scalars, all vol
estimates use returns.shift(1), the buffer band width uses an expanding
mean with shift(1). There are no fitted components, so there is no
train/test boundary to leak across; the leakage surface is lookahead only
and is covered by tests/test_v5_h4_cta.py causality checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cta.signals import ewmac

SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")
CLASSES = {"EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx", "XAUUSD": "metal"}
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}

# 6 H4 bars per trading day, ~260 trading days per year.
BARS_PER_DAY = 6
PERIODS_PER_YEAR = BARS_PER_DAY * 260
ANN = np.sqrt(PERIODS_PER_YEAR)

# Daily Carver speeds (8,32)...(64,256) expressed in H4 bars.
H4_SPEEDS = tuple((f * BARS_PER_DAY, s * BARS_PER_DAY)
                  for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)))

CONFIG = dict(
    target_vol=0.10,
    vol_halflife=42 * BARS_PER_DAY,
    buffer_frac=0.10,
    entry_delay_bars=1,
    spread_cost_mult=1.0,
)


def load_h4_panel(data_dir: str | Path = "data", symbols=SYMBOLS):
    """Load aligned H4 close and spread panels from <data_dir>/<SYM>_H4_long.csv.

    Closes are outer-joined and forward-filled (a stale close produces a zero
    return, never a future value). Per-bar spreads are floored at the symbol
    median so zero-spread export artifacts cannot make trading look free.
    """
    closes, spreads = {}, {}
    for sym in symbols:
        df = pd.read_csv(Path(data_dir) / f"{sym}_H4_long.csv",
                         parse_dates=["time"], index_col="time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        closes[sym] = df["close"]
        spreads[sym] = df["spread"].clip(lower=df["spread"].median())
    close = pd.DataFrame(closes).ffill()
    spread = pd.DataFrame(spreads).ffill()
    return close, spread


def cluster_inv_vol(signals: pd.DataFrame, returns: pd.DataFrame, classes: dict,
                    target: float, halflife: int) -> pd.DataFrame:
    """Cluster-equal risk budget, inverse-vol within class. Past-only (shift(1)).

    H4-aware port of src.cta.portfolio.cluster_risk_weights (that module
    hardcodes daily annualization and stays untouched as the locked daily
    champion).
    """
    sigma = returns.shift(1).ewm(halflife=halflife, min_periods=120).std()
    active = signals.replace(0.0, np.nan).notna() & sigma.notna()
    uniq = sorted(set(classes.values()))
    kc = {c: active[[a for a in signals.columns if classes.get(a) == c]].sum(axis=1)
          for c in uniq}
    n_classes = sum((kc[c] > 0).astype(int) for c in uniq).clip(lower=1)
    budget = pd.DataFrame(index=signals.index, columns=signals.columns, dtype=float)
    for a in signals.columns:
        k = kc[classes[a]].replace(0, np.nan)
        budget[a] = (target / ANN) / (np.sqrt(n_classes) * k)
    pos = signals.mul(budget) / sigma
    return pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def vol_target_h4(positions: pd.DataFrame, returns: pd.DataFrame,
                  target: float, halflife: int, max_lev: float = 3.0) -> pd.DataFrame:
    """Scale the book so trailing realized portfolio vol ~= target. Past-only."""
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    realized = port_ret.ewm(halflife=halflife, min_periods=120).std().shift(1) * ANN
    k = (target / realized).clip(upper=max_lev)
    k = k.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return positions.mul(k, axis=0)


def buffer_band_causal(pos: pd.DataFrame, frac: float,
                       min_periods: int = 120) -> pd.DataFrame:
    """No-trade band like src.cta.strategy.buffer_band, but the band width is an
    EXPANDING past-only mean |position| (shift(1)) instead of a full-sample mean,
    so the band itself cannot see the future."""
    if frac <= 0:
        return pos
    scale = pos.abs().replace(0.0, np.nan).expanding(min_periods=min_periods).mean().shift(1)
    band = (frac * scale).fillna(0.0).values
    arr = pos.values
    out = np.zeros_like(arr)
    prev = np.zeros(arr.shape[1])
    for r in range(arr.shape[0]):
        tgt = arr[r]
        keep = np.abs(tgt - prev) <= band[r]
        prev = np.where(keep, prev, tgt)
        out[r] = prev
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def h4_positions(close: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Full pre-registered position construction. Lookahead-free."""
    cfg = {**CONFIG, **(config or {})}
    returns = close.pct_change(fill_method=None)
    trend = ewmac(close, speeds=H4_SPEEDS)
    raw = cluster_inv_vol(trend, returns, CLASSES,
                          cfg["target_vol"], cfg["vol_halflife"])
    pos = vol_target_h4(raw, returns, cfg["target_vol"], cfg["vol_halflife"])
    return buffer_band_causal(pos, cfg["buffer_frac"])


def h4_pnl(positions: pd.DataFrame, close: pd.DataFrame, spread: pd.DataFrame,
           entry_delay_bars: int = 1, spread_cost_mult: float = 1.0) -> pd.DataFrame:
    """Mark-to-market P&L with turnover costs.

    Position formed at close t is held over the next bar's return via
    shift(entry_delay_bars) — the single anti-lookahead line. Cost per unit
    turnover = full spread (in price units) / price, times spread_cost_mult.
    """
    if entry_delay_bars < 1:
        raise ValueError("entry_delay_bars must be >= 1 to avoid lookahead")
    returns = close.pct_change(fill_method=None)
    pos_lag = positions.shift(entry_delay_bars)
    gross = (pos_lag * returns).sum(axis=1)
    turnover = (positions - positions.shift(1)).abs()
    pips = pd.Series(PIP_SIZE).reindex(positions.columns)
    spread_frac = spread.mul(pips, axis=1) / close * spread_cost_mult
    cost = (turnover * spread_frac).sum(axis=1)
    per_symbol_net = pos_lag * returns - turnover * spread_frac
    out = pd.DataFrame({"gross": gross, "net": gross - cost,
                        "cost": cost, "turnover": turnover.sum(axis=1)})
    return pd.concat([out, per_symbol_net.add_prefix("net_")], axis=1)
