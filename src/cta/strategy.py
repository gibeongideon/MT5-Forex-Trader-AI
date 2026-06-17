"""strategy.py — the locked champion position construction, single source of truth.

Both the backtester (`scripts/cta_backtest.py`) and the live runner
(`scripts/basket_runner.py`) build target positions through `champion_positions` so the
deployed signal is byte-identical to the validated backtest.

Locked config (data/SMALL_BASKET.md, 2026-06-17):
  instruments = GOLD, UST10Y, SPX, WTI, EURUSD   (one per asset class)
  sleeve=combined (EWMAC slow + xsmom), risk=cluster, target_vol=10%,
  rebalance=monthly, buffer=0.4, trend_speeds=slow
  → FULL net Sharpe +0.746, CI [+0.34,+1.15], ~0 beta, 18.8% maxDD, turnover 683%/yr.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.cta.signals import ewmac, xsmom, combine
from src.cta.portfolio import cluster_risk_weights, inv_vol_weights, vol_target

# EWMAC speed sets (fast,slow span pairs). "slow" is the locked production set.
TREND_SPEEDS = {
    "fast":    ((8, 32), (16, 64), (32, 128), (64, 256)),
    "slow":    ((32, 128), (64, 256)),
    "slowest": ((64, 256), (128, 512)),
}

# Locked production basket + config
BASKET = ["GOLD", "UST10Y", "SPX", "WTI", "EURUSD"]
CONFIG = dict(target_vol=0.10, trend_speeds="slow", risk="cluster",
              rebalance="monthly", buffer=0.4)


def rebalance_hold(pos: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Hold positions between rebalances to cut turnover (lookahead-free; pnl applies
    pos.shift(1) downstream). Exactly mirrors the validated backtester."""
    if freq == "daily":
        return pos
    rule = {"weekly": "W", "monthly": "ME"}[freq]
    return pos.resample(rule).last().reindex(pos.index, method="ffill")


def buffer_band(pos: pd.DataFrame, frac: float) -> pd.DataFrame:
    """No-trade band: move toward target only when it deviates from the held position by
    > frac × (avg |position|) PER INSTRUMENT. Path-dependent but lookahead-free."""
    if frac <= 0:
        return pos
    avg = pos.abs().replace(0.0, np.nan).mean().fillna(0.0)
    bufv = (frac * avg).values
    arr = pos.values
    out = np.zeros_like(arr)
    prev = np.zeros(arr.shape[1])
    for r in range(arr.shape[0]):
        tgt = arr[r]
        keep = np.abs(tgt - prev) <= bufv
        prev = np.where(keep, prev, tgt)
        out[r] = prev
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def champion_positions(close: pd.DataFrame, returns: pd.DataFrame, classes: dict,
                       target_vol=0.10, trend_speeds="slow", risk="cluster",
                       rebalance="monthly", buffer=0.4) -> pd.DataFrame:
    """Build the full target-position panel for the champion 'combined' sleeve.

    combined = EWMAC(slow) trend + cross-sectional momentum → cluster-equal risk →
    portfolio vol-target → calendar rebalance → no-trade buffer. Lookahead-free.
    """
    trend = ewmac(close, speeds=TREND_SPEEDS[trend_speeds])
    mom = combine(trend, xsmom(close))
    raw = (cluster_risk_weights(mom, returns, classes, target_vol) if risk == "cluster"
           else inv_vol_weights(mom, returns, target_vol))
    pos = vol_target(raw, returns, target=target_vol)
    pos = rebalance_hold(pos, rebalance)
    pos = buffer_band(pos, buffer)
    return pos
