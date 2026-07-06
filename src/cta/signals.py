"""Momentum signals — pure functions, lookahead-free (use close ≤ t only).

Pre-registered lookbacks; do NOT sweep-then-best-report.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def tsmom(close: pd.DataFrame, lookbacks=(21, 63, 126, 252)) -> pd.DataFrame:
    """Time-series momentum: mean of sign(trailing return) across lookbacks ∈ [-1,1]."""
    sigs = [np.sign(close / close.shift(L) - 1.0) for L in lookbacks]
    return sum(sigs) / len(sigs)


def xsmom(close: pd.DataFrame, lookback: int = 252, skip: int = 21,
          q: float = 1/3) -> pd.DataFrame:
    """Cross-sectional momentum: trailing 12m return skipping last 1m; long top /
    short bottom q-tercile each day. Returns +1/0/-1 per instrument."""
    r = close.shift(skip) / close.shift(lookback) - 1.0

    def _rank(row: pd.Series) -> pd.Series:
        v = row.dropna()
        out = pd.Series(0.0, index=row.index)
        if len(v) < 3:
            return out
        lo, hi = v.quantile(q), v.quantile(1 - q)
        out[row >= hi] = 1.0
        out[row <= lo] = -1.0
        return out

    return r.apply(_rank, axis=1)


def ewmac(close: pd.DataFrame, speeds=((8, 32), (16, 64), (32, 128), (64, 256)),
          cap: float = 20.0, target: float = 10.0, fdm: float = 1.3) -> pd.DataFrame:
    """Continuous trend-strength forecast (Carver EWMAC), lookahead-free.

    Per speed (fast,slow): raw = (EWMA_fast − EWMA_slow) / price_vol, where
    price_vol = close * ewm_std(returns, span=36) (price-unit daily vol). Each speed's
    raw is scaled to mean |forecast| ≈ `target` using a PAST-ONLY adaptive scalar
    (expanding mean-abs, shifted), capped ±cap; speeds averaged × fdm, capped ±cap.
    Returns the forecast/target panel (|1.0| ≈ an average-size position) → fed to sizing.
    """
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in speeds:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        # past-only scalar so long-run mean |forecast| ≈ target
        scalar = target / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap, cap)
        combined = fc if combined is None else combined + fc
    combined = (combined / len(speeds)) * fdm
    return combined.clip(-cap, cap) / target


def combine(a: pd.DataFrame, b: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
    """Fixed-weight blend of two signal panels (no weight tuning)."""
    return w * a + (1 - w) * b


def fx_carry(index, rates: pd.DataFrame, pairs: dict, all_aliases) -> pd.DataFrame:
    """FX carry signal: sign(rate_base - rate_quote) per pair, +1/0/-1.
    Monthly rates forward-filled to daily (lookahead-free: a month's rate is known
    at month start). Non-FX instruments get 0 (carry doesn't apply)."""
    r = rates.reindex(index, method="ffill")
    out = pd.DataFrame(0.0, index=index, columns=list(all_aliases))
    for alias, (base, quote) in pairs.items():
        if alias in out.columns and base in r.columns and quote in r.columns:
            out[alias] = np.sign(r[base] - r[quote])
    return out
