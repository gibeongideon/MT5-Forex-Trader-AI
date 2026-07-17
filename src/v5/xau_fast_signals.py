"""Signal for the FAST intraday XAUUSD trend runner (magic 360543).

The slow H4 champion (src/v5/xau_dual_signals.champion_signal) recipe —
LONG-ONLY, concentrated max(EWMAC, breakout) blend — re-tuned to INTRADAY
speeds on M30 bars. Higher activity (~13-19 trades/mo vs the champion's 3-4)
by trading a faster trend, gated to high-conviction entries only.

Research (data/v5_runs/fast-trend/, 2026-07-17): fast trend is a real gross
edge but SPREAD-GATED. Net Sharpe vs account spread (M30, conviction-selected,
eval 2017+):  $0.10 -> 1.07 | $0.14 -> 1.04 | $0.24 -> 0.91 (= champion) |
$0.36 (cent) -> 0.75. Uniform lot sizing is Sharpe-neutral; the only lever is
conviction concentration + trade selection (skip weak trades that don't clear
the spread). => deploy ONLY on a raw/ECN gold account with spread <= ~$0.24.

Math is causal (EWMA/rolling on closes<=t, expanding scalars shift(1)), same
construction as xau_dual_signals so the live signal is byte-identical to the
backtested one (scripts/v5_xau_fast_trend_discrete.make_fast_champion, M30/fast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# M30 bars/day ~ 48. "fast" speed set (in BARS):
EWMAC_FAST = ((16, 64), (32, 128))
BKO_FAST = (48, 96)                     # ~1 day, ~2 days of M30 bars


def _ewmac_fc(close: pd.Series, pairs, cap: float = 2.0) -> pd.Series:
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in pairs:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(pairs)).clip(-cap, cap)


def _breakout_fc(close: pd.Series, windows, cap: float = 2.0) -> pd.Series:
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
    return (combined / len(windows)).clip(-cap, cap)


def _norm(s: pd.Series) -> pd.Series:
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def fast_champion_signal(close: pd.Series) -> pd.Series:
    """LONG-ONLY concentrated intraday trend forecast in [0, 2]."""
    ew = _ewmac_fc(close, EWMAC_FAST)
    bk = _breakout_fc(close, BKO_FAST)
    mx = np.maximum(ew.clip(lower=0.0), bk.clip(lower=0.0))
    return (0.5 * (_norm(mx.clip(lower=0.0) ** 1.5) * 0.8 + 0.15)
            + 0.5 * (_norm(bk.clip(lower=0.0) ** 1.5) * 0.8 + 0.15)).clip(0, 2)


SIGNALS = {"fast": fast_champion_signal}
