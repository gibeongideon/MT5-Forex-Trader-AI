"""Signal definitions for the two H4 XAUUSD bots deployed 2026-07-14.

Math is copied VERBATIM from the validated research lab
(data/v5_runs/xau-sharpe1-lab/xau_lab.py, campaigns 1-10) so the live
signal is byte-identical to the backtested one. Both are lookahead-free:
EWMAs/rolling stats on closes <= t, expanding past-only scalars, shift(1).

Deployed variants (discrete engine src.v5.xau_trend.run_trades, exit=trail,
flip=confidence, sl_atr=trail_atr=3.0, conf_risk_scale {0.5,1.0,1.5},
eval 2017+, net of half-spread + $0.10 slippage):

  ls_signal        long/short 0.7*EWMAC-mid + 0.3*breakout-fast
                   Sharpe 0.807  CI[0.22,1.53]  maxDD -14.6%  CAGR +8.8%
  champion_signal  LONG-ONLY 0.5*conc(max(ewmac_mid+, bko_fast+)) +
                   0.5*conc(bko_fast+) + 0.15 resting tilt, conc = ^1.5
                   Sharpe 0.968  CI[0.33,1.61]  maxDD -10.0%  CAGR +9.6%

Research context: continuous-engine champion reached eval 1.041 (stress-
robust, costx3 1.01, delay2 1.02); the discrete ports above are what the
existing trade-ticket executor can implement. Full log:
data/v5_runs/XAU_SHARPE1_RESEARCH.md and xau-sharpe1-lab/results.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

D = 6  # H4 bars per trading day

EWMAC_MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
BKO_FAST = tuple(d * D for d in (10, 20, 40))


def ewmac_fc(close: pd.Series, speeds, cap: float = 2.0) -> pd.Series:
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


def breakout_fc(close: pd.Series, windows, cap: float = 2.0) -> pd.Series:
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


def _norm(s: pd.Series) -> pd.Series:
    """Past-only rescale to unit mean |forecast|."""
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def _conc(s: pd.Series, p: float) -> pd.Series:
    """Long-only forecast concentration: normalized s^p on the positive part."""
    return _norm(s.clip(lower=0.0) ** p)


def ls_signal(close: pd.Series) -> pd.Series:
    """Long/short trend+breakout ensemble (best LS book of the 2026-07-14 hunt)."""
    base = ewmac_fc(close, EWMAC_MID)
    bko = breakout_fc(close, BKO_FAST)
    return (0.7 * base + 0.3 * bko).clip(-2, 2)


def champion_signal(close: pd.Series) -> pd.Series:
    """LONG-ONLY concentrated blend — the Sharpe>=1 champion's discrete port."""
    base = ewmac_fc(close, EWMAC_MID)
    bko = breakout_fc(close, BKO_FAST)
    maxewbko = np.maximum(base.clip(lower=0.0), bko.clip(lower=0.0))
    return (0.5 * (_conc(maxewbko, 1.5) * 0.8 + 0.15)
            + 0.5 * (_conc(bko, 1.5) * 0.8 + 0.15)).clip(0, 2)


SIGNALS = {"ls": ls_signal, "champ": champion_signal}
