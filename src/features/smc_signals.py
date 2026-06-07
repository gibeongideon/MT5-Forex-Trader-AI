"""
SMC / ICT Signal Library — Phase 21.

Ported from trader_reference MQL5 indicators to pure Python/pandas.
All functions are pure (no state), backward-looking only (no lookahead),
and accept a full OHLCV DataFrame as input.

Signal functions registered as OHLCV-type — the pipeline passes the full
DataFrame (not just close) because they require open/high/low/close.

Sources:
  OrderBlock.mq5   — order_blocks(), fair_value_gaps()
  DailyHighLow.mq5 — prev_day_levels()
  AndeanOscillator.mq5 — andean_oscillator()
  SuperTrend.mq5   — supertrend()
  (custom)         — heiken_ashi()

Usage in comparison scripts:
    from src.features import feature_pipeline as _fp
    from src.features.smc_signals import (
        order_blocks, fair_value_gaps, prev_day_levels,
        andean_oscillator, supertrend, heiken_ashi,
    )
    _fp._OHLCV_FUNS |= {order_blocks, fair_value_gaps, prev_day_levels,
                         andean_oscillator, supertrend, heiken_ashi}

    SMC_SPEC = [
        (("ob_bull", "ob_bear"),                     order_blocks,     {}),
        (("fvg_bull", "fvg_bear", "fvg_size"),        fair_value_gaps,  {}),
        (("pdh_dist", "pdl_dist", "pd_pos"),          prev_day_levels,  {}),
    ]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _ohlcv(df: pd.DataFrame):
    """Return lower-cased O/H/L/C/V arrays as numpy float64."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    o = d["open"].values.astype(np.float64)
    h = d["high"].values.astype(np.float64)
    l = d["low"].values.astype(np.float64)
    c = d["close"].values.astype(np.float64)
    return o, h, l, c, d.index


def _atr14(h, l, c):
    """ATR(14) as numpy array, min_periods=1."""
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    return np.where(atr < 1e-8, 1e-8, atr)


# ── 1. Order Blocks ──────────────────────────────────────────────────────────

def order_blocks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Detect Order Block formation bars (ported from OrderBlock.mq5 OB_MODE_DEFAULT).

    Bullish OB fires at bar t when:
      bar[t-2] bearish, bar[t-1] bearish, bar[t] bullish,
      close[t-1] < low[t-2],  close[t] > high[t-1]

    Bearish OB fires at bar t when:
      bar[t-2] bullish, bar[t-1] bullish, bar[t] bearish,
      close[t-1] > high[t-2], close[t] < low[t-1]

    Returns
    -------
    ob_bull : pd.Series  (1 where bullish OB formed, else 0)
    ob_bear : pd.Series  (1 where bearish OB formed, else 0)
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)

    ob_bull = np.zeros(n, dtype=np.float32)
    ob_bear = np.zeros(n, dtype=np.float32)

    for i in range(2, n):
        bull_bar_t2 = c[i - 2] < o[i - 2]
        bull_bar_t1 = c[i - 1] < o[i - 1]
        bull_bar_t  = c[i]     > o[i]
        bull_confirm = c[i - 1] < l[i - 2] and c[i] > h[i - 1]
        if bull_bar_t2 and bull_bar_t1 and bull_bar_t and bull_confirm:
            ob_bull[i] = 1.0

        bear_bar_t2  = c[i - 2] > o[i - 2]
        bear_bar_t1  = c[i - 1] > o[i - 1]
        bear_bar_t   = c[i]     < o[i]
        bear_confirm = c[i - 1] > h[i - 2] and c[i] < l[i - 1]
        if bear_bar_t2 and bear_bar_t1 and bear_bar_t and bear_confirm:
            ob_bear[i] = 1.0

    return (
        pd.Series(ob_bull, index=idx, name="ob_bull"),
        pd.Series(ob_bear, index=idx, name="ob_bear"),
    )


# ── 2. Fair Value Gaps ───────────────────────────────────────────────────────

def fair_value_gaps(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect Fair Value Gap (FVG) bars (ported from OrderBlock.mq5 OB_MODE_FVG).

    Bullish FVG at bar t:
      bars t-3..t-2 bearish, bar t-1 bullish,
      close[t-2] < low[t-3], close[t-1] > high[t-2], low[t] > high[t-2]
      Gap = [high[t-2], low[t]]  (gap above t-2's high)

    Bearish FVG at bar t:
      bars t-3..t-2 bullish, bar t-1 bearish,
      close[t-2] > high[t-3], close[t-1] < low[t-2], high[t] < low[t-2]
      Gap = [high[t], low[t-2]]  (gap below t-2's low)

    Returns
    -------
    fvg_bull : 1 where bullish FVG detected
    fvg_bear : 1 where bearish FVG detected
    fvg_size : gap size in pips (EURUSD 4-decimal, 1 pip = 0.0001)
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)

    fvg_bull = np.zeros(n, dtype=np.float32)
    fvg_bear = np.zeros(n, dtype=np.float32)
    fvg_size = np.zeros(n, dtype=np.float32)

    for i in range(3, n):
        # Bullish FVG
        b_t3 = c[i - 3] < o[i - 3]
        b_t2 = c[i - 2] < o[i - 2]
        b_t1 = c[i - 1] > o[i - 1]
        b_confirm = (c[i - 2] < l[i - 3] and
                     c[i - 1] > h[i - 2] and
                     l[i]     > h[i - 2])
        if b_t3 and b_t2 and b_t1 and b_confirm:
            gap = l[i] - h[i - 2]
            fvg_bull[i] = 1.0
            fvg_size[i] = max(gap / 0.0001, 0.0)

        # Bearish FVG
        bea_t3 = c[i - 3] > o[i - 3]
        bea_t2 = c[i - 2] > o[i - 2]
        bea_t1 = c[i - 1] < o[i - 1]
        bea_confirm = (c[i - 2] > h[i - 3] and
                       c[i - 1] < l[i - 2] and
                       h[i]     < l[i - 2])
        if bea_t3 and bea_t2 and bea_t1 and bea_confirm:
            gap = l[i - 2] - h[i]
            fvg_bear[i] = 1.0
            fvg_size[i] = max(gap / 0.0001, 0.0)

    return (
        pd.Series(fvg_bull, index=idx, name="fvg_bull"),
        pd.Series(fvg_bear, index=idx, name="fvg_bear"),
        pd.Series(fvg_size, index=idx, name="fvg_size"),
    )


# ── 3. Previous Day High / Low ───────────────────────────────────────────────

def prev_day_levels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Previous day's high and low as reference levels (DailyHighLow.mq5).

    Requires DatetimeIndex. Resamples to D1, shifts 1 day (no lookahead),
    then forward-fills to M15.

    Returns (all ATR-normalised)
    -------
    pdh_dist : (close - prev_day_high) / ATR(14)   [+ = above prev high]
    pdl_dist : (close - prev_day_low)  / ATR(14)   [+ = above prev low]
    pd_pos   : position within prev day range [0–1], clipped
    """
    o, h, l, c, idx = _ohlcv(df)
    atr = _atr14(h, l, c)

    d = df.copy()
    d.columns = [col.lower() for col in d.columns]

    if not hasattr(d.index, "hour"):
        # Non-datetime index fallback — return zeros
        z = pd.Series(np.zeros(len(df)), index=idx)
        return z.rename("pdh_dist"), z.rename("pdl_dist"), z.rename("pd_pos")

    prev_high = (d["high"].resample("D").max()
                 .shift(1)
                 .reindex(d.index, method="ffill"))
    prev_low  = (d["low"].resample("D").min()
                 .shift(1)
                 .reindex(d.index, method="ffill"))

    close_s   = pd.Series(c, index=idx)
    atr_s     = pd.Series(atr, index=idx)
    day_range = (prev_high - prev_low).replace(0, np.nan)

    pdh_dist = ((close_s - prev_high) / atr_s).fillna(0).astype(np.float32)
    pdl_dist = ((close_s - prev_low)  / atr_s).fillna(0).astype(np.float32)
    pd_pos   = ((close_s - prev_low)  / day_range).clip(0, 1).fillna(0.5).astype(np.float32)

    return (
        pdh_dist.rename("pdh_dist"),
        pdl_dist.rename("pdl_dist"),
        pd_pos.rename("pd_pos"),
    )


# ── 4. Andean Oscillator ─────────────────────────────────────────────────────

def andean_oscillator(
    df: pd.DataFrame,
    length: int = 50,
    signal_length: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Andean Oscillator (AndeanOscillator.mq5).

    Decomposes momentum into separate Bull and Bear components using
    adaptive EMA-style tracking on close and close-squared.

    alpha = 2 / (length + 1)

    up1[i] = max(max(c, o), up1[i-1] - (up1[i-1] - c) * alpha)
    up2[i] = max(max(c^2, o^2), up2[i-1] - (up2[i-1] - c^2) * alpha)
    dn1[i] = min(min(c, o), dn1[i-1] + (c - dn1[i-1]) * alpha)
    dn2[i] = min(min(c^2, o^2), dn2[i-1] + (c^2 - dn2[i-1]) * alpha)
    bull = sqrt(max(dn2 - dn1^2, 0))
    bear = sqrt(max(up2 - up1^2, 0))
    signal = EMA(max(bull, bear), signal_length)

    Returns
    -------
    andean_bull   : bull component
    andean_bear   : bear component
    andean_signal : EMA of max(bull, bear)
    andean_diff   : bull - bear (positive = bullish momentum)
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)
    alpha = 2.0 / (length + 1)
    sig_alpha = 2.0 / (signal_length + 1)

    up1 = np.zeros(n)
    up2 = np.zeros(n)
    dn1 = np.zeros(n)
    dn2 = np.zeros(n)
    bull_arr = np.zeros(n)
    bear_arr = np.zeros(n)
    sig_arr  = np.zeros(n)

    up1[0] = c[0]; up2[0] = c[0] ** 2
    dn1[0] = c[0]; dn2[0] = c[0] ** 2

    for i in range(1, n):
        c2 = c[i] ** 2
        o2 = o[i] ** 2

        t_up = max(c[i], o[i])
        up1[i] = max(t_up, up1[i - 1] - (up1[i - 1] - c[i]) * alpha)

        t_up2 = max(c2, o2)
        up2[i] = max(t_up2, up2[i - 1] - (up2[i - 1] - c2) * alpha)

        t_dn = min(c[i], o[i])
        dn1[i] = min(t_dn, dn1[i - 1] + (c[i] - dn1[i - 1]) * alpha)

        t_dn2 = min(c2, o2)
        dn2[i] = min(t_dn2, dn2[i - 1] + (c2 - dn2[i - 1]) * alpha)

        bull_arr[i] = np.sqrt(max(dn2[i] - dn1[i] ** 2, 0.0))
        bear_arr[i] = np.sqrt(max(up2[i] - up1[i] ** 2, 0.0))

        mx = max(bull_arr[i], bear_arr[i])
        if i == 1:
            sig_arr[i] = mx
        else:
            sig_arr[i] = mx * sig_alpha + sig_arr[i - 1] * (1.0 - sig_alpha)

    diff_arr = bull_arr - bear_arr

    return (
        pd.Series(bull_arr.astype(np.float32), index=idx, name="andean_bull"),
        pd.Series(bear_arr.astype(np.float32), index=idx, name="andean_bear"),
        pd.Series(sig_arr.astype(np.float32),  index=idx, name="andean_signal"),
        pd.Series(diff_arr.astype(np.float32), index=idx, name="andean_diff"),
    )


# ── 5. SuperTrend ─────────────────────────────────────────────────────────────

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    SuperTrend indicator (SuperTrend.mq5, FxGeek 2011).

    middle[i] = (high + low) / 2
    upper[i]  = middle + multiplier * ATR(period)
    lower[i]  = middle - multiplier * ATR(period)

    trend[i] = +1 if close > upper[i-1]
               -1 if close < lower[i-1]
               else trend[i-1]

    Returns
    -------
    supertrend_dir  : +1 (bullish) or -1 (bearish)
    supertrend_dist : (close - active_band) / ATR  [+ = price far from band in trend direction]
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)

    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    atr = np.where(atr < 1e-8, 1e-8, atr)

    mid   = (h + l) / 2.0
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    trend = np.ones(n, dtype=np.float32)
    dist  = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        if c[i] > upper[i - 1]:
            trend[i] = 1.0
        elif c[i] < lower[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]

        active_band = lower[i] if trend[i] == 1.0 else upper[i]
        dist[i] = float((c[i] - active_band) / atr[i])

    return (
        pd.Series(trend, index=idx, name="supertrend_dir"),
        pd.Series(dist,  index=idx, name="supertrend_dist"),
    )


# ── 6. Heiken Ashi ───────────────────────────────────────────────────────────

def heiken_ashi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Heiken Ashi smoothed candle direction and body size.

    ha_close[i] = (open + high + low + close) / 4
    ha_open[i]  = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_dir      = sign(ha_close - ha_open)  [+1 bullish, -1 bearish]
    ha_body_norm = |ha_close - ha_open| / ATR(14)

    Returns
    -------
    ha_dir       : +1 or -1
    ha_body_norm : body size normalised by ATR
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)
    atr = _atr14(h, l, c)

    ha_close = (o + h + l + c) / 4.0
    ha_open  = np.zeros(n)
    ha_open[0] = (o[0] + c[0]) / 2.0

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    body = ha_close - ha_open
    ha_dir  = np.sign(body).astype(np.float32)
    ha_body = (np.abs(body) / atr).astype(np.float32)

    return (
        pd.Series(ha_dir,  index=idx, name="ha_dir"),
        pd.Series(ha_body, index=idx, name="ha_body_norm"),
    )
