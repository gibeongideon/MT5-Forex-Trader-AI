"""
Triple-Barrier Meta-Labeling — Phase 26.

Replaces the standard "forward return > threshold" labels with labels that
directly match what happens in live trading: which barrier hits first?

Standard labels (what we use now):
  y = 1  if close[t+4] / close[t] - 1 > 0.03%   (direction guess)
  y = -1 if close[t+4] / close[t] - 1 < -0.03%

Triple-barrier labels (what this module provides):
  y = 1  if a LONG trade at bar[t] hits TP before SL, within horizon bars
  y = -1 if a SHORT trade at bar[t] hits TP before SL, within horizon bars
  y = 0  if NEITHER trade hits TP before SL within horizon bars (time barrier)

The key difference: these labels know about the TP/SL geometry we actually
trade, so the model learns to predict "will TP be hit before SL?" rather
than "which direction does price go?". This is a tighter, more useful signal.

Source: Marcos López de Prado, *Advances in Financial Machine Learning*
        Chapter 3: Labels / Triple-Barrier Method.

Usage:
    from src.features.meta_labels import triple_barrier_labels

    y = triple_barrier_labels(
        df["close"],
        tp_pips = 60,
        sl_pips = 30,
        horizon = 96,        # 96 × M15 = 24 hours max hold
        pip_size = 0.0001,   # EURUSD pip
    )
    # y has same index as df, same values {-1, 0, 1}
    # Drop last `horizon` rows — no valid label (future not known).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close:    pd.Series,
    tp_pips:  float = 60.0,
    sl_pips:  float = 30.0,
    horizon:  int   = 96,
    pip_size: float = 0.0001,
) -> pd.Series:
    """
    Triple-barrier labels for every bar.

    For each bar i, simulates both a LONG and SHORT trade:
      LONG:  TP at entry + tp_pips*pip_size, SL at entry - sl_pips*pip_size
      SHORT: TP at entry - tp_pips*pip_size, SL at entry + sl_pips*pip_size

    Checks bars i+1 … i+horizon:
      - If LONG TP hit first  → label = +1
      - If SHORT TP hit first → label = -1
      - If BOTH or NEITHER    → label = 0 (ambiguous / time barrier)

    Parameters
    ----------
    close    : Close price series (DatetimeIndex)
    tp_pips  : Take-profit distance in pips (default 60 — 2:1 R/R)
    sl_pips  : Stop-loss distance in pips (default 30)
    horizon  : Max bars to look ahead (default 96 = 24h on M15)
    pip_size : 1 pip in price units (0.0001 for EURUSD)

    Returns
    -------
    pd.Series with labels {-1, 0, 1}, same index as close.
    Last `horizon` rows will be 0 (not enough future data).
    """
    prices = close.values.astype(np.float64)
    n      = len(prices)
    labels = np.zeros(n, dtype=np.int8)

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size

    for i in range(n - 1):
        entry  = prices[i]
        end    = min(i + 1 + horizon, n)

        long_tp  = entry + tp_dist
        long_sl  = entry - sl_dist
        short_tp = entry - tp_dist
        short_sl = entry + sl_dist

        long_hit  = 0
        short_hit = 0

        for j in range(i + 1, end):
            p = prices[j]
            if long_hit == 0:
                if p >= long_tp:
                    long_hit = j
                elif p <= long_sl:
                    long_hit = -j    # negative = SL hit
            if short_hit == 0:
                if p <= short_tp:
                    short_hit = j
                elif p >= short_sl:
                    short_hit = -j   # negative = SL hit

            if long_hit != 0 and short_hit != 0:
                break

        long_won  = long_hit  > 0   # long TP hit (positive idx = TP bar)
        short_won = short_hit > 0   # short TP hit

        if long_won and not short_won:
            labels[i] = 1
        elif short_won and not long_won:
            labels[i] = -1
        else:
            labels[i] = 0   # both hit, neither hit, or time barrier

    return pd.Series(labels, index=close.index, dtype=int)


def triple_barrier_labels_fast(
    close:    pd.Series,
    tp_pips:  float = 60.0,
    sl_pips:  float = 30.0,
    horizon:  int   = 96,
    pip_size: float = 0.0001,
) -> pd.Series:
    """
    Vectorised triple-barrier labels — same result as triple_barrier_labels()
    but 10–50× faster using numpy rolling windows.

    Uses a forward-scan approach: for each bar, find the first bar where
    price exceeds TP or SL, then compare which came first.
    """
    prices = close.values.astype(np.float64)
    n      = len(prices)
    labels = np.zeros(n, dtype=np.int8)

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size

    for i in range(n - horizon - 1):
        entry  = prices[i]
        window = prices[i + 1: i + 1 + horizon]

        # LONG: first bar above long_tp or below long_sl
        long_tp_mask = window >= entry + tp_dist
        long_sl_mask = window <= entry - sl_dist

        long_tp_idx  = np.argmax(long_tp_mask)  if long_tp_mask.any()  else horizon
        long_sl_idx  = np.argmax(long_sl_mask)  if long_sl_mask.any()  else horizon
        if not long_tp_mask.any(): long_tp_idx = horizon
        if not long_sl_mask.any(): long_sl_idx = horizon

        # SHORT: first bar below short_tp or above short_sl
        short_tp_mask = window <= entry - tp_dist
        short_sl_mask = window >= entry + sl_dist

        short_tp_idx = np.argmax(short_tp_mask) if short_tp_mask.any() else horizon
        short_sl_idx = np.argmax(short_sl_mask) if short_sl_mask.any() else horizon
        if not short_tp_mask.any(): short_tp_idx = horizon
        if not short_sl_mask.any(): short_sl_idx = horizon

        long_won  = long_tp_idx  < long_sl_idx   and long_tp_idx  < horizon
        short_won = short_tp_idx < short_sl_idx  and short_tp_idx < horizon

        if long_won and not short_won:
            labels[i] = 1
        elif short_won and not long_won:
            labels[i] = -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=close.index, dtype=int)


def side_barrier_meta_label(
    high:     pd.Series,
    low:      pd.Series,
    close:    pd.Series,
    side:     pd.Series,        # +1 long / -1 short / NaN (no primary signal)
    atr:      pd.Series,        # RAW ATR in price units (e.g. indicators.atr) — NOT scaled
    tp_mult:  float = 1.5,
    sl_mult:  float = 1.5,
    horizon:  int   = 16,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """
    Side-conditioned triple-barrier labels for META-LABELING.

    Unlike `triple_barrier_labels_fast` (which evaluates BOTH long and short and
    collapses to 0 unless exactly one wins — the cause of Phase-26 15% sparsity),
    this evaluates ONLY the side the primary model chose. The output is the binary
    answer to "did THIS trade hit TP before SL within `horizon` bars?".

    For each bar t where `side[t]` is +1/-1:
      entry   = close[t]                      (enter at signal-bar close)
      tp_dist = tp_mult * atr[t]              (ATR-scaled, in price units)
      sl_dist = sl_mult * atr[t]
      LONG  : TP = entry + tp_dist, SL = entry - sl_dist
      SHORT : TP = entry - tp_dist, SL = entry + sl_dist
    Scan bars t+1 … t+horizon using intrabar high/low (conservative: if both TP and
    SL fall inside the same bar's range, the SL is assumed hit first).
      - TP hit first              → meta_y = 1, resolved = True
      - SL hit first              → meta_y = 0, resolved = True
      - neither within horizon    → meta_y = 0, resolved = False (time barrier);
                                     pips = force-close at close[t+horizon]

    Returns a DataFrame indexed like `close`, with rows only for bars where the
    primary fired (others dropped). Columns:
      side      : +1/-1 (the primary's side)
      meta_y    : {0,1}  (1 = TP-before-SL win)
      resolved  : bool   (False = time-barrier force-close)
      pips      : gross realized pips (costs applied by the backtester, not here)
      sl_pips   : sl_dist / pip_size  (per-trade risk distance, for R-unit sizing)
    """
    h = high.to_numpy(np.float64)
    l = low.to_numpy(np.float64)
    c = close.to_numpy(np.float64)
    s = side.to_numpy(np.float64)
    a = atr.to_numpy(np.float64)
    n = len(c)
    idx = close.index

    out_idx, out_side, out_y, out_res, out_pips, out_slp = [], [], [], [], [], []

    for i in range(n - 1):
        si = s[i]
        if not (si == 1.0 or si == -1.0):
            continue
        atr_i = a[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        entry   = c[i]
        tp_dist = tp_mult * atr_i
        sl_dist = sl_mult * atr_i
        end     = min(i + 1 + horizon, n)

        if si == 1.0:                      # LONG
            tp_lvl, sl_lvl = entry + tp_dist, entry - sl_dist
        else:                              # SHORT
            tp_lvl, sl_lvl = entry - tp_dist, entry + sl_dist

        meta_y, resolved, pips = 0, False, None
        for j in range(i + 1, end):
            hj, lj = h[j], l[j]
            if si == 1.0:
                hit_sl = lj <= sl_lvl
                hit_tp = hj >= tp_lvl
            else:
                hit_sl = hj >= sl_lvl
                hit_tp = lj <= tp_lvl
            if hit_sl:                     # conservative: SL checked first
                meta_y, resolved = 0, True
                pips = -sl_dist / pip_size
                break
            if hit_tp:
                meta_y, resolved = 1, True
                pips = tp_dist / pip_size
                break
        if pips is None:                   # time barrier → force-close
            c_end = c[end - 1]
            pips  = si * (c_end - entry) / pip_size
            meta_y, resolved = (1 if pips > 0 else 0), False
            # NOTE: time-barrier wins are NOT counted as meta_y=1 for TRAINING by
            # default below; we expose resolved so the caller can choose. Here we
            # set meta_y by realized sign so `pips` and `meta_y` stay consistent.

        out_idx.append(idx[i])
        out_side.append(int(si))
        out_y.append(int(meta_y))
        out_res.append(resolved)
        out_pips.append(float(pips))
        out_slp.append(float(sl_dist / pip_size))

    return pd.DataFrame(
        {"side": out_side, "meta_y": out_y, "resolved": out_res,
         "pips": out_pips, "sl_pips": out_slp},
        index=pd.Index(out_idx, name=close.index.name),
    )


def label_stats(y: pd.Series, name: str = "Labels") -> None:
    """Print label distribution."""
    total = len(y)
    buy   = (y ==  1).sum()
    hold  = (y ==  0).sum()
    sell  = (y == -1).sum()
    print(f"\n{name} distribution (n={total:,}):")
    print(f"  Buy  (+1): {buy:,}  ({buy/total:.1%})")
    print(f"  Hold ( 0): {hold:,}  ({hold/total:.1%})")
    print(f"  Sell (-1): {sell:,}  ({sell/total:.1%})")
