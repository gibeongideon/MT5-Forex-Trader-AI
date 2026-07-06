"""trend_labels.py — turning-point / trend-direction labels (TARGETS ONLY).

These define the *current trend direction* (up/down) so a classifier can learn to flag the
turning point ("the tip"). They are intentionally FORWARD-LOOKING (a label at bar t uses bars
after t) — that is correct for a training target. The model's FEATURES must stay strictly
past-only; never feed these series, or anything derived from future bars, as features.

  trend_scan_labels — López de Prado trend-scanning: label each bar by the sign of the most
                      statistically-significant forward linear trend.
  zigzag_labels     — ATR-pivot swings: a new pivot when price reverses >= k*ATR; segments
                      between pivots are labelled by their direction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def trend_scan_labels(close: pd.Series, h_min: int = 4, h_max: int = 24,
                      t_thresh: float = 2.0) -> pd.Series:
    """Trend-scanning labels in {-1, 0, +1}.

    For each bar t, fit OLS  price ~ a + b·(0..h)  over each forward horizon h in [h_min, h_max];
    keep the horizon whose slope has the largest |t-stat|. Label = sign(slope) if |t-stat| >=
    t_thresh else 0 (no clear trend). Near a reversal the new direction's label appears at the
    pivot, which is exactly the "enter at the tip" signal.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    out = np.zeros(n)
    # precompute x stats per horizon (x = 0..h-1)
    for t in range(n):
        best_abs_t, best_sign = t_thresh, 0
        hi = min(h_max, n - 1 - t)
        for h in range(h_min, hi + 1):
            y = c[t:t + h + 1]
            m = len(y)
            if m < 3:
                continue
            x = np.arange(m, dtype=float)
            xm = x.mean()
            sxx = ((x - xm) ** 2).sum()
            if sxx <= 0:
                continue
            ym = y.mean()
            b = ((x - xm) * (y - ym)).sum() / sxx          # slope
            resid = y - (ym + b * (x - xm))
            dof = m - 2
            if dof <= 0:
                continue
            se = np.sqrt((resid ** 2).sum() / dof / sxx)    # SE(slope)
            if se <= 0:
                continue
            tval = b / se
            if abs(tval) >= best_abs_t:
                best_abs_t = abs(tval)
                best_sign = int(np.sign(b))
        out[t] = best_sign
    return pd.Series(out, index=close.index, name="trend")


def zigzag_labels(high: pd.Series, low: pd.Series, close: pd.Series,
                  atr: pd.Series, k: float = 3.0) -> pd.Series:
    """ATR-pivot zigzag labels in {-1, +1}.

    Walk forward tracking the running extreme since the last confirmed pivot. A reversal of
    >= k*ATR (ATR measured at the pivot) confirms a new pivot; the segment from the prior pivot
    to here is labelled by its direction (+1 up, -1 down). The unconfirmed tail copies the last
    confirmed direction. Forward-looking (target only)."""
    h = np.asarray(high, dtype=float); lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float); a = np.asarray(atr, dtype=float)
    n = len(close)
    out = np.zeros(n)
    if n == 0:
        return pd.Series(out, index=close.index, name="zz")
    a_mean = float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else 0.0

    last_pivot = 0
    hi_i = lo_i = 0                          # running extremes since last pivot
    direction = 0                           # +1 up-leg, -1 down-leg, 0 unknown
    for i in range(1, n):
        if h[i] >= h[hi_i]: hi_i = i
        if lo[i] <= lo[lo_i]: lo_i = i
        thr = k * (a[i] if np.isfinite(a[i]) and a[i] > 0 else a_mean)
        if thr <= 0:
            continue
        if direction >= 0 and h[hi_i] - lo[i] >= thr:       # drop from running high → pivot at hi_i
            if hi_i > last_pivot:
                out[last_pivot:hi_i + 1] = float(np.sign(c[hi_i] - c[last_pivot])) or 1.0
            last_pivot = hi_i; direction = -1; hi_i = lo_i = i
        elif direction <= 0 and h[i] - lo[lo_i] >= thr:     # rise from running low → pivot at lo_i
            if lo_i > last_pivot:
                out[last_pivot:lo_i + 1] = float(np.sign(c[lo_i] - c[last_pivot])) or -1.0
            last_pivot = lo_i; direction = 1; hi_i = lo_i = i
    # tail (unconfirmed leg) by its actual direction
    out[last_pivot:] = float(np.sign(c[-1] - c[last_pivot])) or (direction if direction != 0 else 1.0)
    return pd.Series(out, index=close.index, name="zz")
