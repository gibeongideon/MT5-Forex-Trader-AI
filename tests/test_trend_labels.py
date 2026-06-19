"""Tests for src/features/trend_labels.py — turning-point / trend-direction labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.trend_labels import trend_scan_labels, zigzag_labels


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h")


def test_trend_scan_up_then_down():
    # clean ramp up then ramp down (peak in the middle) + tiny noise
    up = np.linspace(100, 140, 120)
    dn = np.linspace(140, 100, 120)
    c = pd.Series(np.r_[up, dn], index=_idx(240))
    lab = trend_scan_labels(c, h_min=4, h_max=24, t_thresh=2.0)
    # early bars in the up-leg are labelled +1; mid bars of the down-leg are -1
    assert lab.iloc[20:60].mean() > 0.5      # up-leg → mostly +1
    assert lab.iloc[140:180].mean() < -0.5   # down-leg → mostly -1


def test_trend_scan_flat_is_neutral():
    # flat + noise → no significant trend → mostly 0
    rng = np.random.default_rng(0)
    c = pd.Series(100 + rng.normal(0, 0.01, 200).cumsum() * 0.0, index=_idx(200))
    lab = trend_scan_labels(c, h_min=4, h_max=24, t_thresh=3.0)
    assert (lab == 0).mean() > 0.5


def test_zigzag_alternates_and_thresholds():
    # triangle wave with amplitude >> ATR → clear alternating pivots
    seg = np.r_[np.linspace(100, 120, 50), np.linspace(120, 100, 50)]
    c = pd.Series(np.tile(seg, 3), index=_idx(300))
    high = c + 0.1; low = c - 0.1
    atr = pd.Series(1.0, index=c.index)        # k*ATR = 3 << 20-pt swings
    zz = zigzag_labels(high, low, c, atr, k=3.0)
    assert set(np.unique(zz)).issubset({-1.0, 1.0})
    assert (zz == 1).any() and (zz == -1).any()    # both directions present


def test_zigzag_ignores_subthreshold_noise():
    # gentle uptrend with sub-ATR wiggles → should stay +1 (no false reversals)
    c = pd.Series(np.linspace(100, 200, 300) + np.sin(np.arange(300)) * 0.05, index=_idx(300))
    atr = pd.Series(5.0, index=c.index)        # k*ATR=15 >> 0.05 wiggle
    zz = zigzag_labels(c + 0.01, c - 0.01, c, atr, k=3.0)
    assert (zz == 1).mean() > 0.9              # essentially all up


def test_labels_index_aligned():
    c = pd.Series(np.linspace(1, 2, 50), index=_idx(50))
    assert trend_scan_labels(c).index.equals(c.index)
    assert zigzag_labels(c, c, c, pd.Series(0.01, index=c.index)).index.equals(c.index)
