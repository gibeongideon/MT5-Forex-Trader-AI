"""Tests for src/features/mtf_features.py — leak-free lower-TF → 4H aggregation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.mtf_features import mtf_features


def _m15(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    close = 1000 + rng.normal(0, 1, n).cumsum()
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "tick_volume": 1, "spread": 0.2}, index=idx)


def test_index_is_4h_grid_and_columns_tagged():
    f = mtf_features(df=_m15())
    assert (f.index.to_series().diff().dropna() == pd.Timedelta("4h")).all()
    # one feature set per lower TF, prefixed
    for tag in ("h2", "h1", "m30", "m15"):
        assert any(c.startswith(tag + "_") for c in f.columns)
    assert f"m15_ewmac" in f.columns and "h1_rsi" in f.columns


def test_no_lookahead_perturb_last_bar():
    """Perturbing ONLY the last M15 bar must not change any earlier 4H feature row."""
    d = _m15(seed=1)
    f0 = mtf_features(df=d)
    d2 = d.copy()
    d2.iloc[-1, d2.columns.get_indexer(["high", "low", "close", "open"])] *= 1.05  # shock final bar
    f1 = mtf_features(df=d2)
    # all rows except the final 4H bin must be identical
    common = f0.index.intersection(f1.index)[:-1]
    pd.testing.assert_frame_equal(f0.loc[common], f1.loc[common])


def test_future_m15_does_not_affect_past_4h():
    """Appending FUTURE M15 bars must not change earlier 4H feature rows (causal)."""
    d = _m15(seed=2)
    cut = 3000
    f_short = mtf_features(df=d.iloc[:cut])
    f_full = mtf_features(df=d)
    common = f_short.index.intersection(f_full.index)[:-1]   # drop last (partial) bin of the short run
    pd.testing.assert_frame_equal(f_short.loc[common], f_full.loc[common])
