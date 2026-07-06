"""
Tests for FractalLabeler — no MT5 / bridge required.

Run:
    conda run -n envmt5 python -m pytest tests/test_fractal_labeler.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.fractal_labeler import FractalLabeler


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="15min")
    return pd.DataFrame({"close": prices}, index=idx)


def _trending(n: int = 300, step: float = 0.0001) -> pd.DataFrame:
    prices = [1.1000 + i * step for i in range(n)]
    return _make_df(prices)


def _flat(n: int = 300) -> pd.DataFrame:
    return _make_df([1.1000] * n)


def _v_shape(depth: int = 30) -> pd.DataFrame:
    """A clean V-shaped fractal followed by stable prices."""
    descent = [1.1000 - i * 0.0001 for i in range(depth)]
    ascent  = [1.1000 - depth * 0.0001 + i * 0.0001 for i in range(depth)]
    tail    = [ascent[-1]] * 100
    return _make_df(descent + ascent + tail)


# ── Correlation output ────────────────────────────────────────────────────────

class TestComputeCorrelations:

    def test_output_shape(self):
        df = _trending(200)
        lab = FractalLabeler(min_window=6, max_window=30)
        corrs = lab.compute_correlations(df)
        assert len(corrs) == len(df)

    def test_output_range(self):
        df = _trending(200)
        lab = FractalLabeler(min_window=6, max_window=30)
        corrs = lab.compute_correlations(df)
        assert corrs.min() >= -1.0 - 1e-9
        assert corrs.max() <=  1.0 + 1e-9

    def test_insufficient_data_returns_zeros(self):
        # Only 4 bars — shorter than min_window=6 → all zeros
        df = _make_df([1.0, 1.1, 1.2, 1.3])
        lab = FractalLabeler(min_window=6, max_window=20)
        corrs = lab.compute_correlations(df)
        assert (corrs == 0.0).all()

    def test_symmetric_pattern_has_high_correlation(self):
        df = _v_shape(depth=20)
        lab = FractalLabeler(min_window=6, max_window=50)
        corrs = lab.compute_correlations(df)
        # The apex (bottom of V) should have a high correlation somewhere
        assert corrs.max() > 0.5, f"Max corr={corrs.max():.4f}, expected > 0.5"

    def test_flat_prices_correlation_is_finite(self):
        df = _flat(100)
        lab = FractalLabeler(min_window=6, max_window=20)
        corrs = lab.compute_correlations(df)
        assert corrs.isna().sum() == 0

    def test_min_window_bumped_to_even(self):
        # Odd min_window should be rounded up internally without error
        lab = FractalLabeler(min_window=5, max_window=20)
        assert lab.min_window % 2 == 0
        assert lab.min_window == 6

    def test_max_window_floored_to_even(self):
        lab = FractalLabeler(min_window=6, max_window=21)
        assert lab.max_window % 2 == 0
        assert lab.max_window == 20


# ── Label values ──────────────────────────────────────────────────────────────

class TestLabelValues:

    def test_label_set(self):
        df = _trending(300)
        lab = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.5)
        y = lab.label(df)
        assert set(y.unique()).issubset({0, 1, 2})

    def test_label_index_matches_df(self):
        df = _trending(300)
        lab = FractalLabeler(min_window=6, max_window=30)
        y = lab.label(df)
        assert list(y.index) == list(df.index)

    def test_label_dtype_int(self):
        df = _trending(200)
        lab = FractalLabeler(min_window=6, max_window=20)
        y = lab.label(df)
        assert y.dtype in (int, np.int64, np.int32, np.int8, object)
        assert all(isinstance(v, (int, np.integer)) for v in y.values[:5])

    def test_high_threshold_gives_mostly_no_trade(self):
        # corr_threshold=1.0 → only perfect fractals get labels (rare / never)
        df = _trending(500)
        lab = FractalLabeler(corr_threshold=1.0)
        y = lab.label(df)
        assert (y == 2).mean() > 0.95

    def test_low_threshold_increases_trade_rate(self):
        df = _trending(500)
        lab_low  = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.2)
        lab_high = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.95)
        y_low    = lab_low.label(df)
        y_high   = lab_high.label(df)
        trade_low  = (y_low  != 2).mean()
        trade_high = (y_high != 2).mean()
        assert trade_low >= trade_high, (
            f"Lower threshold should increase trade rate: {trade_low:.3f} vs {trade_high:.3f}"
        )

    def test_no_lookahead_at_tail(self):
        # Last `horizon` bars have no valid future data → should be no-trade (2)
        df = _trending(100)
        horizon = 10
        lab = FractalLabeler(min_window=6, max_window=20,
                             corr_threshold=0.0, horizon=horizon)
        y = lab.label(df)
        # Last `horizon` bars cannot have a valid label
        assert (y.iloc[-horizon:] == 2).all(), (
            "Last `horizon` bars should always be no-trade (no future data)"
        )


# ── Label stats ───────────────────────────────────────────────────────────────

class TestLabelStats:

    def test_stats_keys(self):
        df = _trending(300)
        lab = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.5)
        stats = lab.label_stats(df)
        assert {"total", "buy", "sell", "no_trade", "trade_rate"} == set(stats.keys())

    def test_counts_sum_to_total(self):
        df = _trending(300)
        lab = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.5)
        s = lab.label_stats(df)
        assert s["buy"] + s["sell"] + s["no_trade"] == s["total"]

    def test_trade_rate_in_01(self):
        df = _trending(300)
        lab = FractalLabeler(min_window=6, max_window=30, corr_threshold=0.5)
        s = lab.label_stats(df)
        assert 0.0 <= s["trade_rate"] <= 1.0


# ── No-lookahead: label uses only past + horizon ──────────────────────────────

class TestNoLookahead:

    def test_label_at_i_uses_only_bars_i_and_later(self):
        # Corrupt the first max_window bars (oldest) after computing labels,
        # then re-compute — result should differ since those bars affect windows.
        # More directly: verify the LAST bar's label is always 2 (horizon not met).
        df = _trending(200)
        lab = FractalLabeler(min_window=6, max_window=30,
                             corr_threshold=0.0, horizon=5)
        y = lab.label(df)
        assert y.iloc[-1] == 2, "Last bar should be 2 (no future horizon data)"

    def test_sufficient_data_produces_some_labels(self):
        # V-shape data has genuine symmetric fractal patterns → should produce labels
        df = _v_shape(depth=30)
        # Append more V-shapes to get sufficient data
        extra = pd.concat([_v_shape(depth=20)] * 5)
        extra.index = pd.date_range(
            df.index[-1] + pd.Timedelta("15min"),
            periods=len(extra), freq="15min"
        )
        df = pd.concat([df, extra])
        lab = FractalLabeler(min_window=6, max_window=60,
                             corr_threshold=0.7, horizon=5)
        y = lab.label(df)
        assert (y != 2).sum() > 0, (
            f"Expected some trade labels on V-shape data with threshold=0.7, "
            f"got {(y != 2).sum()}"
        )
