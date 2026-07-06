"""
Fractal Pattern Labeler — based on READ 7.

Identifies symmetric price patterns via Pearson correlation and assigns
training labels based on future price movement.

For each bar i, scans backward windows of even size in [min_window, max_window].
A window is split into left and right halves; the right half is time-reversed
and the Pearson correlation between left and reversed-right is computed. A high
correlation indicates a symmetric (fractal) price structure. Bars with best
correlation >= corr_threshold are labeled by future direction; all others get 2.

Label values
------------
    0 = buy   (price rises > markup over next `horizon` bars)
    1 = sell  (price falls > markup over next `horizon` bars)
    2 = no-trade (outside any fractal, or direction ambiguous)

Reference: MQL5 Community — "Detecting and Classifying Fractal Patterns Using
           Machine Learning" (READ 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FractalLabeler:
    """
    Labels bars based on symmetric fractal price patterns.

    Parameters
    ----------
    min_window : int
        Smallest window to check (rounded up to nearest even number).
    max_window : int
        Largest window to check (rounded down to nearest even number).
    corr_threshold : float
        Minimum Pearson r to classify a bar as "inside a fractal".
    horizon : int
        How many bars ahead to look when determining label direction.
    markup : float
        Minimum relative price change to assign buy (0) or sell (1).
        Moves smaller than this → label stays 2 (neutral).
    """

    def __init__(
        self,
        min_window:     int   = 6,
        max_window:     int   = 60,
        corr_threshold: float = 0.9,
        horizon:        int   = 5,
        markup:         float = 0.0001,
    ) -> None:
        self.min_window     = min_window + (min_window % 2)      # ensure even
        self.max_window     = max_window - (max_window % 2)      # ensure even
        self.corr_threshold = corr_threshold
        self.horizon        = horizon
        self.markup         = markup

    # ── Public API ─────────────────────────────────────────────────────────────

    def label(self, df: pd.DataFrame, close_col: str = "close") -> pd.Series:
        """
        Compute fractal labels for every bar.

        Returns
        -------
        pd.Series
            Integer labels (0/1/2), same index as `df`.
        """
        close = df[close_col].values.astype(np.float64)
        corrs, _ = self._compute_correlations(close)
        return self._assign_labels(close, corrs, df.index)

    def compute_correlations(
        self, df: pd.DataFrame, close_col: str = "close"
    ) -> pd.Series:
        """Best symmetric correlation at each bar (0 where insufficient data)."""
        close = df[close_col].values.astype(np.float64)
        corrs, _ = self._compute_correlations(close)
        return pd.Series(corrs, index=df.index)

    def label_stats(self, df: pd.DataFrame, close_col: str = "close") -> dict:
        """Label distribution summary for diagnostics."""
        y = self.label(df, close_col)
        total  = len(y)
        counts = y.value_counts().to_dict()
        n_buy  = counts.get(0, 0)
        n_sell = counts.get(1, 0)
        return {
            "total":      total,
            "buy":        n_buy,
            "sell":       n_sell,
            "no_trade":   counts.get(2, 0),
            "trade_rate": (n_buy + n_sell) / max(total, 1),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compute_correlations(
        self, close: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized scan: for each even window size w in [min_window, max_window],
        compute Pearson r between left half and time-reversed right half of every
        valid sliding window. Track best r and corresponding w per bar.

        For window ending at bar index i (0-based):
            window = close[i-w+1 : i+1]
            left   = window[:w//2]
            right  = window[w//2:]
            mirror = right[::-1]          # time-reverse
            r      = pearsonr(left, mirror)
        """
        n          = len(close)
        best_corrs = np.zeros(n, dtype=np.float64)
        best_wins  = np.zeros(n, dtype=np.int32)

        for w in range(self.min_window, self.max_window + 1, 2):
            if n < w:
                continue
            half    = w // 2
            windows = np.lib.stride_tricks.sliding_window_view(close, w)
            # windows: shape (n - w + 1, w)
            left         = windows[:, :half]
            right_mirror = windows[:, w - 1 : half - 1 : -1]  # right half, reversed
            corrs        = self._batch_pearsonr(left, right_mirror)

            # corrs[j] → window ending at bar index (j + w - 1)
            end_idx = np.arange(w - 1, n)
            improve = corrs > best_corrs[end_idx]
            best_corrs[end_idx[improve]] = corrs[improve]
            best_wins[end_idx[improve]]  = w

        return best_corrs, best_wins

    @staticmethod
    def _batch_pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Row-wise Pearson correlation. x, y: (n_windows, half_size)."""
        xm    = x - x.mean(axis=1, keepdims=True)
        ym    = y - y.mean(axis=1, keepdims=True)
        num   = (xm * ym).sum(axis=1)
        denom = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
        denom = np.where(denom < 1e-10, 1e-10, denom)
        return num / denom

    def _assign_labels(
        self,
        close: np.ndarray,
        corrs: np.ndarray,
        index: pd.Index,
    ) -> pd.Series:
        n      = len(close)
        labels = np.full(n, 2, dtype=np.int8)

        fractal_bars = np.where(corrs >= self.corr_threshold)[0]
        # Only bars where there are enough future bars for the horizon
        valid = fractal_bars[fractal_bars + self.horizon < n]
        if len(valid) == 0:
            return pd.Series(labels, index=index)

        current = close[valid]
        future  = close[valid + self.horizon]
        # Avoid division by zero
        safe_current = np.where(np.abs(current) < 1e-10, 1e-10, current)
        change = (future - current) / safe_current

        labels[valid[change >  self.markup]] = 0  # buy
        labels[valid[change < -self.markup]] = 1  # sell
        # |change| <= markup → stays 2 (neutral)

        return pd.Series(labels.astype(int), index=index)
