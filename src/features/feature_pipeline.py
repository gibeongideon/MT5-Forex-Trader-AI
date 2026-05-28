"""
Feature Engineering Pipeline — Phase 3.

Transforms raw OHLCV data into an ML-ready feature matrix.

Design rules:
  1. NO LOOKAHEAD — every feature at bar[t] is computed from bar[t-1] or older.
     All indicators are shifted by 1 bar before being added to the matrix.
  2. Scaler is FITTED on training data only, then applied (transform-only) to
     validation/test data. Never fit on the full dataset.
  3. Feature names are stable strings used throughout Phases 4–9.
  4. Labels are generated here too, but the last N rows are dropped so
     future returns are always available for every labelled row.

Usage:
    from src.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()

    # Build and fit (call once on training data):
    X_train, y_train = pipeline.build(df_train, fit=True)

    # Transform only (call on validation / live bar):
    X_val, y_val = pipeline.build(df_val, fit=False)

    # Save / load the fitted scaler:
    pipeline.save_scaler("data/models/scaler.joblib")
    pipeline.load_scaler("data/models/scaler.joblib")

    # Validate no lookahead (raises AssertionError if a feature leaks future info):
    pipeline.validate_no_lookahead(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.indicators import (
    sma, ema, rsi, macd, bollinger_bands, bollinger_pct_b,
    atr, stochastic, adx,
)


# ─── Default feature spec ─────────────────────────────────────────────────────
# Each tuple: (output_col_name_or_tuple, fn, kwargs)
# All results will be .shift(1) before entering the matrix.

_DEFAULT_SPEC = [
    # Trend
    ("sma_9",                          sma,              {"period": 9}),
    ("sma_21",                         sma,              {"period": 21}),
    ("sma_50",                         sma,              {"period": 50}),
    ("ema_20",                         ema,              {"period": 20}),
    # Momentum
    ("rsi_14",                         rsi,              {"period": 14}),
    (("macd_line", "macd_sig", "macd_hist"), macd,       {}),
    # Volatility / mean-reversion
    ("bb_pct",                         bollinger_pct_b,  {"period": 20}),
    (("bb_upper", "bb_mid", "bb_lower"), bollinger_bands, {"period": 20}),
    ("atr_14",                         atr,              {"period": 14}),
    # Oscillators
    (("stoch_k", "stoch_d"),           stochastic,       {}),
    ("adx_14",                         adx,              {"period": 14}),
]

_OHLCV_FUNS = {atr, stochastic, adx}


class FeaturePipeline:
    """
    Builds an ML-ready feature matrix from OHLCV data.

    Parameters
    ----------
    label_horizon : int
        How many bars ahead to look when generating labels.
        y=1 (buy) if close[t+horizon] / close[t] - 1 > label_threshold.
    label_threshold : float
        Minimum forward return to count as a directional label.
        Returns between -threshold and +threshold are labelled 0 (hold).
    scale : bool
        Whether to apply StandardScaler to the feature matrix.
    extra_spec : list
        Additional indicator specs in the same format as _DEFAULT_SPEC.
        Appended to the default feature set.
    """

    def __init__(
        self,
        label_horizon:   int   = 4,
        label_threshold: float = 0.0003,
        scale:           bool  = True,
        extra_spec:      list  = None,
    ):
        self.label_horizon   = label_horizon
        self.label_threshold = label_threshold
        self.scale           = scale
        self._spec           = _DEFAULT_SPEC + (extra_spec or [])
        self._scaler: Optional[StandardScaler] = None
        self._feature_cols: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        df:   pd.DataFrame,
        fit:  bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix X and label series y.

        Parameters
        ----------
        df   : OHLCV DataFrame with columns open/high/low/close/tick_volume,
               indexed by datetime.
        fit  : If True, fit the scaler on this data (use for training set only).
               If False, apply the previously fitted scaler (for val/test/live).

        Returns
        -------
        X : pd.DataFrame — feature matrix, NaN rows dropped
        y : pd.Series    — labels aligned to X (1=buy, -1=sell, 0=hold)
                           Last label_horizon rows are dropped.
        """
        raw = self._compute_indicators(df)
        feat = self._add_derived_features(raw)

        # Shift ALL indicator/derived columns by 1 to prevent lookahead.
        # close/open/high/low are kept unshifted for label generation only —
        # they are dropped before the feature matrix is finalised.
        indicator_cols = [c for c in feat.columns if c not in df.columns]
        feat[indicator_cols] = feat[indicator_cols].shift(1)

        # Generate labels from the unshifted close (future return).
        y = self._make_labels(feat["close"])

        # Drop raw OHLCV columns — not features, only used for labelling.
        feat = feat.drop(columns=["open", "high", "low", "close", "tick_volume",
                                   "spread", "real_volume"], errors="ignore")

        # Drop rows with NaN (indicator warmup period + shift).
        feat = feat.dropna()
        y    = y.reindex(feat.index)

        # Drop the last label_horizon rows: their labels reference future bars
        # that may not exist, so they are unreliable.
        if len(feat) > self.label_horizon:
            feat = feat.iloc[: -self.label_horizon]
            y    = y.iloc[: -self.label_horizon]

        self._feature_cols = list(feat.columns)

        if self.scale:
            if fit:
                self._scaler = StandardScaler()
                feat_scaled = pd.DataFrame(
                    self._scaler.fit_transform(feat),
                    index=feat.index,
                    columns=feat.columns,
                )
            else:
                if self._scaler is None:
                    raise RuntimeError(
                        "Scaler not fitted. Call build(df, fit=True) on training data first, "
                        "or load a saved scaler with load_scaler()."
                    )
                feat_scaled = pd.DataFrame(
                    self._scaler.transform(feat),
                    index=feat.index,
                    columns=feat.columns,
                )
            return feat_scaled, y

        return feat, y

    def build_live(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a single-row feature vector for the latest bar (live trading).

        Returns a 1-row DataFrame suitable for model.predict_proba().
        No labels are generated. fit=False is assumed (scaler must be loaded).
        """
        raw  = self._compute_indicators(df)
        feat = self._add_derived_features(raw)
        indicator_cols = [c for c in feat.columns if c not in df.columns]
        feat[indicator_cols] = feat[indicator_cols].shift(1)
        feat = feat.drop(columns=["open", "high", "low", "close", "tick_volume",
                                   "spread", "real_volume"], errors="ignore")
        # Take only the last row (current bar).
        row = feat.iloc[[-1]].dropna(axis=1)

        if self.scale and self._scaler is not None:
            cols = [c for c in self._feature_cols if c in row.columns]
            row = row[cols]
            return pd.DataFrame(
                self._scaler.transform(row),
                index=row.index,
                columns=row.columns,
            )
        return row

    def validate_no_lookahead(self, df: pd.DataFrame) -> None:
        """
        Assert that no feature column correlates suspiciously strongly with
        the next-bar return. Raises AssertionError with offending columns if found.

        This is a heuristic check — it catches accidental shift(0) bugs where
        a feature at bar[t] contains information about bar[t] close prices.
        """
        X, _ = self.build(df.copy(), fit=True)
        future_return = df["close"].pct_change().shift(-1).reindex(X.index)

        violations = []
        for col in X.columns:
            corr = X[col].corr(future_return)
            if abs(corr) > 0.95:
                violations.append(f"{col} (corr={corr:.3f})")

        assert not violations, (
            f"Potential lookahead bias detected in: {violations}\n"
            "Ensure all features use .shift(1) before entering the matrix."
        )
        print(f"Lookahead validation PASSED — {len(X.columns)} features checked, 0 violations.")

    def feature_names(self) -> list[str]:
        """Return feature column names from the last build() call."""
        return list(self._feature_cols)

    def save_scaler(self, path: str | Path = "data/models/scaler.joblib") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        print(f"Scaler saved → {path}")

    def load_scaler(self, path: str | Path = "data/models/scaler.joblib") -> None:
        self._scaler = joblib.load(path)
        print(f"Scaler loaded ← {path}")

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for item in self._spec:
            names, fn, kwargs = item
            arg = df if fn in _OHLCV_FUNS else df["close"]
            output = fn(arg, **kwargs)
            if isinstance(names, str):
                result[names] = output
            else:
                for col, series in zip(names, output):
                    result[col] = series
        return result

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features and rolling statistics on close price."""
        close = df["close"]

        # Percentage returns
        df["return_1"]  = close.pct_change(1)
        df["return_3"]  = close.pct_change(3)
        df["return_5"]  = close.pct_change(5)
        df["return_10"] = close.pct_change(10)

        # Normalised lag (close relative to rolling mean)
        df["close_lag_1"] = close.shift(1)
        df["close_lag_2"] = close.shift(2)
        df["close_lag_5"] = close.shift(5)

        # Rolling volatility
        df["rolling_std_10"] = close.rolling(10).std()
        df["rolling_std_20"] = close.rolling(20).std()
        df["rolling_std_50"] = close.rolling(50).std()

        # Price position within recent range
        rolling_high_20 = df["high"].rolling(20).max() if "high" in df.columns else close.rolling(20).max()
        rolling_low_20  = df["low"].rolling(20).min()  if "low"  in df.columns else close.rolling(20).min()
        rng = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
        df["price_position_20"] = (close - rolling_low_20) / rng

        # MA spread (fast - slow, normalised by slow)
        if "sma_9" in df.columns and "sma_21" in df.columns:
            df["ma_spread_9_21"] = (df["sma_9"] - df["sma_21"]) / df["sma_21"].replace(0, np.nan)

        if "sma_21" in df.columns and "sma_50" in df.columns:
            df["ma_spread_21_50"] = (df["sma_21"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan)

        # RSI momentum (change in RSI over last 3 bars)
        if "rsi_14" in df.columns:
            df["rsi_momentum"] = df["rsi_14"].diff(3)

        # ATR normalised by price (relative volatility)
        if "atr_14" in df.columns:
            df["atr_pct"] = df["atr_14"] / close.replace(0, np.nan)

        # Session / time-of-day features
        # EURUSD session boundaries (UTC): Asia 00-07, London 07-16, NY 12-21
        if hasattr(df.index, "hour"):
            hour = df.index.hour
            # Cyclical encoding — preserves continuity across midnight
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            # Binary session flags
            df["is_london_open"]     = ((hour >= 7)  & (hour < 16)).astype(np.float32)
            df["is_ny_open"]         = ((hour >= 12) & (hour < 21)).astype(np.float32)
            df["is_london_ny_overlap"] = ((hour >= 12) & (hour < 16)).astype(np.float32)
            df["is_asia"]            = ((hour >= 0)  & (hour < 7)).astype(np.float32)

        return df

    def _make_labels(self, close: pd.Series) -> pd.Series:
        """
        Generate forward-looking labels:
            y = 1  if return over next label_horizon bars > +label_threshold
            y = -1 if return over next label_horizon bars < -label_threshold
            y = 0  otherwise (hold zone)

        The shift(-horizon) aligns each bar's label with its row.
        The last label_horizon rows will have NaN labels and are dropped in build().
        """
        future_close  = close.shift(-self.label_horizon)
        forward_return = (future_close - close) / close

        labels = pd.Series(0, index=close.index, dtype=int)
        labels[forward_return >  self.label_threshold] =  1
        labels[forward_return < -self.label_threshold] = -1
        return labels
