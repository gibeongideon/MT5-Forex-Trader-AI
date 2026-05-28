"""
K-Means Candle Tokenizer — RANDOM_IDEAS.MD Idea #2.

Clusters each bar by its shape characteristics into one of N discrete
archetypes (tokens). The cluster ID is added as a single integer feature
that XGBoost can split on, capturing intra-bar structure not encoded by
the 31 base indicators (which describe rolling statistics, not bar shape).

Per-bar features used for clustering (5 dimensions):
  log_return    : signed directional move (close/open - 1)
  upper_wick    : upper wick as % of ATR (shadow above body)
  lower_wick    : lower wick as % of ATR (shadow below body)
  vol_zscore    : volume z-score vs rolling 20-bar mean
  atr_norm_range: (high - low) / ATR  — relative bar range

KMeans is fitted on training data only (no lookahead). At inference,
each bar is assigned to its nearest cluster centroid.

Typical cluster archetypes (k=32) include:
  - Aggressive bullish breakout (high log_return, large range, high vol)
  - Hammer/pin bar (low wick dominant, small body)
  - Doji / indecision (near-zero log_return, balanced wicks)
  - Low-volatility drift (tiny range, small wicks, average vol)

Usage:
    tok = CandleTokenizer(n_clusters=32)
    tok.fit(df_train)                  # fit on training split only
    cluster_df = tok.transform(df)     # adds 'candle_cluster' column
    tok.save("data/models/candle_tokenizer.joblib")
    tok.load("data/models/candle_tokenizer.joblib")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


class CandleTokenizer:
    """
    KMeans candle shape tokenizer.

    Parameters
    ----------
    n_clusters   : number of candle archetypes (32 recommended)
    random_state : reproducibility seed
    """

    def __init__(self, n_clusters: int = 32, random_state: int = 42):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self._kmeans      = None
        self._vol_mean: Optional[float] = None
        self._vol_std:  Optional[float] = None
        self._atr_mean: Optional[float] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _compute_bar_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 5-dimensional per-bar shape features."""
        cols = {c.lower() for c in df.columns}
        d    = df.copy()
        d.columns = [c.lower() for c in d.columns]

        open_  = d["open"].values.astype(np.float64)
        high   = d["high"].values.astype(np.float64)
        low    = d["low"].values.astype(np.float64)
        close  = d["close"].values.astype(np.float64)

        # Signed log return: direction + magnitude
        log_ret = np.log(close / np.where(open_ > 0, open_, 1e-8))

        # Body boundaries
        body_top = np.maximum(open_, close)
        body_bot = np.minimum(open_, close)

        # ATR proxy: rolling 14-bar high-low mean (or raw range if short series)
        raw_range = high - low
        atr_proxy = pd.Series(raw_range).rolling(14, min_periods=1).mean().values
        atr_proxy = np.where(atr_proxy < 1e-8, 1e-8, atr_proxy)

        upper_wick = (high - body_top) / atr_proxy
        lower_wick = (body_bot - low)  / atr_proxy
        atr_norm   = raw_range / atr_proxy

        # Volume z-score vs rolling 20-bar mean
        if "tick_volume" in cols:
            vol = d["tick_volume"].values.astype(np.float64)
        else:
            vol = np.ones(len(d))
        vol_mean_roll = pd.Series(vol).rolling(20, min_periods=5).mean().fillna(vol.mean() or 1.0).values
        vol_std_roll  = pd.Series(vol).rolling(20, min_periods=5).std().fillna(1.0).values
        vol_std_roll  = np.where(vol_std_roll < 1e-8, 1e-8, vol_std_roll)
        vol_zscore    = (vol - vol_mean_roll) / vol_std_roll

        feats = np.column_stack([
            np.clip(log_ret,    -0.01, 0.01),   # clip extreme moves
            np.clip(upper_wick, 0.0,  5.0),
            np.clip(lower_wick, 0.0,  5.0),
            np.clip(vol_zscore, -3.0, 3.0),
            np.clip(atr_norm,   0.0,  5.0),
        ]).astype(np.float32)

        # Replace any NaN/inf with 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CandleTokenizer":
        """Fit KMeans on training data. Call on train split only."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = self._compute_bar_features(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(
            n_clusters   = self.n_clusters,
            random_state = self.random_state,
            n_init       = 10,
            max_iter     = 300,
        )
        km.fit(X_scaled)

        self._kmeans  = km
        self._scaler  = scaler
        self._trained_on = str(df.index[0]) if len(df) > 0 else "unknown"
        print(
            f"[CandleTokenizer] Fitted KMeans(k={self.n_clusters}) on "
            f"{len(df):,} bars — cluster sizes: "
            f"min={np.bincount(km.labels_).min()}  "
            f"max={np.bincount(km.labels_).max()}",
            flush=True,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign cluster IDs. Returns DataFrame with 'candle_cluster' column."""
        if self._kmeans is None:
            raise RuntimeError("Not fitted. Call .fit() first.")
        X        = self._compute_bar_features(df)
        X_scaled = self._scaler.transform(X)
        labels   = self._kmeans.predict(X_scaled).astype(np.int32)
        return pd.DataFrame({"candle_cluster": labels}, index=df.index)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "kmeans":     self._kmeans,
            "scaler":     self._scaler,
            "n_clusters": self.n_clusters,
            "trained_on": getattr(self, "_trained_on", None),
        }, path)
        print(f"[CandleTokenizer] Saved → {path}", flush=True)

    def load(self, path: str) -> "CandleTokenizer":
        data = joblib.load(path)
        self._kmeans    = data["kmeans"]
        self._scaler    = data["scaler"]
        self.n_clusters = data["n_clusters"]
        self._trained_on = data.get("trained_on")
        print(f"[CandleTokenizer] Loaded ← {path}  (k={self.n_clusters})", flush=True)
        return self
