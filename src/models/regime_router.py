"""
Regime-Conditioned Model Router — Technique #2 from IMPROVEMENT.MD.

Instead of one XGBoost trying to learn signals for ALL market conditions,
RegimeRouter:
  1. Detects the current market regime from feature columns that describe
     volatility and trend structure (ATR, ADX, RSI).
  2. Trains a SPECIALIST XGBoost for each regime — one model only sees
     trending data, another only ranging data, etc.
  3. At inference, detects the regime and routes to the specialist.

Regime definitions (KMeans k=4 on [atr_ratio, adx, rsi_centred]):
  0 — Trending up   : high ADX, RSI > 50, moderate-high ATR
  1 — Trending down : high ADX, RSI < 50, moderate-high ATR
  2 — Ranging       : low ADX, low-moderate ATR, RSI ~50
  3 — High vol      : very high ATR relative to history

Implements ModelInterface so it is a transparent drop-in for any
walk-forward or pipeline that accepts a model_type string.

Usage in config.yaml:
    pipeline:
      model_type: regime_router

Usage in walk-forward script:
    WalkForwardConfig(model_type="regime_router", ...)
"""

from __future__ import annotations

import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.models.model_interface import ModelInterface

warnings.filterwarnings("ignore")


# ── Feature columns used for regime detection ─────────────────────────────────
# These are guaranteed present by FeaturePipeline (see feature_pipeline.py).
_REGIME_COLS = ["atr_14", "adx_14", "rsi_14"]


class RegimeRouter(ModelInterface):
    """
    Routes each bar to a specialist XGBoost based on detected market regime.

    Parameters
    ----------
    n_regimes   : number of regime clusters (default 4)
    min_regime_samples : minimum bars per regime to train a specialist;
                         regimes below this threshold fall back to the
                         global model.
    """

    def __init__(self, n_regimes: int = 4, min_regime_samples: int = 200):
        self.n_regimes          = n_regimes
        self.min_regime_samples = min_regime_samples

        self._kmeans        = None   # sklearn KMeans — regime detector
        self._regime_clf    = None   # sklearn LogisticRegression — fast regime predictor
        self._specialists: dict[int, ModelInterface] = {}
        self._fallback: Optional[ModelInterface] = None
        self._regime_col_indices: list[int] = []
        self._feature_cols: list[str] = []
        self._trained_on: Optional[str] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _regime_features(self, X: pd.DataFrame) -> np.ndarray:
        """Extract [atr_ratio, adx, rsi_centred] for regime clustering."""
        feats = np.zeros((len(X), 3), dtype=np.float32)

        if "atr_14" in X.columns:
            atr = X["atr_14"].values.astype(np.float32)
            # ATR ratio: current ATR vs rolling 100-bar mean; fill early NaNs with 1.0
            roll_mean = (pd.Series(atr)
                         .rolling(100, min_periods=10).mean()
                         .fillna(pd.Series(atr).expanding().mean())
                         .values)
            roll_mean = np.where(roll_mean < 1e-8, 1e-8, roll_mean)
            feats[:, 0] = np.clip(atr / roll_mean, 0.0, 5.0)

        if "adx_14" in X.columns:
            feats[:, 1] = np.clip(X["adx_14"].values.astype(np.float32), 0.0, 100.0) / 100.0

        if "rsi_14" in X.columns:
            feats[:, 2] = (X["rsi_14"].values.astype(np.float32) - 50.0) / 50.0

        return feats

    def _detect_regimes(self, X: pd.DataFrame) -> np.ndarray:
        """Cluster X into regime labels using KMeans on regime features."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        rf = self._regime_features(X)
        scaler = StandardScaler()
        rf_scaled = scaler.fit_transform(rf)

        km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = km.fit_predict(rf_scaled)

        # Store both scaler and kmeans together for inference
        self._regime_scaler = scaler
        self._kmeans = km
        return labels.astype(np.int32)

    def _predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime labels for X using the fitted classifier."""
        rf = self._regime_features(X)
        rf_scaled = self._regime_scaler.transform(rf)
        return self._kmeans.predict(rf_scaled).astype(np.int32)

    # ── ModelInterface ────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        from src.models.xgboost_model import XGBoostModel

        self._feature_cols = list(X.columns)
        self._trained_on   = str(X.index[0]) if len(X) > 0 else "unknown"

        # Step 1: detect regimes on training data
        regimes = self._detect_regimes(X)
        counts  = np.bincount(regimes, minlength=self.n_regimes)

        print(f"[RegimeRouter] Regime distribution (n={len(X):,}):", flush=True)
        for r, n in enumerate(counts):
            label = ["trending-up", "trending-dn", "ranging", "high-vol"][r] if r < 4 else f"regime-{r}"
            print(f"  Regime {r} ({label}): {n:,} bars ({n/len(X):.1%})", flush=True)

        # Step 2: train regime classifier (LogisticRegression on regime features)
        from sklearn.linear_model import LogisticRegression
        rf = self._regime_features(X)
        rf_scaled = self._regime_scaler.transform(rf)
        clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        clf.fit(rf_scaled, regimes)
        self._regime_clf = clf

        # Step 3: train fallback on all data
        fallback = XGBoostModel()
        fallback.train(X, y)
        self._fallback = fallback

        # Step 4: train specialist per regime
        self._specialists = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            n_r  = mask.sum()
            if n_r < self.min_regime_samples:
                print(f"  Regime {r}: only {n_r} samples — using fallback", flush=True)
                continue
            specialist = XGBoostModel()
            specialist.train(X[mask], y[mask])
            self._specialists[r] = specialist
            print(f"  Regime {r}: trained specialist on {n_r:,} bars", flush=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._fallback is None:
            raise RuntimeError("RegimeRouter not trained — call train() first.")

        # Predict regime for each row
        regimes = self._predict_regimes(X)
        proba   = np.zeros((len(X), 3), dtype=np.float64)

        for r in range(self.n_regimes):
            mask = regimes == r
            if not mask.any():
                continue
            if r in self._specialists:
                proba[mask] = self._specialists[r].predict_proba(X[mask])
            else:
                proba[mask] = self._fallback.predict_proba(X[mask])

        return proba

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "n_regimes":          self.n_regimes,
            "min_regime_samples": self.min_regime_samples,
            "kmeans":             self._kmeans,
            "regime_scaler":      self._regime_scaler,
            "regime_clf":         self._regime_clf,
            "specialists":        self._specialists,
            "fallback":           self._fallback,
            "feature_cols":       self._feature_cols,
            "trained_on":         self._trained_on,
        }, path)
        print(f"[RegimeRouter] Saved → {path}", flush=True)

    def load(self, path: str) -> "RegimeRouter":
        data = joblib.load(path)
        self.n_regimes          = data["n_regimes"]
        self.min_regime_samples = data["min_regime_samples"]
        self._kmeans            = data["kmeans"]
        self._regime_scaler     = data["regime_scaler"]
        self._regime_clf        = data["regime_clf"]
        self._specialists       = data["specialists"]
        self._fallback          = data["fallback"]
        self._feature_cols      = data["feature_cols"]
        self._trained_on        = data.get("trained_on")
        print(f"[RegimeRouter] Loaded ← {path}  "
              f"(n_regimes={self.n_regimes})", flush=True)
        return self

    def metadata(self) -> dict:
        return {
            "model_type":    "regime_router",
            "n_regimes":     self.n_regimes,
            "n_specialists": len(self._specialists),
            "features":      self._feature_cols,
            "trained_on":    self._trained_on,
        }
