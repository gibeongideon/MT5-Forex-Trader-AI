"""
XGBoost prediction model — Phase 4.

Outputs calibrated [P_buy, P_hold, P_sell] probabilities.
Implements ModelInterface so it is drop-in swappable with any other model.

Probability calibration:
    Raw XGBoost probabilities can be poorly calibrated (overconfident).
    We wrap with sklearn's CalibratedClassifierCV (isotonic regression) so that
    a predicted probability of 0.70 actually means the event happens ~70% of
    the time in historical data. This is critical for confidence-based position
    sizing in later phases.

Label convention (must match feature_pipeline.py):
    y = 1  → buy
    y = 0  → hold
    y = -1 → sell

Output convention (ModelInterface contract):
    predict_proba() → [P_buy, P_hold, P_sell]
    i.e. column order is [1, 0, -1] mapped to index [0, 1, 2]
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.model_interface import ModelInterface


class XGBoostModel(ModelInterface):
    """
    XGBoost 3-class classifier with isotonic probability calibration.

    Parameters
    ----------
    n_estimators   : number of boosting rounds
    max_depth      : tree depth (lower = less overfitting)
    learning_rate  : step size shrinkage
    subsample      : fraction of rows sampled per tree
    colsample      : fraction of features sampled per tree
    calibration_cv : number of CV folds for calibration (0 = disable)
    """

    def __init__(
        self,
        n_estimators:   int   = 300,
        max_depth:      int   = 4,
        learning_rate:  float = 0.05,
        subsample:      float = 0.8,
        colsample:      float = 0.8,
        calibration_cv: int   = 5,
        random_state:   int   = 42,
    ):
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.learning_rate  = learning_rate
        self.subsample      = subsample
        self.colsample      = colsample
        self.calibration_cv = calibration_cv
        self.random_state   = random_state

        self._model: CalibratedClassifierCV | XGBClassifier | None = None
        self._feature_names: list[str] = []
        self._classes: np.ndarray | None = None
        self._trained_on: str = ""

    # ── ModelInterface ────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"

        base = XGBClassifier(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            subsample     = self.subsample,
            colsample_bytree = self.colsample,
            random_state  = self.random_state,
            eval_metric   = "mlogloss",
            verbosity     = 0,
            use_label_encoder = False,
        )

        if self.calibration_cv > 0:
            self._model = CalibratedClassifierCV(
                base, method="isotonic", cv=self.calibration_cv
            )
        else:
            self._model = base

        # XGBoost requires 0-indexed classes; remap -1/0/1 → 0/1/2
        self._classes = np.sort(np.unique(y.values))
        y_enc = pd.Series(
            np.searchsorted(self._classes, y.values), index=y.index
        )
        self._model.fit(X, y_enc)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns shape (n_rows, 3) — columns: [P_buy, P_hold, P_sell].
        For a single-row input, returns shape (3,).
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Align columns to training feature order
        cols = [c for c in self._feature_names if c in X.columns]
        raw  = self._model.predict_proba(X[cols])

        # sklearn returns probabilities in sorted class order.
        # Our classes are typically [-1, 0, 1] → [sell, hold, buy].
        # We reorder to always return [P_buy, P_hold, P_sell].
        class_list = list(self._classes)
        idx_buy    = class_list.index(1)  if 1  in class_list else None
        idx_hold   = class_list.index(0)  if 0  in class_list else None
        idx_sell   = class_list.index(-1) if -1 in class_list else None

        ordered = np.zeros((len(raw), 3))
        if idx_buy  is not None: ordered[:, 0] = raw[:, idx_buy]
        if idx_hold is not None: ordered[:, 1] = raw[:, idx_hold]
        if idx_sell is not None: ordered[:, 2] = raw[:, idx_sell]

        return ordered[0] if len(ordered) == 1 else ordered

    def save(self, path: str | Path = "data/models/xgboost.joblib") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":         self._model,
            "feature_names": self._feature_names,
            "classes":       self._classes,
            "trained_on":    self._trained_on,
            "params": {
                "n_estimators":   self.n_estimators,
                "max_depth":      self.max_depth,
                "learning_rate":  self.learning_rate,
                "subsample":      self.subsample,
                "colsample":      self.colsample,
                "calibration_cv": self.calibration_cv,
            },
        }
        joblib.dump(payload, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path = "data/models/xgboost.joblib") -> "XGBoostModel":
        payload = joblib.load(path)
        self._model         = payload["model"]
        self._feature_names = payload["feature_names"]
        self._classes       = payload["classes"]
        self._trained_on    = payload.get("trained_on", "")
        params = payload.get("params", {})
        for k, v in params.items():
            setattr(self, k, v)
        print(f"Model loaded ← {path}")
        return self

    def metadata(self) -> dict:
        return {
            "name":       "XGBoostModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "n_estimators":  self.n_estimators,
                "max_depth":     self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }

    # ── Extra helpers ─────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 15) -> pd.Series:
        """Return top-N feature importances (only works before calibration wrapping)."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        base = self._model
        if hasattr(base, "estimator"):
            base = base.estimator
        if not hasattr(base, "feature_importances_"):
            raise RuntimeError("Feature importances not available on calibrated model.")
        imp = pd.Series(base.feature_importances_, index=self._feature_names)
        return imp.sort_values(ascending=False).head(top_n)
