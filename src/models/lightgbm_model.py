"""
LightGBM prediction model — Phase 5.

Drop-in replacement for XGBoostModel. Implements ModelInterface and outputs
calibrated [P_buy, P_hold, P_sell] probabilities, same contract as every
other model in this system.

LightGBM differences vs XGBoost:
  - Faster training on large datasets (leaf-wise tree growth)
  - Often better on high-cardinality categorical features
  - Slightly different hyperparameter names/defaults
  - Native support for multi-class without label remapping

Label convention (must match feature_pipeline.py):
    y = 1  → buy
    y = 0  → hold
    y = -1 → sell

Output convention (ModelInterface contract):
    predict_proba() → [P_buy, P_hold, P_sell]
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.models.model_interface import ModelInterface


class LightGBMModel(ModelInterface):
    """
    LightGBM 3-class classifier with optional isotonic probability calibration.

    Parameters
    ----------
    n_estimators   : number of boosting rounds
    max_depth      : max tree depth (-1 = unlimited, let num_leaves control)
    num_leaves     : max leaves per tree (primary complexity control in LightGBM)
    learning_rate  : step size shrinkage
    subsample      : fraction of rows sampled per tree
    colsample      : fraction of features sampled per tree
    calibration_cv : CV folds for calibration (0 = disable)
    """

    def __init__(
        self,
        n_estimators:   int   = 300,
        max_depth:      int   = -1,
        num_leaves:     int   = 31,
        learning_rate:  float = 0.05,
        subsample:      float = 0.8,
        colsample:      float = 0.8,
        calibration_cv: int   = 5,
        random_state:   int   = 42,
    ):
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.num_leaves     = num_leaves
        self.learning_rate  = learning_rate
        self.subsample      = subsample
        self.colsample      = colsample
        self.calibration_cv = calibration_cv
        self.random_state   = random_state

        self._model: CalibratedClassifierCV | LGBMClassifier | None = None
        self._feature_names: list[str] = []
        self._classes: np.ndarray | None = None
        self._trained_on: str = ""

    # ── ModelInterface ────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"

        # LightGBM handles -1/0/1 labels natively — no remapping needed,
        # but we store classes for output reordering in predict_proba.
        self._classes = np.sort(np.unique(y.values))

        base = LGBMClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            num_leaves       = self.num_leaves,
            learning_rate    = self.learning_rate,
            subsample        = self.subsample,
            colsample_bytree = self.colsample,
            random_state     = self.random_state,
            verbose          = -1,
        )

        if self.calibration_cv > 0:
            self._model = CalibratedClassifierCV(
                base, method="isotonic", cv=self.calibration_cv
            )
        else:
            self._model = base

        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns shape (n_rows, 3) — columns: [P_buy, P_hold, P_sell].
        For a single-row input, returns shape (3,).
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        cols = [c for c in self._feature_names if c in X.columns]
        raw  = self._model.predict_proba(X[cols])

        # sklearn returns probabilities in sorted class order.
        # Classes are [-1, 0, 1] → reorder to [P_buy, P_hold, P_sell].
        class_list = list(self._classes)
        idx_buy    = class_list.index(1)  if 1  in class_list else None
        idx_hold   = class_list.index(0)  if 0  in class_list else None
        idx_sell   = class_list.index(-1) if -1 in class_list else None

        ordered = np.zeros((len(raw), 3))
        if idx_buy  is not None: ordered[:, 0] = raw[:, idx_buy]
        if idx_hold is not None: ordered[:, 1] = raw[:, idx_hold]
        if idx_sell is not None: ordered[:, 2] = raw[:, idx_sell]

        return ordered[0] if len(ordered) == 1 else ordered

    def save(self, path: str | Path = "data/models/lightgbm.joblib") -> None:
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
                "num_leaves":     self.num_leaves,
                "learning_rate":  self.learning_rate,
                "subsample":      self.subsample,
                "colsample":      self.colsample,
                "calibration_cv": self.calibration_cv,
            },
        }
        joblib.dump(payload, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path = "data/models/lightgbm.joblib") -> "LightGBMModel":
        payload = joblib.load(path)
        self._model         = payload["model"]
        self._feature_names = payload["feature_names"]
        self._classes       = payload["classes"]
        self._trained_on    = payload.get("trained_on", "")
        for k, v in payload.get("params", {}).items():
            setattr(self, k, v)
        print(f"Model loaded ← {path}")
        return self

    def metadata(self) -> dict:
        return {
            "name":       "LightGBMModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "n_estimators": self.n_estimators,
                "num_leaves":   self.num_leaves,
                "learning_rate": self.learning_rate,
            },
        }
