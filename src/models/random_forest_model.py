"""
Random Forest prediction model — Phase 5.

Drop-in replacement for XGBoostModel. Implements ModelInterface and outputs
calibrated [P_buy, P_hold, P_sell] probabilities.

Random Forest differences vs gradient boosting:
  - Trains trees in parallel (faster on multi-core CPUs)
  - Naturally resistant to overfitting via bagging
  - Probabilities from RF are already averaged across trees,
    so calibration is less critical but still applied for consistency.

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.model_interface import ModelInterface


class RandomForestModel(ModelInterface):
    """
    Random Forest 3-class classifier with optional isotonic calibration.

    Parameters
    ----------
    n_estimators   : number of trees
    max_depth      : max tree depth (None = grow until leaves are pure)
    min_samples_leaf: min samples required at a leaf (controls tree size)
    max_features   : features to consider at each split ('sqrt' or float)
    calibration_cv : CV folds for calibration (0 = disable)
    """

    def __init__(
        self,
        n_estimators:     int         = 300,
        max_depth:        int | None  = 10,
        min_samples_leaf: int         = 5,
        max_features:     str | float = "sqrt",
        calibration_cv:   int         = 5,
        random_state:     int         = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self.calibration_cv   = calibration_cv
        self.random_state     = random_state

        self._model: CalibratedClassifierCV | RandomForestClassifier | None = None
        self._feature_names: list[str] = []
        self._classes: np.ndarray | None = None
        self._trained_on: str = ""

    # ── ModelInterface ────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"
        self._classes       = np.sort(np.unique(y.values))

        base = RandomForestClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            max_features     = self.max_features,
            random_state     = self.random_state,
            n_jobs           = -1,  # use all CPU cores
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

        class_list = list(self._classes)
        idx_buy    = class_list.index(1)  if 1  in class_list else None
        idx_hold   = class_list.index(0)  if 0  in class_list else None
        idx_sell   = class_list.index(-1) if -1 in class_list else None

        ordered = np.zeros((len(raw), 3))
        if idx_buy  is not None: ordered[:, 0] = raw[:, idx_buy]
        if idx_hold is not None: ordered[:, 1] = raw[:, idx_hold]
        if idx_sell is not None: ordered[:, 2] = raw[:, idx_sell]

        return ordered[0] if len(ordered) == 1 else ordered

    def save(self, path: str | Path = "data/models/random_forest.joblib") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":         self._model,
            "feature_names": self._feature_names,
            "classes":       self._classes,
            "trained_on":    self._trained_on,
            "params": {
                "n_estimators":     self.n_estimators,
                "max_depth":        self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features":     self.max_features,
                "calibration_cv":   self.calibration_cv,
            },
        }
        joblib.dump(payload, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path = "data/models/random_forest.joblib") -> "RandomForestModel":
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
            "name":       "RandomForestModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "n_estimators":     self.n_estimators,
                "max_depth":        self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
            },
        }
