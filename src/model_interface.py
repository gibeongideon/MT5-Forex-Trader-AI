"""
Abstract interface that every prediction model in this system must implement.

This is the single contract between the trading engine and any signal/model.
The engine only ever calls predict_proba() — it never knows whether the
underlying model is XGBoost, LightGBM, a rule engine, or an LLM.

Output convention (always):
    np.array([P_buy, P_hold, P_sell])
    - three floats that sum to 1.0
    - index 0 = probability of upward move worth buying
    - index 1 = probability of no clear edge (hold)
    - index 2 = probability of downward move worth selling

All implementations live under src/models/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelInterface(ABC):

    # ── Core contract ─────────────────────────────────────────────────────

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability vector for the current feature row(s).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix — either a single row (live trading) or
            multiple rows (batch backtest).

        Returns
        -------
        np.ndarray, shape (3,) for single row or (n, 3) for batch.
            [P_buy, P_hold, P_sell] — always sums to 1.0 per row.
        """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> "ModelInterface":
        """Fit the model. Returns self for chaining."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> "ModelInterface":
        """Load a previously saved model. Returns self."""

    @abstractmethod
    def metadata(self) -> dict:
        """
        Return model metadata dict. Must include at minimum:
            name        : str   — human-readable model name
            version     : str   — e.g. "1.0"
            trained_on  : str   — date range used for training
            features    : list  — feature column names used
            n_classes   : int   — always 3 for this system
        """

    # ── Convenience helpers (shared, not abstract) ────────────────────────

    def signal(self, X: pd.DataFrame, threshold: float = 0.55) -> str:
        """
        Return 'buy', 'sell', or 'hold' for the latest row.
        Only acts if the winning class exceeds threshold.
        """
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[-1]
        p_buy, p_hold, p_sell = proba
        if p_buy >= threshold:
            return "buy"
        if p_sell >= threshold:
            return "sell"
        return "hold"

    def confidence(self, X: pd.DataFrame) -> float:
        """Return the highest single-class probability for the latest row."""
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[-1]
        return float(proba.max())

    def __repr__(self) -> str:
        try:
            m = self.metadata()
            return f"{m.get('name', self.__class__.__name__)} v{m.get('version', '?')}"
        except Exception:
            return self.__class__.__name__
