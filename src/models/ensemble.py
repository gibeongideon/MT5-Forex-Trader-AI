"""
Signal Stacking & Meta-Learning — Phase 6.

Combines outputs of multiple ModelInterface instances into a single
[P_buy, P_hold, P_sell] probability via a trained meta-learner.

Architecture:
    Layer 0:  Raw features → [XGBoost, LightGBM, CatBoost, RF, LSTM, RuleEngine]
                              each outputs [P_buy, P_hold, P_sell]
    Layer 1:  Stacked Layer-0 probs (+ optional rule score) → MetaLearner
                              outputs final [P_buy, P_hold, P_sell]

Leakage prevention:
    The meta-learner is trained ONLY on out-of-fold predictions from the base
    models. We use k-fold cross-validation on the training set so the meta-
    learner never sees a prediction made on data the base model was trained on.
    This is the standard "stacking" protocol.

Usage:
    from src.models.ensemble import Ensemble
    from src.models.xgboost_model import XGBoostModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.catboost_model import CatBoostModel

    ens = Ensemble(
        base_models=[XGBoostModel(), LightGBMModel(), CatBoostModel()],
        meta_model="logistic",   # or "lightgbm"
        n_folds=5,
    )
    ens.train(X_train, y_train)
    proba = ens.predict_proba(X_test)   # → [P_buy, P_hold, P_sell]
    ens.save("data/models/ensemble.joblib")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.models.model_interface import ModelInterface


class Ensemble(ModelInterface):
    """
    Two-layer stacking ensemble (mode="stack") or weighted-blend ensemble (mode="blend").

    Parameters
    ----------
    base_models  : list of trained-or-untrained ModelInterface instances
    meta_model   : "logistic" | "lightgbm" — the Layer-1 combiner (stack mode only)
    n_folds      : CV folds for generating out-of-fold base predictions (stack mode only)
    use_original : if True, also pass the original features to the meta-learner
                   alongside the stacked probabilities (stack mode only)
    mode         : "blend" (weighted average, default) | "stack" (meta-learner OOF)
    weights      : per-model weights for blend mode; normalized to sum to 1 automatically.
                   Defaults to equal weights if not provided.
    """

    def __init__(
        self,
        base_models: List[ModelInterface],
        meta_model:  str = "logistic",
        n_folds:     int = 5,
        use_original: bool = False,
        mode:        str = "blend",
        weights:     Optional[List[float]] = None,
    ):
        self.base_models   = base_models  # may be empty when loading from file
        self.meta_model_type = meta_model
        self.n_folds       = n_folds
        self.use_original  = use_original
        self.mode          = mode
        self._weights      = weights if weights else [1.0] * len(base_models)

        self._meta: Optional[object] = None
        self._feature_names: list[str] = []
        self._classes: np.ndarray | None = None
        self._trained_on: str = ""

    # ── ModelInterface ────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        extra_meta: Optional[pd.DataFrame] = None,
    ) -> "Ensemble":
        """
        Blend mode: trains each base model independently on (X, y). No OOF, no meta-learner.
        Stack mode:
          1. Generate out-of-fold Layer-0 predictions via StratifiedKFold.
          2. Train meta-learner on stacked OOF predictions (+ extra_meta if given).
          3. Retrain all base models on the full training set.

        Parameters
        ----------
        extra_meta : optional DataFrame aligned to X.index.
            Extra columns appended to the meta-feature matrix only (not passed to
            base models). Use this to inject latent/regime features into the
            meta-learner without inflating the base model feature space.
            (Stack mode only — ignored in blend mode.)
        """
        if not self.base_models:
            raise ValueError("Ensemble needs at least one base model.")

        # ── Blend mode ────────────────────────────────────────────────────────
        if self.mode == "blend":
            self._feature_names = list(X.columns)
            self._trained_on = f"{X.index[0].date()} → {X.index[-1].date()}"
            print(f"Ensemble (blend): training {len(self.base_models)} base models...")
            for i, model in enumerate(self.base_models):
                name = type(model).__name__
                w = self._weights[i] if i < len(self._weights) else 1.0
                print(f"  [{i+1}/{len(self.base_models)}] {name}  weight={w:.2f}")
                model.train(X, y)
            print("Ensemble (blend) training complete.")
            return self

        # ── Stack mode (existing logic) ───────────────────────────────────────
        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"
        self._classes       = np.sort(np.unique(y.values))
        self._extra_meta_cols: list[str] = list(extra_meta.columns) if extra_meta is not None else []

        n_models  = len(self.base_models)
        n_rows    = len(X)
        # Each base model contributes 3 probability columns
        oof_stack = np.zeros((n_rows, n_models * 3), dtype=np.float32)

        # Pre-align extra_meta to X's index once
        if extra_meta is not None:
            extra_aligned = extra_meta.reindex(X.index).fillna(0.0).values.astype(np.float32)
            oof_extra = np.zeros((n_rows, extra_aligned.shape[1]), dtype=np.float32)
        else:
            extra_aligned = None
            oof_extra = None

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)
        X_np = X.values
        y_np = y.values

        print(f"Ensemble: generating out-of-fold predictions "
              f"({self.n_folds} folds × {n_models} models)...")

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            for m_i, model in enumerate(self.base_models):
                # Clone-like: build a fresh instance of the same type+params
                fresh = model.__class__(**_get_init_params(model))
                fresh.train(X_tr, y_tr)
                proba = fresh.predict_proba(X_val)        # (n_val, 3)
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)
                oof_stack[val_idx, m_i*3 : m_i*3+3] = proba

            if oof_extra is not None:
                oof_extra[val_idx] = extra_aligned[val_idx]

            print(f"  fold {fold_i+1}/{self.n_folds} done")

        # Build meta-feature matrix
        meta_X = self._build_meta_features(
            oof_stack,
            X if self.use_original else None,
            oof_extra,
        )

        # Train meta-learner
        print("Training meta-learner...")
        self._meta = _build_meta_model(self.meta_model_type)
        self._meta.fit(meta_X, y_np)

        # Retrain all base models on the full dataset
        print("Retraining base models on full training set...")
        for model in self.base_models:
            model.train(X, y)

        print(f"Ensemble training complete. "
              f"Meta features: {meta_X.shape[1]}  "
              f"Meta model: {self.meta_model_type}")
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        extra_meta: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Returns shape (n_rows, 3) — columns: [P_buy, P_hold, P_sell].
        For single-row input, returns shape (3,).

        Blend mode: weighted average of base model probabilities.
        Stack mode: passes stacked probabilities through the meta-learner.

        Parameters
        ----------
        extra_meta : same extra columns passed at train() time, aligned to X.index.
                     (Stack mode only — ignored in blend mode.)
        """
        # ── Blend mode ────────────────────────────────────────────────────────
        if self.mode == "blend":
            total = sum(self._weights)
            blended = None
            for model, w in zip(self.base_models, self._weights):
                p = model.predict_proba(X)
                if p.ndim == 1:
                    p = p.reshape(1, -1)
                blended = (blended + p * w) if blended is not None else p * w
            result = blended / total
            return result[0] if len(X) == 1 else result

        # ── Stack mode (existing logic) ───────────────────────────────────────
        if self._meta is None:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        n_models  = len(self.base_models)
        n_rows    = len(X)
        stack     = np.zeros((n_rows, n_models * 3), dtype=np.float32)

        for m_i, model in enumerate(self.base_models):
            proba = model.predict_proba(X)
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            stack[:, m_i*3 : m_i*3+3] = proba

        extra_arr = None
        if extra_meta is not None:
            extra_arr = extra_meta.reindex(X.index).fillna(0.0).values.astype(np.float32)

        meta_X = self._build_meta_features(stack, X if self.use_original else None, extra_arr)
        raw    = self._meta.predict_proba(meta_X)   # sklearn order: sorted classes

        # Reorder to [P_buy, P_hold, P_sell]
        class_list = list(self._classes)
        idx_buy  = class_list.index(1)  if 1  in class_list else None
        idx_hold = class_list.index(0)  if 0  in class_list else None
        idx_sell = class_list.index(-1) if -1 in class_list else None

        ordered = np.zeros((len(raw), 3))
        if idx_buy  is not None: ordered[:, 0] = raw[:, idx_buy]
        if idx_hold is not None: ordered[:, 1] = raw[:, idx_hold]
        if idx_sell is not None: ordered[:, 2] = raw[:, idx_sell]

        return ordered[0] if n_rows == 1 else ordered

    def save(self, path: str | Path = "data/models/ensemble.joblib") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_models":      self.base_models,
            "meta":             self._meta,
            "meta_model_type":  self.meta_model_type,
            "feature_names":    self._feature_names,
            "classes":          self._classes,
            "trained_on":       self._trained_on,
            "n_folds":          self.n_folds,
            "use_original":     self.use_original,
            "extra_meta_cols":  getattr(self, "_extra_meta_cols", []),
            "mode":             self.mode,
            "weights":          self._weights,
        }
        joblib.dump(payload, path)
        print(f"Ensemble saved → {path}")

    def load(self, path: str | Path = "data/models/ensemble.joblib") -> "Ensemble":
        payload = joblib.load(path)
        self.base_models       = payload["base_models"]
        self._meta             = payload["meta"]
        self.meta_model_type   = payload["meta_model_type"]
        self._feature_names    = payload["feature_names"]
        self._classes          = payload["classes"]
        self._trained_on       = payload.get("trained_on", "")
        self.n_folds           = payload.get("n_folds", 5)
        self.use_original      = payload.get("use_original", False)
        self._extra_meta_cols  = payload.get("extra_meta_cols", [])
        self.mode              = payload.get("mode", "stack")
        self._weights          = payload.get("weights", [1.0] * len(self.base_models))
        print(f"Ensemble loaded ← {path}")
        return self

    def metadata(self) -> dict:
        base_names = [type(m).__name__ for m in self.base_models]
        return {
            "name":       "Ensemble",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "mode":            self.mode,
                "base_models":     base_names,
                "weights":         self._weights,
                "meta_model":      self.meta_model_type,
                "n_folds":         self.n_folds,
                "use_original":    self.use_original,
            },
        }

    # ── Inspection ────────────────────────────────────────────────────────────

    def model_weights(self) -> pd.DataFrame:
        """
        For logistic-regression meta-learner: show the weight each base model
        gets for each output class. Helps understand which model the ensemble
        trusts most.
        """
        if self._meta is None:
            raise RuntimeError("Ensemble not trained.")
        if not hasattr(self._meta, "coef_"):
            raise RuntimeError("model_weights() only supported for logistic meta-learner.")

        n_models = len(self.base_models)
        names    = [type(m).__name__ for m in self.base_models]
        cols     = []
        for name in names:
            cols += [f"{name}_P_buy", f"{name}_P_hold", f"{name}_P_sell"]

        classes = [f"class_{c}" for c in self._classes]
        df = pd.DataFrame(self._meta.coef_, index=classes, columns=cols[:self._meta.coef_.shape[1]])
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_meta_features(
        self,
        stack: np.ndarray,
        X_orig: Optional[pd.DataFrame] = None,
        extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        parts = [stack]
        if X_orig is not None:
            parts.append(X_orig.values)
        if extra is not None:
            parts.append(extra)
        return np.hstack(parts) if len(parts) > 1 else stack


# ── Module-level helpers ──────────────────────────────────────────────────────

def _build_meta_model(model_type: str):
    t = model_type.lower()
    if t == "logistic":
        return LogisticRegression(
            max_iter=1000, C=1.0, multi_class="multinomial", solver="lbfgs"
        )
    if t in ("lightgbm", "lgbm"):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
    raise ValueError(f"Unknown meta model type: '{model_type}'. Choose: logistic | lightgbm")


def _get_init_params(model: ModelInterface) -> dict:
    """Extract constructor kwargs from a model instance for cloning."""
    import inspect
    sig    = inspect.signature(model.__class__.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(model, name):
            params[name] = getattr(model, name)
        elif param.default is not inspect.Parameter.empty:
            params[name] = param.default
    return params
