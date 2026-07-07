"""Fold-local OOS meta-probabilities for the XAUUSD engine — V5 Track 1.

Meta-labeling (López de Prado): the validated EWMAC engine decides side/timing;
this predicts, per engine trade, P(the trade wins) out-of-sample, so it can only
resize or skip — never pick direction. Extends the rejected 2026-07-05 attempt
in two ways the pre-registration requires:
  1. a CALIBRATED ENSEMBLE (xgboost + lightgbm + logistic, isotonic/sigmoid)
     instead of a lone XGBoost;
  2. a paired FEATURE-SET CONTROL — run the same folds on the prior XAU-only
     feature set to isolate whether the new exogenous columns add real AUC.

Leakage discipline mirrors `scripts/v5_xau_meta.py`: expanding yearly folds,
train only on trades CLOSED before the test window starts (purge), all
components fit inside the fold. OOS probabilities are NaN for pre-`start_year`
trades that never enter a test window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 42
_XGB = dict(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8,
            random_state=SEED, n_jobs=4, eval_metric="logloss")
_LGBM = dict(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8,
             random_state=SEED, n_jobs=4, verbose=-1)


@dataclass
class MetaOOSConfig:
    start_year: int = 2018
    min_train: int = 100
    calibrate: bool = True
    models: tuple[str, ...] = ("xgboost", "lightgbm", "logistic")
    seed: int = SEED
    max_folds: int | None = None      # cap yearly folds (smoke runs)


@dataclass
class MetaOOSResult:
    probs: pd.Series            # OOS P(win) aligned to trades.index (NaN pre-start)
    fold_aucs: list[float]
    mean_auc: float
    n_folds: int
    feature_names: list[str] = field(default_factory=list)


def _make_estimator(kind: str, calibrate: bool):
    from sklearn.calibration import CalibratedClassifierCV
    if kind == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(**_XGB)
        method = "isotonic"
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(**_LGBM)
        method = "isotonic"
    elif kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        method = "sigmoid"
    else:
        raise ValueError(f"unknown model {kind!r}")
    if calibrate:
        # Internal 3-fold calibration on the training slice only (no OOS leak).
        return CalibratedClassifierCV(est, method=method, cv=3)
    return est


def _fit_predict_proba(kind: str, Xtr, ytr, Xte, calibrate: bool) -> np.ndarray:
    # Calibration needs both classes present with enough support; fall back to
    # the raw estimator (or the base rate) on degenerate folds.
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    use_cal = calibrate and min(n_pos, n_neg) >= 3
    est = _make_estimator(kind, use_cal)
    # inf (e.g. ret/ATR when ATR->0) must map to 0, NOT float-max, or the
    # logistic scaler overflows when it squares them.
    Xtr = np.nan_to_num(np.asarray(Xtr, float), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(np.asarray(Xte, float), nan=0.0, posinf=0.0, neginf=0.0)
    est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)
    classes = list(getattr(est, "classes_", [0, 1]))
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(Xte))


def generate_meta_oos(
    X: pd.DataFrame,
    y: pd.Series,
    close_time: pd.Series,
    open_time: pd.Series,
    cfg: MetaOOSConfig | None = None,
) -> MetaOOSResult:
    """Expanding-yearly OOS ensemble P(win) for trade-indexed features.

    X, y, close_time, open_time are all aligned to the SAME trade index (one row
    per engine trade). Returns calibrated, blended OOS probabilities.
    """
    from sklearn.metrics import roc_auc_score

    cfg = cfg or MetaOOSConfig()
    idx = X.index
    probs = pd.Series(np.nan, index=idx, dtype=float)
    open_t = pd.to_datetime(open_time)
    close_t = pd.to_datetime(close_time)
    y = y.astype(int)
    fold_aucs: list[float] = []
    n_folds = 0

    max_year = int(close_t.dt.year.max())
    for year in range(cfg.start_year, max_year + 1):
        if cfg.max_folds is not None and n_folds >= cfg.max_folds:
            break
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        train_mask = (close_t < test_start).values
        test_mask = ((open_t >= test_start) & (open_t < test_end)).values
        if train_mask.sum() < cfg.min_train or test_mask.sum() == 0:
            continue
        Xtr, ytr = X[train_mask], y[train_mask].values
        Xte = X[test_mask]
        if len(np.unique(ytr)) < 2:      # single-class train fold → base rate
            probs.iloc[np.where(test_mask)[0]] = float(ytr.mean())
            continue
        blend = np.zeros(len(Xte), dtype=float)
        for kind in cfg.models:
            blend += _fit_predict_proba(kind, Xtr, ytr, Xte, cfg.calibrate)
        blend /= len(cfg.models)
        probs.iloc[np.where(test_mask)[0]] = blend
        n_folds += 1
        yte = y[test_mask].values
        if 0 < yte.sum() < len(yte):
            fold_aucs.append(float(roc_auc_score(yte, blend)))

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    return MetaOOSResult(probs=probs, fold_aucs=fold_aucs, mean_auc=mean_auc,
                         n_folds=n_folds, feature_names=list(X.columns))
