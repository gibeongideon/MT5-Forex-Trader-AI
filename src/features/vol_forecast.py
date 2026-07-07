"""Ex-ante volatility forecasting for XAUUSD H4 — V5 Track 2.

Volatility is far more predictable than direction (volatility clustering is one
of the most robust stylized facts in finance). Sizing by an ex-ante volatility
forecast — targeting constant risk per trade in *return* space rather than the
engine's current fixed-%-over-ATR — is a well-established Sharpe lever that does
NOT require predicting price direction, so it stays inside the
`AGENT_INSTRUCTIONS.MD` allow-list.

Two forecasters, both causal (value at bar t uses only bars <= t-1):
  ewma_vol   — RiskMetrics-style EWMA of squared returns (one parameter, lambda)
  har_rv     — Heterogeneous Auto-Regressive Realized Volatility (Corsi 2009):
               regress next-bar RV on (RV_bar, RV_week, RV_month) averages.
               Fit fold-locally by the caller; `har_rv_features` returns the
               design matrix + target so the fit stays leakage-safe.

All outputs are in per-bar return units (fraction), aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ewma_vol(close: pd.Series, lam: float = 0.94, min_periods: int = 20) -> pd.Series:
    """RiskMetrics EWMA volatility forecast for bar t, using returns <= t-1.

    sigma^2_t = lam * sigma^2_{t-1} + (1-lam) * r^2_{t-1}
    Implemented as an EWM of the *lagged* squared return so row t is strictly
    past-only (no in-sample leakage of the current bar's move).
    """
    r2 = log_returns(close).pow(2)
    var = r2.shift(1).ewm(alpha=1.0 - lam, min_periods=min_periods).mean()
    return np.sqrt(var)


# Bars-per-day on H4 (6 bars) → "week"/"month" horizons in Corsi's HAR.
_H4_DAY = 6
_H4_WEEK = _H4_DAY * 5
_H4_MONTH = _H4_DAY * 22


def _rv(close: pd.Series) -> pd.Series:
    """Per-bar realized variance proxy = squared log return."""
    return log_returns(close).pow(2)


def har_rv_features(
    close: pd.Series,
    *,
    day: int = _H4_DAY,
    week: int = _H4_WEEK,
    month: int = _H4_MONTH,
) -> pd.DataFrame:
    """Causal HAR-RV design matrix + target (all in variance units).

    Columns: rv_d, rv_w, rv_m  (averages of past realized variance over the
    day/week/month windows, each ending at bar t-1) and `target` = realized
    variance at bar t. The caller fits a linear model on train rows and predicts
    test rows; because the regressors are strictly lagged, a fit on any prefix
    never sees future data. Rows with NaN regressors/target are the caller's to
    drop per fold.
    """
    rv = _rv(close)
    lagged = rv.shift(1)
    feat = pd.DataFrame(index=close.index)
    feat["rv_d"] = lagged.rolling(day, min_periods=day).mean()
    feat["rv_w"] = lagged.rolling(week, min_periods=week).mean()
    feat["rv_m"] = lagged.rolling(month, min_periods=month).mean()
    feat["target"] = rv
    return feat


def har_fit_predict(feat: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    """Fit HAR-RV OLS on `train_mask` rows, predict volatility for all rows.

    Uses numpy least squares (statsmodels not required). Returns a volatility
    (sqrt of predicted variance, clipped >=0) Series aligned to `feat.index`.
    Non-negative variance is enforced; degenerate folds fall back to the train
    mean RV.
    """
    cols = ["rv_d", "rv_w", "rv_m"]
    train = feat[train_mask].dropna(subset=cols + ["target"])
    if len(train) < len(cols) + 2:
        fallback = float(np.sqrt(max(feat.loc[train_mask, "target"].mean(), 0.0)) or 0.0)
        return pd.Series(fallback, index=feat.index)
    A = np.column_stack([np.ones(len(train)), train[cols].to_numpy(float)])
    b = train["target"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    X = feat[cols].to_numpy(float)
    pred_var = coef[0] + X @ coef[1:]
    pred_var = np.where(np.isfinite(pred_var), pred_var, np.nan)
    pred_var = np.clip(pred_var, 0.0, None)
    return pd.Series(np.sqrt(pred_var), index=feat.index)


def vol_target_scale(
    sigma: pd.Series,
    target_vol: float,
    *,
    floor: float = 0.25,
    cap: float = 3.0,
) -> pd.Series:
    """Risk multiplier that targets `target_vol` per-bar return vol.

    mult = clip(target_vol / sigma, floor, cap). Where sigma is missing/zero,
    the multiplier is 1.0 (fall back to the engine's native sizing). The cap/
    floor bound leverage so a quiet-vol spike cannot blow up position size.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = target_vol / sigma
    mult = mult.replace([np.inf, -np.inf], np.nan)
    return mult.clip(floor, cap).fillna(1.0)
