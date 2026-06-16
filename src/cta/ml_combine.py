"""ML forecast combination (lever #4) — blend factor forecasts into one, strict OOS.

Anti-overfit design:
  • Features are KNOWN weak factors (EWMAC speeds, xsmom, short-term reversal, vol regime),
    not raw prices — the model only learns how to WEIGHT them.
  • POOLED across all instruments (one model) → ~200k samples, momentum is universal.
  • RIDGE (L2) — essentially optimal factor-blend weights; minimal overfit surface.
  • WALK-FORWARD monthly retrain on an EXPANDING past window, with the forward-return
    target PURGED so training never peeks past the retrain date.
Output = per-instrument combined forecast → fed to cluster_risk_weights/vol_target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ANN = np.sqrt(252)


def _ewmac_speed(close, fast, slow, cap=20.0, target=10.0):
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    raw = (close.ewm(span=fast, min_periods=fast).mean()
           - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
    scalar = target / raw.abs().expanding(min_periods=60).mean().shift(1)
    return (raw * scalar).clip(-cap, cap) / target


def build_features(close, returns):
    """Dict of lookahead-free factor panels (each uses data <= t only)."""
    from src.cta.signals import xsmom
    vol = returns.shift(1).ewm(halflife=42, min_periods=20).std()
    feats = {}
    for f, s in [(8, 32), (16, 64), (32, 128), (64, 256)]:
        feats[f"ewmac_{f}_{s}"] = _ewmac_speed(close, f, s)
    feats["xsmom"] = xsmom(close)
    feats["rev5"] = -(close / close.shift(5) - 1.0) / (vol * np.sqrt(5))   # 1-wk reversal
    feats["vol_z"] = (vol.sub(vol.mean(axis=1), axis=0)).div(vol.std(axis=1), axis=0)  # vol regime
    return feats


def ml_forecast(close, returns, horizon: int = 21, alpha: float = 10.0,
                min_train_rows: int = 8000, retrain: str = "ME") -> pd.DataFrame:
    """Walk-forward ridge combination → forecast panel (held between monthly retrains)."""
    feats = build_features(close, returns)
    names = list(feats)
    long = pd.concat({n: feats[n].stack(future_stack=True) for n in names}, axis=1)
    long.columns = names
    vol = returns.shift(1).ewm(halflife=42, min_periods=20).std()
    fwd = close.shift(-horizon) / close - 1.0                      # FUTURE return (label only)
    long["y"] = (fwd / (vol * np.sqrt(horizon))).clip(-3, 3).stack(future_stack=True)
    long = long.dropna()
    ddates = long.index.get_level_values(0)

    fc = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    purge = pd.Timedelta(days=int(horizon * 1.6))                  # exclude unrealized targets
    for t in close.resample(retrain).last().index:
        cut = t - purge
        tr = long[ddates <= cut]
        if len(tr) < min_train_rows:
            continue
        m = Ridge(alpha=alpha, solver="svd").fit(tr[names].values, tr["y"].values)
        avail = close.index[close.index <= t]
        if len(avail) == 0:
            continue
        t_use = avail[-1]
        row = np.column_stack([feats[n].loc[t_use].values for n in names])
        valid = ~np.isnan(row).any(axis=1)
        pred = np.full(close.shape[1], np.nan)
        if valid.any():
            pred[valid] = m.predict(row[valid])
        fc.loc[t_use] = pred
    return fc.ffill()
