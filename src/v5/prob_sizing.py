"""Sizing policies for the XAUUSD probabilistic-sizing program — V5.

Pure functions mapping a per-trade signal (calibrated win-probability, or an
ex-ante volatility forecast) to a RISK MULTIPLIER applied on top of the engine's
native per-trade risk. They are used post-hoc on realized R-multiples in the
research harness (identical-metric comparison, exactly as the 2026-07-05 meta
experiment did) so the live engine stays untouched until a policy passes the
promotion gate.

  prob_to_risk_mult  — monotone linear map P(win) -> risk multiplier
  prob_gate          — skip (mult 0) trades below a probability threshold
  kelly_mult         — fractional-Kelly multiplier from P(win) and payoff ratio
  (vol targeting lives in src/features/vol_forecast.py::vol_target_scale)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.vol_forecast import vol_target_scale  # re-export

__all__ = ["prob_to_risk_mult", "prob_gate", "kelly_mult", "vol_target_scale"]


def prob_to_risk_mult(
    p: pd.Series,
    *,
    lo: float = 0.2,
    hi: float = 0.8,
    out_lo: float = 0.5,
    out_hi: float = 1.5,
    neutral: float = 1.0,
) -> pd.Series:
    """Linear map of P(win) in [lo,hi] -> [out_lo,out_hi], clipped. NaN->neutral.

    Same functional form as the prior experiment's variant B, parameterized. A
    flat probability series therefore collapses to a near-constant multiplier —
    which is precisely why the harness pairs this with a constant-multiplier
    control: if the model has no information, this cannot beat the control.
    """
    span = hi - lo
    if span <= 0:
        raise ValueError("hi must exceed lo")
    mult = ((p - lo) / span) * (out_hi - out_lo) + out_lo
    return mult.clip(out_lo, out_hi).fillna(neutral)


def prob_gate(p: pd.Series, threshold: float, *, neutral: float = 1.0) -> pd.Series:
    """Trade-selection: multiplier 0 where P(win) < threshold, else `neutral`.

    NaN probabilities keep `neutral` (trade normally) so pre-OOS trades are not
    silently skipped.
    """
    taken = (p >= threshold)
    return pd.Series(np.where(p.isna(), neutral,
                              np.where(taken, neutral, 0.0)),
                     index=p.index, dtype=float)


def kelly_mult(
    p: pd.Series,
    payoff_ratio: float,
    *,
    fraction: float = 0.5,
    cap: float = 1.5,
    neutral: float = 1.0,
) -> pd.Series:
    """Fractional-Kelly multiplier from P(win) and payoff ratio b (avg win/loss).

    Kelly f* = p - (1-p)/b, clipped to [0, cap], scaled by `fraction`. NaN->neutral.
    A non-positive edge yields 0 (stand aside). `payoff_ratio` should be the
    historical |avg winning R| / |avg losing R| measured causally by the caller.
    """
    b = max(float(payoff_ratio), 1e-6)
    f = p - (1.0 - p) / b
    mult = (fraction * f).clip(lower=0.0, upper=cap)
    return mult.fillna(neutral)
