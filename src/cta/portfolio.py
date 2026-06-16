"""Portfolio construction — inverse-vol sizing + portfolio vol-targeting.

Diagonal only (no covariance matrix) in v1: simpler, no rank-deficiency bugs.
Every vol estimate uses returns.shift(1) so TODAY's return never sizes today's
position — the classic CTA backtest lookahead trap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def inv_vol_weights(signals: pd.DataFrame, returns: pd.DataFrame,
                    target: float = 0.10, halflife: int = 42) -> pd.DataFrame:
    """Risk-budgeted positions: each active instrument gets an EQUAL daily-vol budget
    so position magnitudes are O(1) NAV fractions (not raw 1/sigma).
      per-instrument daily vol budget = (target/√252) / √N_active
      pos_i = signal_i * budget / sigma_i
    With ~uncorrelated instruments this already lands portfolio vol near `target`;
    vol_target() then corrects for realized correlation. All past-only (sigma uses shift(1))."""
    sigma = returns.shift(1).ewm(halflife=halflife, min_periods=20).std()
    n_active = signals.replace(0.0, np.nan).notna().sum(axis=1).clip(lower=1)
    budget = (target / ANN) / np.sqrt(n_active)          # per-instrument daily vol target
    pos = signals.mul(budget, axis=0) / sigma
    return pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def vol_target(positions: pd.DataFrame, returns: pd.DataFrame,
               target: float = 0.10, halflife: int = 42,
               max_lev: float = 3.0) -> pd.DataFrame:
    """Final scalar correction so trailing realized portfolio vol ≈ target (annualized).
    Past-only (shift(1)); per-instrument leverage from inv_vol_weights is already O(1),
    so the scalar k stays near 1 — capped at max_lev for safety."""
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    realized = port_ret.ewm(halflife=halflife, min_periods=20).std().shift(1) * ANN
    k = (target / realized).clip(upper=max_lev)
    k = k.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return positions.mul(k, axis=0)
