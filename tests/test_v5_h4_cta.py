"""Leakage and cost guards for the V5 H4 CTA path.

The strategy has no fitted components, so the only leakage surface is
lookahead. The causality test mutates FUTURE bars and requires every
position up to the cutoff to remain bit-identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v5.h4_cta import SYMBOLS, h4_pnl, h4_positions

N_BARS = 4000
CUTOFF = 3000


def _synthetic_panel(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="4h")
    close = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(1e-4, 2e-3, N_BARS)))
         for s in SYMBOLS}, index=idx)
    spread = pd.DataFrame(1.0, index=idx, columns=list(SYMBOLS))
    return close, spread


def test_positions_are_causal_under_future_mutation():
    close, _ = _synthetic_panel()
    base = h4_positions(close)

    mutated = close.copy()
    mutated.iloc[CUTOFF:] *= (1.0 + np.random.default_rng(1).normal(0, 0.05, (N_BARS - CUTOFF, len(SYMBOLS))))
    changed = h4_positions(mutated)

    pd.testing.assert_frame_equal(base.iloc[:CUTOFF], changed.iloc[:CUTOFF])


def test_pnl_is_causal_under_future_mutation():
    close, spread = _synthetic_panel()
    base = h4_pnl(h4_positions(close), close, spread)

    mutated = close.copy()
    mutated.iloc[CUTOFF:] *= 1.10
    changed = h4_pnl(h4_positions(mutated), mutated, spread)

    # bar at CUTOFF uses the return INTO it, so everything strictly before is fixed
    pd.testing.assert_frame_equal(base.iloc[:CUTOFF], changed.iloc[:CUTOFF])


def test_costs_reduce_net_below_gross():
    close, spread = _synthetic_panel()
    pnl = h4_pnl(h4_positions(close), close, spread)
    assert pnl["cost"].min() >= 0.0
    assert pnl["cost"].sum() > 0.0
    assert pnl["net"].sum() < pnl["gross"].sum()


def test_doubled_costs_double_the_drag():
    close, spread = _synthetic_panel()
    pos = h4_positions(close)
    c1 = h4_pnl(pos, close, spread, spread_cost_mult=1.0)["cost"].sum()
    c2 = h4_pnl(pos, close, spread, spread_cost_mult=2.0)["cost"].sum()
    assert c2 == pytest.approx(2.0 * c1)


def test_zero_entry_delay_rejected():
    close, spread = _synthetic_panel()
    pos = h4_positions(close)
    with pytest.raises(ValueError):
        h4_pnl(pos, close, spread, entry_delay_bars=0)


def test_gross_pnl_uses_lagged_positions():
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")
    close = pd.DataFrame({s: [100.0, 101.0, 102.0, 101.0, 103.0] for s in SYMBOLS},
                         index=idx)
    spread = pd.DataFrame(0.0, index=idx, columns=list(SYMBOLS))
    pos = pd.DataFrame(0.0, index=idx, columns=list(SYMBOLS))
    pos.iloc[1] = 1.0  # decided at close of bar 1
    pnl = h4_pnl(pos, close, spread)
    # earns bar 2's return only (102/101-1 per symbol, 4 symbols)
    expected = (102.0 / 101.0 - 1.0) * len(SYMBOLS)
    assert pnl["gross"].iloc[2] == pytest.approx(expected)
    assert pnl["gross"].iloc[1] == 0.0
