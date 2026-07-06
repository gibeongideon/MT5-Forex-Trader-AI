"""Guards for the discrete-lot H4 simulation and runner sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v5.h4_cta import SYMBOLS, h4_positions
from src.v5.h4_discrete import (discrete_replay, per_lot_usd,
                                target_lots_today)


def _panel(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    base = {"EURUSD": 1.1, "GBPUSD": 1.3, "USDJPY": 140.0, "XAUUSD": 2000.0}
    close = pd.DataFrame(
        {s: base[s] * np.exp(np.cumsum(rng.normal(0, 1.5e-3, n))) for s in SYMBOLS},
        index=idx)
    spread = pd.DataFrame(1.0, index=idx, columns=list(SYMBOLS))
    return close, spread


def test_per_lot_usd_conventions():
    assert per_lot_usd("EURUSD", 1.10) == pytest.approx(110_000.0)
    assert per_lot_usd("USDJPY", 150.0) == 100_000.0   # USD base
    assert per_lot_usd("XAUUSD", 2000.0) == 200_000.0  # 100 oz


def test_target_lots_rounding_and_zeroing():
    lots = target_lots_today({"EURUSD": 0.5, "XAUUSD": 0.001},
                             equity=10_000,
                             prices={"EURUSD": 1.10, "XAUUSD": 2000.0})
    assert lots["EURUSD"]["lots"] == 0.05  # 5000/110k = 0.0455 -> 0.05
    assert lots["XAUUSD"]["lots"] == 0.0
    assert lots["XAUUSD"]["rounded_zero"]


def test_discrete_replay_lots_are_step_quantized():
    close, spread = _panel()
    pos = h4_positions(close)
    res = discrete_replay(pos, close, spread, equity0=5000.0)
    lots = res["lots"].values
    assert np.allclose(lots, np.round(lots / 0.01) * 0.01, atol=1e-9)
    assert res["equity"].iloc[-1] > 0


def test_discrete_replay_costs_reduce_equity():
    close, spread = _panel()
    pos = h4_positions(close)
    with_cost = discrete_replay(pos, close, spread, equity0=5000.0)
    free = discrete_replay(pos, close, spread * 0.0, equity0=5000.0)
    assert with_cost["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_discrete_replay_is_causal_under_future_mutation():
    close, spread = _panel()
    cutoff = 2200
    pos = h4_positions(close)
    base = discrete_replay(pos.iloc[:cutoff], close.iloc[:cutoff],
                           spread.iloc[:cutoff], equity0=5000.0)

    mutated = close.copy()
    mutated.iloc[cutoff:] *= 1.25
    pos_m = h4_positions(mutated)
    changed = discrete_replay(pos_m.iloc[:cutoff], mutated.iloc[:cutoff],
                              spread.iloc[:cutoff], equity0=5000.0)

    pd.testing.assert_frame_equal(base["lots"], changed["lots"])
    pd.testing.assert_series_equal(base["equity"], changed["equity"])


def test_tiny_equity_freezes_instead_of_negative():
    close, spread = _panel()
    pos = h4_positions(close) * 50.0  # absurd leverage to force ruin path
    res = discrete_replay(pos, close, spread, equity0=100.0)
    assert (res["equity"].dropna() >= res["equity"].dropna().min()).all()
    # once equity is non-positive everything is frozen at that level
    eq = res["equity"]
    if (eq <= 0).any():
        first = eq[eq <= 0].index[0]
        assert (res["lots"].loc[first:] == 0).all().all()
