"""Regression guards for the V5 lever pipeline.

The critical invariant: with every lever at its default, `lever_positions`
must be byte-identical to the validated champion construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cta.signals import combine, ewmac, xsmom
from src.cta.strategy import rebalance_hold
from src.v5.h4_cta import buffer_band_causal, cluster_inv_vol, vol_target_h4
from src.v5.levers import SPEED_SETS, carry_signal, lever_positions

ANN = np.sqrt(252)
KEPT = ["EURUSD", "USDJPY", "GOLD", "SPX"]
CLASSES = {"EURUSD": "FX_USD", "USDJPY": "FX_USD", "GOLD": "METAL", "SPX": "EQ_INDEX"}


def _panel(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {a: 100.0 * np.exp(np.cumsum(rng.normal(2e-4, 0.008, n))) for a in KEPT},
        index=idx)


def test_all_levers_off_matches_champion_pipeline():
    close = _panel()
    got = lever_positions(close, KEPT, CLASSES)

    returns = close.pct_change(fill_method=None)
    sig = combine(ewmac(close, speeds=SPEED_SETS["slow"]), xsmom(close))
    raw = cluster_inv_vol(sig, returns, CLASSES, 0.10, 42, ann=ANN)
    pos = vol_target_h4(raw, returns, 0.10, 42, ann=ANN)
    pos = rebalance_hold(pos, "monthly")
    expected = buffer_band_causal(pos, 0.4)

    pd.testing.assert_frame_equal(got, expected)


def test_regime_none_is_identity():
    close = _panel()
    base = lever_positions(close, KEPT, CLASSES, {"regime": "none"})
    default = lever_positions(close, KEPT, CLASSES)
    pd.testing.assert_frame_equal(base, default)


def test_regime_gates_change_positions():
    close = _panel()
    base = lever_positions(close, KEPT, CLASSES)
    gated = lever_positions(close, KEPT, CLASSES, {"regime": "trend"})
    assert not base.equals(gated)


def test_carry_signal_zero_for_non_fx_and_validates_currencies():
    close = _panel()
    rates = pd.DataFrame(
        {"USD": 5.0, "EUR": 3.0, "JPY": 0.1},
        index=pd.date_range("2015-01-01", periods=140, freq="MS"))
    car = carry_signal(close, KEPT, rates)
    assert (car["GOLD"] == 0).all() and (car["SPX"] == 0).all()
    # sign convention: long the high yielder — USD>JPY => long USDJPY
    assert (car["USDJPY"].dropna() == 1.0).all()
    # EUR<USD => short EURUSD
    assert (car["EURUSD"].dropna() == -1.0).all()

    with pytest.raises(ValueError, match="missing currencies"):
        carry_signal(close, KEPT, rates.drop(columns=["JPY"]))


def test_ml_combine_emits_finite_positions():
    # leading NaN before the first monthly rebalance date is inherent to
    # rebalance_hold (champion path included); the backtest fills it with 0
    close = _panel(n=2000)
    pos = lever_positions(close, KEPT, CLASSES,
                          {"ml_combine": True, "ml_min_rows": 500}).dropna()
    assert len(pos) > 1000
    assert np.isfinite(pos.values).all()
    assert pos.abs().sum().sum() > 0  # the ridge path actually traded


def test_ml_combine_before_first_train_is_flat_not_nan():
    close = _panel(n=400)
    pos = lever_positions(close, KEPT, CLASSES,
                          {"ml_combine": True, "ml_min_rows": 10_000_000}).dropna()
    assert len(pos) > 300
    assert pos.abs().sum().sum() == 0.0
