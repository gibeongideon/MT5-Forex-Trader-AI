"""Leakage/ordering guards for the standalone XAUUSD H4 trade engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v5.xau_trend import (confidence_bucket, run_trades, wilder_atr,
                              xau_signal, _round_lot)


def _df(n=3000, seed=5, drift=2e-4):
    rng = np.random.default_rng(seed)
    close = 2000.0 * np.exp(np.cumsum(rng.normal(drift, 3e-3, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="4h")
    o = np.roll(close, 1)
    o[0] = close[0]
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 1e-3, n)))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 1e-3, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "spread": 3.4}, index=idx)


def test_confidence_buckets():
    assert confidence_bucket(0.6) == "low"
    assert confidence_bucket(-1.2) == "med"
    assert confidence_bucket(2.0) == "high"


def test_round_lot():
    assert _round_lot(0.004) == 0.0        # below min
    assert _round_lot(0.017) == 0.02
    assert _round_lot(-0.017) == -0.02
    assert _round_lot(999.0) == 20.0       # capped


def test_atr_is_causal():
    df = _df()
    base = wilder_atr(df, 14)
    mutated = df.copy()
    mutated.iloc[2000:] *= 1.3
    changed = wilder_atr(mutated, 14)
    pd.testing.assert_series_equal(base.iloc[:2000], changed.iloc[:2000])


def test_engine_is_causal_under_future_mutation():
    df = _df()
    cutoff = 2500
    base = run_trades(df.iloc[:cutoff], exit_mode="trail")
    mutated = df.copy()
    mutated.iloc[cutoff:] *= 1.2
    changed = run_trades(mutated.iloc[:cutoff], exit_mode="trail")
    pd.testing.assert_frame_equal(base["trades"], changed["trades"])
    pd.testing.assert_series_equal(base["equity"], changed["equity"])


def test_equity_reconciles_with_trade_pnl():
    df = _df()
    res = run_trades(df, equity0=5000.0, exit_mode="trail")
    # trade records round pnl to cents; allow the accumulated rounding
    assert res["equity"].dropna().iloc[-1] == pytest.approx(
        5000.0 + res["trades"]["pnl"].sum(), abs=0.01 * len(res["trades"]))


def test_sl_first_when_bar_hits_both():
    # deterministic uptrend so the engine is long, then one giant-range bar
    n = 2600
    idx = pd.date_range("2022-01-01", periods=n, freq="4h")
    close = 2000.0 * (1.0002 ** np.arange(n))
    o = np.roll(close, 1)
    o[0] = close[0]
    h = np.maximum(o, close) + 0.5
    l = np.minimum(o, close) - 0.5
    wide = 2550
    h[wide] = close[wide] + 400.0   # touches any plausible TP
    l[wide] = close[wide] - 400.0   # touches any plausible SL
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                       "spread": 3.4}, index=idx)
    res = run_trades(df, exit_mode="sltp", params={"enter_thresh": 0.0})
    trades = res["trades"]
    at_wide = trades[trades["close_time"] == idx[wide]]
    assert len(at_wide) == 1
    assert at_wide.iloc[0]["exit_reason"] in ("stop_loss", "trail_stop")


def test_flip_mode_confidence_never_flips_on_weak_signal():
    df = _df(seed=9, drift=0.0)
    conf = run_trades(df, exit_mode="flip", flip_mode="confidence")
    alwa = run_trades(df, exit_mode="flip", flip_mode="always")
    # 'always' can only flip at least as often as 'confidence'
    n_flip_conf = (conf["trades"]["exit_reason"] == "flip").sum()
    n_flip_alwa = (alwa["trades"]["exit_reason"] == "flip").sum()
    assert n_flip_alwa >= n_flip_conf


def test_costs_hurt():
    df = _df()
    cheap = run_trades(df, exit_mode="trail",
                       params={"spread_cost_mult": 0.0, "slippage_pips": 0.0})
    dear = run_trades(df, exit_mode="trail",
                      params={"spread_cost_mult": 3.0, "slippage_pips": 3.0})
    assert dear["equity"].dropna().iloc[-1] < cheap["equity"].dropna().iloc[-1]


def test_invalid_modes_raise():
    df = _df(n=300)
    with pytest.raises(ValueError):
        run_trades(df, exit_mode="bogus")
    with pytest.raises(ValueError):
        run_trades(df, flip_mode="bogus")
