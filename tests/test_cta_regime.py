"""Tests for src/cta/regime.py — lookahead-free guarantee + none-identity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cta.regime import trend_gate, vol_gate, regime_gate


def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    cols = ["A", "B", "C"]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=cols)
    close = 100 * (1 + rets).cumprod()
    signals = pd.DataFrame(rng.choice([-1.0, 0.0, 1.0], (n, 3)), index=idx, columns=cols)
    return close, rets, signals


def test_regime_none_is_identity():
    close, rets, sig = _panel()
    out = regime_gate(close, rets, sig, mode="none")
    pd.testing.assert_frame_equal(out, sig)


def test_trend_gate_zeroes_counter_trend():
    # strict uptrend → a SHORT signal must be gated to 0; a LONG signal kept
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
    sig = pd.DataFrame({"A": [-1.0] * 300}, index=idx)        # short into an uptrend
    gated = trend_gate(close, sig, sma_window=200)
    assert (gated.iloc[250:]["A"] == 0).all()                # after warmup, counter-trend zeroed
    long_sig = pd.DataFrame({"A": [1.0] * 300}, index=idx)
    assert (trend_gate(close, long_sig, 200).iloc[250:]["A"] == 1.0).all()


def test_gates_are_lookahead_free():
    """Perturbing ONLY the last row must not change any earlier gate output."""
    close, rets, sig = _panel(seed=1)
    t0 = trend_gate(close, sig)
    v0 = vol_gate(rets)

    close2 = close.copy(); close2.iloc[-1] *= 1.5            # shock the final bar only
    rets2 = rets.copy();   rets2.iloc[-1] += 0.20
    t1 = trend_gate(close2, sig)
    v1 = vol_gate(rets2)

    pd.testing.assert_frame_equal(t0.iloc[:-1], t1.iloc[:-1])
    pd.testing.assert_frame_equal(v0.iloc[:-1], v1.iloc[:-1])


def test_vol_gate_in_unit_interval():
    close, rets, sig = _panel(seed=2)
    mult = vol_gate(rets)
    assert ((mult >= 0.0) & (mult <= 1.0)).all().all()
