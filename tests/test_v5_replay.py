import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.replay import replay_signal_frame
from src.v5.validation import BrokerExecutionRules


def _prices():
    idx = pd.date_range("2026-01-01 00:00", periods=5, freq="15min")
    return pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            "high": [1.1005, 1.1015, 1.1045, 1.1035, 1.1045],
            "low": [1.0995, 1.1005, 1.1015, 1.1025, 1.1035],
            "close": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
        },
        index=idx,
    )


def test_replay_enters_after_configured_delay_and_normalizes_lot():
    prices = _prices()
    signals = pd.DataFrame(
        {
            "signal": ["buy", "hold", "hold", "hold", "hold"],
            "confidence": [0.8, 0.0, 0.0, 0.0, 0.0],
            "requested_lot": [0.026, 0.0, 0.0, 0.0, 0.0],
        },
        index=prices.index,
    )
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.0,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )

    result = replay_signal_frame(
        prices,
        signals,
        rules,
        sl_pips=10,
        tp_pips=30,
        initial_balance=10_000,
    )

    trade = result.trades[0]
    assert trade["entry_time"] == prices.index[1]
    assert trade["entry_price"] == pytest.approx(1.1011)
    assert trade["volume"] == 0.02
    assert trade["exit_reason"] == "tp"
    assert trade["pnl_pips"] == pytest.approx(28.5)
    assert result.equity.iloc[-1] > 10_000


def test_replay_skips_trade_when_lot_rounds_below_minimum():
    prices = _prices()
    signals = pd.DataFrame(
        {
            "signal": ["sell", "hold", "hold", "hold", "hold"],
            "confidence": [0.9, 0.0, 0.0, 0.0, 0.0],
            "requested_lot": [0.004, 0.0, 0.0, 0.0, 0.0],
        },
        index=prices.index,
    )
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.0,
        entry_delay_bars=0,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )

    result = replay_signal_frame(prices, signals, rules, sl_pips=10, tp_pips=20)

    assert result.trades == []
    assert result.equity.iloc[-1] == 10_000
