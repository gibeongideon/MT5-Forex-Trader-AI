import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.broker_checks import (
    BrokerCheckError,
    BrokerProfile,
    build_broker_reconciliation,
    resolve_broker_symbol,
)
from src.v5.validation import BrokerExecutionRules


def test_resolve_broker_symbol_prefers_tradable_visible_shortest_suffix():
    profile = BrokerProfile(
        base_symbol="USDJPY",
        tradable_symbols=[
            {"name": "USDJPY", "trade_mode": 0, "visible": True},
            {"name": "USDJPY.pro", "trade_mode": 4, "visible": False},
            {"name": "USDJPY.Z", "trade_mode": 4, "visible": True},
        ],
    )

    assert resolve_broker_symbol(profile) == "USDJPY.Z"


def test_build_broker_reconciliation_flags_magic_and_stop_distance():
    rules = BrokerExecutionRules(
        pip_size=0.01,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.3,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )
    profile = BrokerProfile(
        base_symbol="USDJPY",
        broker_symbol="USDJPY.Z",
        magic_number=20260103,
        min_stop_distance_pips=5.0,
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "USDJPY.Z",
                "magic": 20260103,
                "direction": "buy",
                "volume": 0.01,
                "entry_price": 150.00,
                "sl": 149.90,
                "tp": 150.30,
            }
        ]
    )

    report = build_broker_reconciliation(profile, trades, rules)

    assert report["status"] == "broker_profile_checked"
    assert report["broker_symbol"] == "USDJPY.Z"
    assert report["magic_number"] == 20260103
    assert report["checked_trades"] == 1
    assert report["checks"]["symbol"] == "pass"
    assert report["checks"]["magic"] == "pass"
    assert report["checks"]["stop_distance"] == "pass"


def test_build_broker_reconciliation_rejects_wrong_magic():
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.3,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )
    profile = BrokerProfile(base_symbol="EURUSD", broker_symbol="EURUSD", magic_number=20260102)
    trades = pd.DataFrame(
        [
            {
                "symbol": "EURUSD",
                "magic": 999,
                "direction": "sell",
                "volume": 0.01,
                "entry_price": 1.1000,
                "sl": 1.1010,
                "tp": 1.0970,
            }
        ]
    )

    with pytest.raises(BrokerCheckError, match="magic"):
        build_broker_reconciliation(profile, trades, rules)


def test_build_broker_reconciliation_uses_initial_stop_levels_for_trailing_trades():
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.3,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )
    profile = BrokerProfile(
        base_symbol="EURUSD",
        broker_symbol="EURUSD.Z",
        magic_number=20260102,
        min_stop_distance_pips=5.0,
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "EURUSD.Z",
                "magic": 20260102,
                "direction": "buy",
                "volume": 0.01,
                "entry_price": 1.1000,
                "initial_sl": 1.0990,
                "initial_tp": 1.1030,
                "sl": 1.1002,
                "tp": 1.1030,
            }
        ]
    )

    report = build_broker_reconciliation(profile, trades, rules)

    assert report["checks"]["stop_distance"] == "pass"
