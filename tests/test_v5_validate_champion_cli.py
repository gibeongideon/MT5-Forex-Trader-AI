import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.v5_validate_champion import _broker_rules, _broker_symbol


def _args(**overrides):
    defaults = {
        "spread_pips": None,
        "commission_pips": None,
        "slippage_pips": None,
        "entry_delay_bars": None,
        "max_lot": None,
        "broker_symbol": None,
        "broker_symbol_suffix": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_broker_rules_use_symbol_defaults_when_no_overrides():
    rules = _broker_rules("EURUSD", _args())

    assert rules.pip_size == 0.0001
    assert rules.spread_pips == 1.0
    assert rules.commission_pips == 0.5
    assert rules.slippage_pips == 0.3
    assert rules.entry_delay_bars == 1
    assert rules.max_lot == 0.50


def test_broker_rules_apply_cli_sensitivity_overrides():
    rules = _broker_rules(
        "USDJPY",
        _args(
            spread_pips=2.5,
            commission_pips=0.7,
            slippage_pips=0.8,
            entry_delay_bars=2,
            max_lot=0.25,
        ),
    )

    assert rules.pip_size == 0.01
    assert rules.spread_pips == 2.5
    assert rules.commission_pips == 0.7
    assert rules.slippage_pips == 0.8
    assert rules.entry_delay_bars == 2
    assert rules.max_lot == 0.25
    assert rules.min_lot == 0.01
    assert rules.lot_step == 0.01


def test_broker_symbol_uses_explicit_value_before_suffix():
    assert _broker_symbol("USDJPY", _args(broker_symbol="USDJPY.Z", broker_symbol_suffix=".pro")) == "USDJPY.Z"


def test_broker_symbol_can_apply_suffix():
    assert _broker_symbol("EURUSD", _args(broker_symbol_suffix=".Z")) == "EURUSD.Z"
