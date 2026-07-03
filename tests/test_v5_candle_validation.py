import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.candle_validation import (
    V5CandleTrailValidationConfig,
    replay_candle_trail,
    run_candle_trail_validation,
)
from src.v5.champion_validation import default_broker_rules_for_symbol
from src.v5.validation import BrokerExecutionRules


def _prices(n=24):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 1.1000 + np.arange(n) * 0.0002
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0008,
            "low": close - 0.0003,
            "close": close,
            "tick_volume": np.arange(n) + 100,
        },
        index=idx,
    )


def test_candle_trail_validation_writes_lumibot_style_artifacts(tmp_path):
    prices = _prices()
    signals = pd.DataFrame(
        {
            "P_buy": [0.75, 0.20, 0.10, 0.76, 0.20, 0.10] * 4,
            "P_hold": [0.10] * len(prices),
            "P_sell": [0.15, 0.70, 0.20, 0.14, 0.70, 0.20] * 4,
        },
        index=prices.index,
    )
    cfg = V5CandleTrailValidationConfig(
        symbol="EURUSD",
        run_id="unit-candle-trail",
        artifact_root=tmp_path / "runs",
        broker_rules=default_broker_rules_for_symbol("EURUSD"),
        threshold=0.60,
        requested_lot=0.01,
        sl_pips=10,
        tp_pips=30,
        trail_activation_pips=4,
        trail_pips_behind=2,
        max_bars_low=1,
        max_bars_med=2,
        max_bars_high=4,
    )

    result = run_candle_trail_validation(cfg, prices=prices, signals=signals)

    assert result.run_dir == tmp_path / "runs" / "unit-candle-trail"
    assert result.stats["mode"] == "candle_trail"
    assert result.stats["trades"] > 0
    assert "sharpe" in result.stats
    assert "daily_sharpe" in result.stats
    assert (result.run_dir / "trades.csv").exists()
    settings = json.loads((result.run_dir / "settings.json").read_text())
    reconciliation = json.loads((result.run_dir / "reconciliation.json").read_text())
    trades = pd.read_csv(result.run_dir / "trades.csv")

    assert settings["mode"] == "candle_trail"
    assert reconciliation["status"] == "research_replay_only"
    assert len(trades) == result.stats["trades"]


def test_candle_trail_replay_applies_entry_delay():
    prices = _prices(8)
    signals = pd.DataFrame(
        {
            "P_buy": [0.90] + [0.0] * 7,
            "P_hold": [0.05] * 8,
            "P_sell": [0.05] * 8,
        },
        index=prices.index,
    )
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=0.0,
        commission_pips=0.0,
        slippage_pips=0.0,
        entry_delay_bars=2,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )

    replay = replay_candle_trail(
        prices,
        signals,
        rules,
        threshold=0.60,
        requested_lot=0.01,
        sl_pips=10,
        tp_pips=30,
        trail_activation_pips=40,
        trail_pips_behind=10,
        max_bars_low=1,
        max_bars_med=2,
        max_bars_high=4,
    )

    first_trade = replay["trades"][0]
    assert first_trade["signal_time"] == prices.index[0]
    assert first_trade["entry_time"] == prices.index[2]
    assert first_trade["entry_price"] == prices.iloc[2]["close"]
