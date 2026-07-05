"""Unit guards for the V5 basket runner's journaling and action logic."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.trade_journal import TradeJournal

spec = importlib.util.spec_from_file_location(
    "v5_basket_runner", ROOT / "scripts" / "v5_basket_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_action_transitions():
    assert runner._action(0.0, 0.0) == "flat"
    assert runner._action(0.0, 0.5) == "OPEN"
    assert runner._action(0.5, 0.0) == "CLOSE"
    assert runner._action(0.5, -0.5) == "FLIP"
    assert runner._action(0.5, 0.8) == "ADD"
    assert runner._action(0.8, 0.5) == "TRIM"
    assert runner._action(0.5, 0.5) == "hold"


def test_journal_intents_writes_dry_run_rows(tmp_path):
    journal = TradeJournal(tmp_path / "j.db")
    actions = {"GOLD": "OPEN", "EURUSD": "TRIM", "SPX": "hold", "WTI": "flat"}
    targets = {"GOLD": 0.3, "EURUSD": -0.12, "SPX": 0.2, "WTI": 0.0}
    n = runner.journal_intents(journal, "2026-07-05", actions, targets,
                               run_id="test-run", magic=360500)
    assert n == 2  # hold/flat are not journaled
    rows = journal.get_trades(bot="v5_basket_runner")
    assert len(rows) == 2
    assert set(rows["run_id"]) == {"test-run"}
    assert (rows["dry_run"] == 1).all()
    assert (rows["magic"] == 360500).all()
    gold = rows[rows["symbol"] == "XAUUSD"].iloc[0]
    assert gold["direction"] == "buy"
    eur = rows[rows["symbol"] == "EURUSD"].iloc[0]
    assert eur["direction"] == "sell"


def test_champion_config_loads_and_is_consistent():
    champ = runner.load_champion_config()
    assert champ["tier"] == "basket10-btc"
    assert set(champ["instruments"]) <= set(runner.MT5_MAP)
    assert champ["run_id_pure"] != champ["run_id_overlay"]
    assert champ["lever_cfg"]["regime"] == "none"
