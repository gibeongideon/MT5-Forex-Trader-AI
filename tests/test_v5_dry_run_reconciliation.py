import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.trade_journal import TradeJournal
from src.v5.dry_run_reconciliation import (
    DryRunReconciliationError,
    reconcile_dry_run_journal,
    validate_live_dry_run_journal,
)


def _expected_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "usdjpy-candidate"
    run_dir.mkdir()
    (run_dir / "settings.json").write_text(
        json.dumps(
            {
                "symbol": "USDJPY",
                "broker_symbol": "USDJPY.Z",
                "magic_number": 20260103,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
            }
        )
    )
    pd.DataFrame(
        [
            {
                "signal_time": "2026-01-01 00:00:00",
                "entry_time": "2026-01-01 00:15:00",
                "direction": "buy",
                "confidence": 0.72,
                "volume": 0.01,
                "symbol": "USDJPY.Z",
                "magic": 20260103,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
            },
            {
                "signal_time": "2026-01-01 01:00:00",
                "entry_time": "2026-01-01 01:15:00",
                "direction": "sell",
                "confidence": 0.81,
                "volume": 0.01,
                "symbol": "USDJPY.Z",
                "magic": 20260103,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
            },
        ]
    ).to_csv(run_dir / "trades.csv", index=False)
    return run_dir


def test_trade_journal_records_magic_and_dry_run_fields(tmp_path):
    journal = TradeJournal(tmp_path / "trades.db")

    journal.record(
        {
            "bot": "PipelineBot-USDJPY",
            "symbol": "USDJPY.Z",
            "direction": "buy",
            "entry_time": "2026-01-01 00:15:00",
            "entry_price": 150.01,
            "model": "candle_trail",
            "confidence": 0.72,
            "entry_reason": "trail:buy",
            "exit_reason": "dry_run_open",
            "volume": 0.01,
            "sl_pips": 10.0,
            "tp_pips": 30.0,
            "magic": 20260103,
            "run_id": "usdjpy-candidate",
            "dry_run": True,
        }
    )

    row = journal.get_trades().iloc[0]
    assert row["magic"] == 20260103
    assert row["run_id"] == "usdjpy-candidate"
    assert row["dry_run"] == 1


def test_reconcile_dry_run_journal_matches_expected_order_intents(tmp_path):
    run_dir = _expected_run(tmp_path)
    journal = pd.DataFrame(
        [
            {
                "symbol": "USDJPY.Z",
                "direction": "buy",
                "entry_time": "2026-01-01 00:15:20",
                "volume": 0.01,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
                "magic": 20260103,
                "confidence": 0.721,
            },
            {
                "symbol": "USDJPY.Z",
                "direction": "sell",
                "entry_time": "2026-01-01 01:15:10",
                "volume": 0.01,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
                "magic": 20260103,
                "confidence": 0.809,
            },
        ]
    )

    report = reconcile_dry_run_journal(
        expected_run_dir=run_dir,
        journal=journal,
        tolerance_seconds=60,
        confidence_tolerance=0.02,
    )

    assert report["status"] == "dry_run_reconciled"
    assert report["matched_orders"] == 2
    assert report["unmatched_expected"] == 0
    assert report["unexpected_journal"] == 0


def test_reconcile_dry_run_journal_rejects_wrong_side(tmp_path):
    run_dir = _expected_run(tmp_path)
    journal = pd.DataFrame(
        [
            {
                "symbol": "USDJPY.Z",
                "direction": "sell",
                "entry_time": "2026-01-01 00:15:00",
                "volume": 0.01,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
                "magic": 20260103,
                "confidence": 0.72,
            }
        ]
    )

    with pytest.raises(DryRunReconciliationError, match="unmatched"):
        reconcile_dry_run_journal(expected_run_dir=run_dir, journal=journal)


def test_reconcile_dry_run_journal_can_report_mismatch_without_raising(tmp_path):
    run_dir = _expected_run(tmp_path)
    journal = pd.DataFrame(columns=["symbol", "direction", "entry_time", "volume"])

    report = reconcile_dry_run_journal(
        expected_run_dir=run_dir,
        journal=journal,
        fail_on_mismatch=False,
    )

    assert report["status"] == "dry_run_mismatch"
    assert report["unmatched_expected"] == 2
    assert report["matched_orders"] == 0


def test_validate_live_dry_run_journal_accepts_current_order_intents():
    journal = pd.DataFrame(
        [
            {
                "symbol": "USDJPYr",
                "direction": "buy",
                "entry_time": "2026-07-03 16:00:00",
                "model": "candle_trail",
                "confidence": 0.72,
                "entry_reason": "trail:buy",
                "exit_reason": "dry_run_open",
                "volume": 0.01,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
                "magic": 20260103,
                "run_id": "usdjpy-live",
                "dry_run": 1,
            }
        ]
    )

    report = validate_live_dry_run_journal(
        journal=journal,
        run_id="usdjpy-live",
        broker_symbol="USDJPYr",
        magic_number=20260103,
        expected_sl_pips=10.0,
        expected_tp_pips=30.0,
        max_lot=0.01,
    )

    assert report["status"] == "live_dry_run_validated"
    assert report["checked_orders"] == 1
    assert report["checks"]["symbol"] == "pass"
    assert report["checks"]["magic"] == "pass"
    assert report["checks"]["dry_run"] == "pass"


def test_validate_live_dry_run_journal_reports_waiting_when_no_orders():
    report = validate_live_dry_run_journal(
        journal=pd.DataFrame(columns=["run_id"]),
        run_id="usdjpy-live",
        broker_symbol="USDJPYr",
        magic_number=20260103,
        expected_sl_pips=10.0,
        expected_tp_pips=30.0,
        max_lot=0.01,
        fail_on_missing=False,
    )

    assert report["status"] == "waiting_for_live_dry_run_order"
    assert report["checked_orders"] == 0


def test_validate_live_dry_run_journal_rejects_wrong_symbol():
    journal = pd.DataFrame(
        [
            {
                "symbol": "USDJPY.Z",
                "direction": "buy",
                "entry_time": "2026-07-03 16:00:00",
                "model": "candle_trail",
                "confidence": 0.72,
                "entry_reason": "trail:buy",
                "exit_reason": "dry_run_open",
                "volume": 0.01,
                "sl_pips": 10.0,
                "tp_pips": 30.0,
                "magic": 20260103,
                "run_id": "usdjpy-live",
                "dry_run": 1,
            }
        ]
    )

    with pytest.raises(DryRunReconciliationError, match="symbol"):
        validate_live_dry_run_journal(
            journal=journal,
            run_id="usdjpy-live",
            broker_symbol="USDJPYr",
            magic_number=20260103,
            expected_sl_pips=10.0,
            expected_tp_pips=30.0,
            max_lot=0.01,
        )
