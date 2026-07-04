import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.live_gate import (
    build_live_current_validation_command,
    count_run_orders,
    wait_for_run_orders,
)


def test_count_run_orders_filters_by_run_id():
    journal = pd.DataFrame(
        [
            {"run_id": "target", "direction": "buy"},
            {"run_id": "other", "direction": "sell"},
            {"run_id": "target", "direction": "sell"},
        ]
    )

    assert count_run_orders(journal, "target") == 2


def test_wait_for_run_orders_returns_after_first_positive_poll():
    calls = []

    def counter():
        calls.append(1)
        return 1 if len(calls) == 3 else 0

    result = wait_for_run_orders(counter, poll_seconds=0.0, max_polls=5)

    assert result == {"status": "orders_found", "checked_orders": 1, "polls": 3}


def test_wait_for_run_orders_times_out_without_orders():
    result = wait_for_run_orders(lambda: 0, poll_seconds=0.0, max_polls=2)

    assert result == {"status": "waiting_for_live_dry_run_order", "checked_orders": 0, "polls": 2}


def test_build_live_current_validation_command_uses_expected_gate_arguments(tmp_path):
    command = build_live_current_validation_command(
        run_dir=tmp_path / "run",
        journal=tmp_path / "live.db",
        run_id="usdjpy-v5",
        broker_symbol="USDJPY.Z",
        magic=20260103,
        expected_sl_pips=10,
        expected_tp_pips=30,
        max_lot=0.01,
    )

    assert command[:4] == ["python", "scripts/v5_reconcile_dry_run.py", "--live-current", "--run-dir"]
    assert "--allow-mismatch" not in command
    assert command[command.index("--broker-symbol") + 1] == "USDJPY.Z"
    assert command[command.index("--run-id") + 1] == "usdjpy-v5"
    assert command[command.index("--max-lot") + 1] == "0.01"
