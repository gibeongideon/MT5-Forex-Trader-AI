"""Live dry-run gate helpers for V5 candidate validation."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def count_run_orders(journal: pd.DataFrame | str | Path, run_id: str) -> int:
    """Count journal order-intent rows for a V5 dry-run id."""

    if isinstance(journal, pd.DataFrame):
        frame = journal.copy()
    else:
        with sqlite3.connect(Path(journal)) as conn:
            frame = pd.read_sql_query(
                "SELECT run_id FROM trades WHERE run_id = ?",
                conn,
                params=[str(run_id)],
            )
    if "run_id" not in frame.columns:
        return 0
    return int((frame["run_id"].astype(str) == str(run_id)).sum())


def wait_for_run_orders(
    counter: Callable[[], int],
    *,
    poll_seconds: float,
    max_polls: int,
) -> dict:
    """Poll until the dry-run journal has at least one order row."""

    last_count = 0
    for poll in range(1, max_polls + 1):
        last_count = int(counter())
        if last_count > 0:
            return {"status": "orders_found", "checked_orders": last_count, "polls": poll}
        if poll_seconds > 0 and poll < max_polls:
            time.sleep(poll_seconds)
    return {
        "status": "waiting_for_live_dry_run_order",
        "checked_orders": last_count,
        "polls": int(max_polls),
    }


def build_live_current_validation_command(
    *,
    run_dir: str | Path,
    journal: str | Path,
    run_id: str,
    broker_symbol: str,
    magic: int,
    expected_sl_pips: float,
    expected_tp_pips: float,
    max_lot: float,
    python_executable: str = "python",
) -> list[str]:
    """Build the strict live-current reconciliation command."""

    return [
        python_executable,
        "scripts/v5_reconcile_dry_run.py",
        "--live-current",
        "--run-dir",
        str(run_dir),
        "--journal",
        str(journal),
        "--run-id",
        str(run_id),
        "--broker-symbol",
        str(broker_symbol),
        "--magic",
        str(int(magic)),
        "--expected-sl-pips",
        str(expected_sl_pips),
        "--expected-tp-pips",
        str(expected_tp_pips),
        "--max-lot",
        str(max_lot),
    ]
