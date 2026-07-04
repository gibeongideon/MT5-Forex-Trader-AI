#!/usr/bin/env python
"""Wait for a V5 live dry-run order row, then run live-current validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.live_gate import (
    build_live_current_validation_command,
    count_run_orders,
    wait_for_run_orders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--broker-symbol", required=True)
    parser.add_argument("--magic", type=int, required=True)
    parser.add_argument("--expected-sl-pips", type=float, required=True)
    parser.add_argument("--expected-tp-pips", type=float, required=True)
    parser.add_argument("--max-lot", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-polls", type=int, default=240)
    args = parser.parse_args()

    result = wait_for_run_orders(
        lambda: count_run_orders(args.journal, args.run_id),
        poll_seconds=args.poll_seconds,
        max_polls=args.max_polls,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "orders_found":
        raise SystemExit(2)

    command = build_live_current_validation_command(
        run_dir=args.run_dir,
        journal=args.journal,
        run_id=args.run_id,
        broker_symbol=args.broker_symbol,
        magic=args.magic,
        expected_sl_pips=args.expected_sl_pips,
        expected_tp_pips=args.expected_tp_pips,
        max_lot=args.max_lot,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
