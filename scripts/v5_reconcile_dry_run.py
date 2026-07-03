#!/usr/bin/env python
"""Reconcile a V5 expected replay bundle against a PipelineBot dry-run journal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.dry_run_reconciliation import (
    reconcile_dry_run_journal,
    validate_live_dry_run_journal,
    write_reconciliation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="V5 run directory with trades.csv/settings.json")
    parser.add_argument(
        "--journal",
        default=str(ROOT / "data" / "live_trades.db"),
        help="SQLite live_trades.db or CSV journal export",
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--tolerance-seconds", type=int, default=60)
    parser.add_argument("--confidence-tolerance", type=float, default=0.05)
    parser.add_argument("--lot-tolerance", type=float, default=1e-9)
    parser.add_argument("--allow-mismatch", action="store_true", help="Write mismatch report and exit 0")
    parser.add_argument("--live-current", action="store_true", help="Validate current live dry-run journal rows by run id")
    parser.add_argument("--run-id", default=None, help="Run id to validate in live-current mode")
    parser.add_argument("--broker-symbol", default=None)
    parser.add_argument("--magic", type=int, default=None)
    parser.add_argument("--expected-sl-pips", type=float, default=None)
    parser.add_argument("--expected-tp-pips", type=float, default=None)
    parser.add_argument("--max-lot", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.live_current:
        required = {
            "--run-id": args.run_id,
            "--broker-symbol": args.broker_symbol,
            "--magic": args.magic,
            "--expected-sl-pips": args.expected_sl_pips,
            "--expected-tp-pips": args.expected_tp_pips,
            "--max-lot": args.max_lot,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"--live-current requires {', '.join(missing)}")
        report = validate_live_dry_run_journal(
            journal=Path(args.journal),
            run_id=args.run_id,
            broker_symbol=args.broker_symbol,
            magic_number=args.magic,
            expected_sl_pips=args.expected_sl_pips,
            expected_tp_pips=args.expected_tp_pips,
            max_lot=args.max_lot,
            min_confidence=args.min_confidence,
            fail_on_missing=not args.allow_mismatch,
        )
        out = Path(args.out) if args.out else run_dir / "live_dry_run_validation.json"
        write_reconciliation_report(report, out)
        print(
            f"live dry-run validation: {report['status']} "
            f"checked={report['checked_orders']} out={out}"
        )
        return

    report = reconcile_dry_run_journal(
        expected_run_dir=run_dir,
        journal=Path(args.journal),
        tolerance_seconds=args.tolerance_seconds,
        confidence_tolerance=args.confidence_tolerance,
        lot_tolerance=args.lot_tolerance,
        fail_on_mismatch=not args.allow_mismatch,
    )
    out = Path(args.out) if args.out else run_dir / "dry_run_reconciliation.json"
    write_reconciliation_report(report, out)
    print(
        f"dry-run reconciliation: {report['status']} "
        f"matched={report['matched_orders']}/{report['expected_orders']} out={out}"
    )


if __name__ == "__main__":
    main()
