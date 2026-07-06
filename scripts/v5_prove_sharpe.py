#!/usr/bin/env python
"""Write a leakage/Sharpe proof report for a V5 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.leakage_report import build_leakage_proof, write_leakage_proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--signals", default=None)
    parser.add_argument("--stress-run-dir", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = write_leakage_proof(
        args.run_dir,
        signal_path=args.signals,
        stress_run_dirs=args.stress_run_dir,
        out_path=args.out,
    )
    proof = build_leakage_proof(
        args.run_dir,
        signal_path=args.signals,
        stress_run_dirs=args.stress_run_dir,
    )
    print(
        f"leakage proof: {proof['status']} "
        f"symbol={proof['symbol']} sharpe={proof['run_stats']['sharpe']:.3f} "
        f"oos_rows={proof['oos_predictions']['rows']} out={out}"
    )


if __name__ == "__main__":
    main()
