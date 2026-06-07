"""
Download cross-market M15 data from MT5 — Phase 24 prerequisite.

Downloads GBPUSD, USDJPY, XAUUSD M15 into data/ alongside EURUSD_M15.csv.
These are used by src/features/cross_market.py to add new information
channels that enc8 cannot see (it only sees EURUSD OHLCV).

Run once. Requires MT5 terminal + bridge running first: ./start_mt5.sh

Usage:
    conda run -n envmt5 python scripts/download_crossmarket.py
    conda run -n envmt5 python scripts/download_crossmarket.py --bars 60000
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mt5_connector import MT5Connector

DATA_DIR = ROOT / "data"

SYMBOLS = [
    "GBPUSD",   # GBP/USD — USD institutional flow (same USD driver as EURUSD)
    "USDJPY",   # USD/JPY — risk-on/risk-off signal
    "XAUUSD",   # Gold    — Gold up → USD weak → EURUSD up
    "EURUSD",   # refresh EURUSD too
]

DEFAULT_BARS = 60_000  # ~625 days of M15 — matches EURUSD_M15.csv history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars",    type=int,  default=DEFAULT_BARS)
    p.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = p.parse_args()

    print(f"Connecting to MT5 bridge...")
    with MT5Connector() as conn:
        print(f"Connected.\n")
        for sym in args.symbols:
            out = DATA_DIR / f"{sym}_M15.csv"
            print(f"  {sym:8s} M15  {args.bars:,} bars ...", end=" ", flush=True)
            try:
                df = conn.get_rates(sym, "M15", count=args.bars)
                df.to_csv(out)
                print(f"saved {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})  → {out.name}")
            except Exception as e:
                print(f"FAILED: {e}")

    print("\nDone. Run scripts/compare_crosspair.py next.")


if __name__ == "__main__":
    main()
