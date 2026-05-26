"""
Download OHLCV data from MT5 and save to data/{SYMBOL}_{TIMEFRAME}.csv

Usage:
    conda activate envmt5

    # Update M15 only (default)
    python scripts/download_data.py

    # Download all timeframes for EURUSD
    python scripts/download_data.py --all-timeframes

    # Custom symbol / timeframe
    python scripts/download_data.py --symbol GBPUSD --timeframe H1 --bars 20000

    # Multiple explicit timeframes
    python scripts/download_data.py --timeframes M5 M15 H1 H4 D1

Prerequisites:
    MT5 terminal + bridge must be running first:  ./start_mt5.sh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mt5_connector import MT5Connector

DATA_DIR = Path(__file__).parent.parent / "data"

# Default bar counts per timeframe — enough for ~2 years of meaningful history
_DEFAULT_BARS = {
    "M1":  100_000,   # ~69 days
    "M5":  100_000,   # ~347 days (~1 yr)
    "M15": 100_000,   # ~1,042 days (~2.9 yrs)
    "M30":  70_000,   # ~1,458 days (~4 yrs)
    "H1":   50_000,   # ~5,700 days (~15 yrs) — capped at broker depth
    "H4":   20_000,   # ~22 yrs
    "D1":    5_000,   # ~13.7 yrs
    "W1":    1_000,   # ~19 yrs
}

ALL_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


def download_one(conn, symbol: str, timeframe: str, bars: int) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{symbol}_{timeframe}.csv"

    print(f"  [{timeframe}] downloading {bars:,} bars...", end=" ", flush=True)
    df = conn.get_rates(symbol, timeframe, count=bars)
    df.to_csv(out)
    print(f"saved {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})  → {out.name}")
    return out


def main():
    p = argparse.ArgumentParser(description="Download MT5 historical data to CSV")
    p.add_argument("--symbol",         default="EURUSD")
    p.add_argument("--timeframe",      default="M15",
                   help="Single timeframe (ignored if --all-timeframes or --timeframes given)")
    p.add_argument("--timeframes",     nargs="+", metavar="TF",
                   help="Multiple timeframes, e.g. --timeframes M5 M15 H1 H4 D1")
    p.add_argument("--all-timeframes", action="store_true",
                   help=f"Download all standard timeframes: {ALL_TIMEFRAMES}")
    p.add_argument("--bars",           type=int, default=None,
                   help="Override bar count for all timeframes (default varies per timeframe)")
    args = p.parse_args()

    # Determine which timeframes to download
    if args.all_timeframes:
        timeframes = ALL_TIMEFRAMES
    elif args.timeframes:
        timeframes = args.timeframes
    else:
        timeframes = [args.timeframe]

    print(f"Connecting to MT5...")
    with MT5Connector() as conn:
        print(f"Connected. Downloading {args.symbol} — timeframes: {timeframes}\n")
        for tf in timeframes:
            bars = args.bars or _DEFAULT_BARS.get(tf, 50_000)
            try:
                download_one(conn, args.symbol, tf, bars)
            except Exception as e:
                print(f"  [{tf}] FAILED: {e}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
