"""
Download OHLCV data from MT5 and save to data/{SYMBOL}_{TIMEFRAME}.csv

Usage:
    conda activate envmt5
    python scripts/download_data.py
    python scripts/download_data.py --symbol GBPUSD --timeframe H1 --bars 20000

Prerequisites:
    MT5 terminal + bridge must be running first:  ./start_mt5.sh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mt5_connector import MT5Connector

DATA_DIR = Path(__file__).parent.parent / "data"


def download(symbol: str, timeframe: str, bars: int) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{symbol}_{timeframe}.csv"

    print(f"Connecting to MT5...")
    with MT5Connector() as conn:
        print(f"Downloading {bars:,} bars of {symbol} {timeframe}...")
        df = conn.get_rates(symbol, timeframe, count=bars)

    df.to_csv(out)
    print(f"Saved {len(df):,} bars → {out}")
    print(f"Range: {df.index[0]}  →  {df.index[-1]}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download MT5 historical data to CSV")
    p.add_argument("--symbol",    default="EURUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--bars",      type=int, default=50000,
                   help="Number of bars to download (default 50000 ≈ ~1.5 yrs of M15)")
    args = p.parse_args()

    download(args.symbol, args.timeframe, args.bars)
