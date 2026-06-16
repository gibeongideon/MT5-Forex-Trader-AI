"""
resample_tf.py — Resample deep M15 _long.csv files up to H1 / H4 (or any rule).

Reuses the same CSV schema (time,open,high,low,close,tick_volume,spread,real_volume).
OHLC aggregated correctly, volume summed, spread averaged. UTC preserved.

Usage:
    python scripts/resample_tf.py --symbol EURUSD --tf H1
    python scripts/resample_tf.py --symbol EURUSD --tf H4
    python scripts/resample_tf.py --all --tf H1     # all *_M15_long.csv present
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RULE = {"H1": "1h", "H4": "4h", "M30": "30min"}


def resample(symbol: str, tf: str) -> Path | None:
    src = DATA / f"{symbol}_M15_long.csv"
    if not src.exists():
        print(f"  {symbol}: {src.name} missing — skip"); return None
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    o = df.resample(RULE[tf], label="left", closed="left")
    out = pd.DataFrame({
        "open": o["open"].first(), "high": o["high"].max(),
        "low": o["low"].min(), "close": o["close"].last(),
        "tick_volume": o["tick_volume"].sum(),
        "spread": o["spread"].mean().round(1),
        "real_volume": o["real_volume"].sum() if "real_volume" in df else 0,
    }).dropna(subset=["open"])
    out.index.name = "time"
    dst = DATA / f"{symbol}_{tf}_long.csv"
    out.to_csv(dst)
    print(f"  {symbol} {tf}: {len(df):,} M15 → {len(out):,} {tf} bars  "
          f"{out.index[0]} → {out.index[-1]}  spread={out['spread'].mean():.2f}p  → {dst.name}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tf", default="H1", choices=list(RULE))
    args = ap.parse_args()
    if args.all:
        syms = sorted({p.name.split("_")[0] for p in DATA.glob("*_M15_long.csv")})
    else:
        syms = [args.symbol]
    for s in syms:
        resample(s, args.tf)
    print("Done.")


if __name__ == "__main__":
    main()
