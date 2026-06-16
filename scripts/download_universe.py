"""Download the CTA daily universe from Yahoo Finance → data/{alias}_D1_long.csv.

Schema matches the resampled _long files: time,open,high,low,close,tick_volume,
spread,real_volume. `spread` is stored in PRICE units = cost_bps/1e4 * close so the
P&L cost model (with pip=1.0) reduces to turnover * cost_bps/1e4 — a conservative
fixed daily transaction cost (Yahoo daily has no bid/ask).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.universe import UNIVERSE

DATA = ROOT / "data"
START = "2008-01-01"


def fetch_one(alias: str, spec: dict, retries: int = 3) -> int:
    import yfinance as yf
    for attempt in range(retries):
        try:
            d = yf.download(spec["ticker"], start=START, progress=False,
                            auto_adjust=True, threads=False)
            if d is None or len(d) == 0:
                time.sleep(2); continue
            if isinstance(d.columns, pd.MultiIndex):      # flatten yf multiindex
                d.columns = d.columns.get_level_values(0)
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            d.index.name = "time"
            d["tick_volume"] = d["volume"].fillna(0).astype("int64")
            d["spread"] = (spec["cost_bps"] / 1e4) * d["close"]   # price-unit cost
            d["real_volume"] = 0
            out = d[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
            path = DATA / f"{alias}_D1_long.csv"
            out.to_csv(path)
            print(f"  {alias:7s} ({spec['ticker']:9s}): {len(out):,} bars "
                  f"{out.index[0].date()} → {out.index[-1].date()}  → {path.name}", flush=True)
            return len(out)
        except Exception as e:
            print(f"  {alias}: attempt {attempt+1} failed: {str(e)[:70]}", flush=True)
            time.sleep(3)
    print(f"  {alias}: FAILED after {retries} tries", flush=True)
    return 0


def main():
    print(f"\n=== CTA UNIVERSE DOWNLOAD ({len(UNIVERSE)} instruments, Yahoo daily) ===")
    ok = 0
    for alias, spec in UNIVERSE.items():
        if fetch_one(alias, spec) > 0:
            ok += 1
    print(f"\nDone: {ok}/{len(UNIVERSE)} instruments downloaded.")


if __name__ == "__main__":
    main()
