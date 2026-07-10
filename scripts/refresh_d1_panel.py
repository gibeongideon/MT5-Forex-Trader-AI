"""Refresh the daily CTA panel CSVs (`data/{alias}_D1_long.csv`) from Yahoo Finance.

The D1 panel that drives `lever_positions` is Yahoo data (see `src/cta/universe.py`),
NOT the MT5 feed — MT5 supplies execution prices, Yahoo supplies the signal history.
Schema matches `src/cta/panel.py::_load`:

    time,open,high,low,close,tick_volume,spread,real_volume

`spread` is the synthetic cost column = cost_bps/1e4 * close (price units), exactly as
`universe.py` documents. Existing rows are preserved; freshly downloaded rows win on
overlap, so a partial Yahoo outage cannot silently truncate history.

Usage:
    python scripts/refresh_d1_panel.py --aliases SILVER,UST10Y,SPX,WTI,USDJPY,BTC
    python scripts/refresh_d1_panel.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cta.universe import UNIVERSE

DATA = ROOT / "data"
COLS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def fetch(alias: str) -> pd.DataFrame:
    spec = UNIVERSE[alias]
    raw = yf.download(spec["ticker"], period="max", interval="1d",
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {alias} ({spec['ticker']})")
    if isinstance(raw.columns, pd.MultiIndex):          # yfinance>=0.2 single-ticker frame
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    d = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None))
    d.index.name = "time"
    for c in ("open", "high", "low", "close"):
        d[c] = raw[c].astype(float)
    d["tick_volume"] = raw.get("volume", 0).astype("float64").fillna(0)
    d["spread"] = spec["cost_bps"] / 1e4 * d["close"]
    d["real_volume"] = 0
    return d.dropna(subset=["close"])[COLS]


def merge_write(alias: str, fresh: pd.DataFrame) -> tuple[int, int, str]:
    path = DATA / f"{alias}_D1_long.csv"
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        old.columns = [c.lower() for c in old.columns]
        old.index.name = "time"
        before = len(old)
        merged = pd.concat([old[~old.index.isin(fresh.index)], fresh]).sort_index()
    else:
        before, merged = 0, fresh.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_csv(path)
    return before, len(merged), str(merged.index[-1].date())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aliases", default="SILVER,UST10Y,SPX,WTI,USDJPY,BTC")
    ap.add_argument("--all", action="store_true", help="refresh the whole UNIVERSE")
    args = ap.parse_args()

    aliases = list(UNIVERSE) if args.all else [a.strip() for a in args.aliases.split(",")]
    unknown = [a for a in aliases if a not in UNIVERSE]
    if unknown:
        sys.exit(f"unknown aliases (not in UNIVERSE): {unknown}")

    print(f"{'alias':8}{'ticker':10}{'rows':>14}{'last bar':>13}")
    failed = []
    for a in aliases:
        try:
            before, after, last = merge_write(a, fetch(a))
            print(f"{a:8}{UNIVERSE[a]['ticker']:10}{before:>6} -> {after:<6}{last:>13}")
        except Exception as e:  # noqa: BLE001 — report and continue; partial refresh is safe
            failed.append(a)
            print(f"{a:8}{UNIVERSE[a]['ticker']:10}{'FAILED':>14}  {e}")
    if failed:
        sys.exit(f"\nrefresh incomplete: {failed}")
    print("\nall aliases refreshed")


if __name__ == "__main__":
    main()
