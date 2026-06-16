"""Data layer — canonical aligned daily panels (close, spread) + quality guard.

One canonical close/spread panel (not per-instrument ad-hoc reads) eliminates a class
of alignment bugs. Daily bar (from resample_tf D1, label=left/closed=left): timestamp =
the UTC day; close = last M15 close in that day. A position formed from data ≤ that close
is held over the NEXT day's bar (enforced by positions.shift(1) in pnl.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.universe import UNIVERSE

DATA = ROOT / "data"


def _load(alias: str, tf: str) -> pd.DataFrame | None:
    f = DATA / f"{alias}_{tf}_long.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    return d.sort_index()


def build_panels(aliases, tf: str = "D1", min_bars: int = 300):
    """Return (close_panel, spread_panel, kept_aliases). Applies quality guards:
    enough history, non-zero spread, no all-NaN."""
    closes, spreads, kept = {}, {}, []
    for a in aliases:
        d = _load(a, tf)
        if d is None or len(d) < min_bars:
            print(f"  [panel] skip {a}: missing or <{min_bars} bars"); continue
        if d["spread"].abs().sum() == 0:
            print(f"  [panel] skip {a}: spread all-zero (untradeable cost model)"); continue
        closes[a] = d["close"]; spreads[a] = d["spread"]; kept.append(a)
    if not kept:
        raise RuntimeError("no instruments passed the data-quality guard")
    close = pd.DataFrame(closes).sort_index()
    spread = pd.DataFrame(spreads).reindex(close.index)
    # align to a WEEKDAY grid: crypto trades weekends → union index would inject Sat/Sun
    # rows where every other instrument is NaN, corrupting returns/vol/sizing.
    wd = close.index.dayofweek < 5
    close, spread = close[wd], spread[wd]
    # drop near-empty rows (holidays): require >50% of instruments to have data
    keep_rows = close.notna().mean(axis=1) > 0.5
    return close[keep_rows], spread[keep_rows], kept


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(1) - 1.0


def pip_series(aliases) -> pd.Series:
    # spread column is stored in price units (cost_bps/1e4 * close) → pip=1.0 so
    # pnl cost = turnover * spread / close = turnover * cost_bps/1e4.
    return pd.Series({a: 1.0 for a in aliases})


def asset_classes(aliases) -> dict:
    return {a: UNIVERSE[a]["asset_class"] for a in aliases}
