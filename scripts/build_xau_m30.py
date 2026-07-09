"""Resample XAUUSD M15 -> M30 in the same schema the fade backtest expects."""
from __future__ import annotations

import pandas as pd

SRC = "data/XAUUSD_M15_long.csv"
DST = "data/XAUUSD_M30_long.csv"

d = pd.read_csv(SRC, parse_dates=["time"]).set_index("time").sort_index()

m30 = d.resample("30min", label="left", closed="left").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    tick_volume=("tick_volume", "sum"),
    spread=("spread", "mean"),
    real_volume=("real_volume", "sum"),
).dropna(subset=["open", "high", "low", "close"])

# A bar built from a single M15 leg spans only half the period — drop those so
# every M30 bar is a true two-leg aggregate (guards weekend/holiday stubs).
legs = d.resample("30min", label="left", closed="left").size()
m30 = m30[legs.reindex(m30.index) == 2]

m30.reset_index().to_csv(DST, index=False)
print(f"wrote {DST}: {len(m30)} bars  {m30.index.min()} -> {m30.index.max()}")
print(f"  (from {len(d)} M15 bars; ratio {len(d)/len(m30):.2f})")
print(f"  spread median {m30.spread.median():.2f} col-units -> ${m30.spread.median()*0.10:.3f}")
