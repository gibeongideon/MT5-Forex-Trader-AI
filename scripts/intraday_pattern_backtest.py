"""intraday_pattern_backtest.py — Phase B: backtest hour-anchored entry+exit patterns.

Pattern = enter at UTC hour i (direction from DISCOVER-period drift sign, no confirm peek),
exit at hour j (time exit) or on an ATR barrier. Net of REAL per-bar spread + commission.
Discover (2015-2021) / confirm (2022-2026) + block-bootstrap CI + GO gate.

GO = confirm net Sharpe >= +0.5 AND bootstrap CI lower bound > 0 AND positive in both
discover sub-halves. Cardinal rule: Sharpe >> 1 or hit ~100% => audit (rollover artifacts!).

Usage:
    python scripts/intraday_pattern_backtest.py --symbol EURUSD --entry 7 --exit 16 --dc
    python scripts/intraday_pattern_backtest.py --symbol XAUUSD --window lon_ny --dc
    python scripts/intraday_pattern_backtest.py --symbol GBPUSD --entry 20 --exit 21 --dc  # rollover audit
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.bootstrap import block_bootstrap_sharpe

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}
COMM_PIPS = 0.5
# pre-registered windows (entry_hour, exit_hour) UTC
WINDOWS = {"london_open": (7, 16), "ny_open": (13, 21), "lon_ny": (13, 17),
           "tokyo": (0, 8), "rollover": (20, 21)}
SPLIT = pd.Timestamp("2022-01-01")


def _hourly(sym):
    d = pd.read_csv(ROOT / "data" / f"{sym}_M15_long.csv", index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    o = d["close"].resample("1h")
    h = pd.DataFrame({"close": o.last(), "spread": d["spread"].resample("1h").mean()}).dropna()
    h["hour"] = h.index.hour; h["day"] = h.index.normalize()
    return h


def _trades(h, entry_h, exit_h, direction, pip):
    """One trade/day: enter at entry_h close, exit at exit_h close. Net return (bp) per trade."""
    cl = h.pivot_table(index="day", columns="hour", values="close")
    sp = h.pivot_table(index="day", columns="hour", values="spread")
    if entry_h not in cl.columns or exit_h not in cl.columns:
        return pd.Series(dtype=float)
    gross = (cl[exit_h] / cl[entry_h] - 1.0) * direction
    cost = (sp[entry_h].fillna(sp[entry_h].mean()) + sp[exit_h].fillna(sp[exit_h].mean())) * pip / cl[entry_h] \
           + 2 * COMM_PIPS * pip / cl[entry_h]
    return (gross - cost).dropna()           # net daily return (fraction)


def _orb_trades(sym, range_start, range_end, exit_h, pip):
    """Open-range breakout: range over [range_start, range_end) UTC; enter at the close of
    the first M15 bar that breaks it (in break direction), exit at exit_h close. Net of spread."""
    d = pd.read_csv(ROOT / "data" / f"{sym}_M15_long.csv", index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    d["day"] = d.index.normalize(); d["hour"] = d.index.hour
    out = {}
    for day, g in d.groupby("day"):
        rng = g[(g["hour"] >= range_start) & (g["hour"] < range_end)]
        post = g[(g["hour"] >= range_end) & (g["hour"] < exit_h)]
        if len(rng) < 2 or len(post) < 2:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        entry = direction = None
        for ts, row in post.iterrows():
            if row["high"] > hi:
                entry, direction, e_spr = row["close"], +1, row["spread"]; break
            if row["low"] < lo:
                entry, direction, e_spr = row["close"], -1, row["spread"]; break
        if entry is None:
            continue
        x = post.iloc[-1]
        gross = direction * (x["close"] / entry - 1.0)
        cost = (e_spr + x["spread"]) * pip / entry + 2 * COMM_PIPS * pip / entry
        out[day] = gross - cost
    return pd.Series(out)


def _stats(r, label):
    r = r.dropna()
    if len(r) < 40:
        print(f"  [{label}] too few trades ({len(r)})"); return None
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")
    lo, hi = block_bootstrap_sharpe(r.values, block=5)
    hit = (r > 0).mean() * 100
    eq = (1 + r).cumprod(); dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    print(f"  [{label}] Sharpe={sh:+.2f} (95%CI [{lo:+.2f},{hi:+.2f}])  hit={hit:.1f}%  "
          f"mean={r.mean()*1e4:+.2f}bp  trades={len(r)}  maxDD={dd:.1f}%")
    return sh, lo, hi


def run(sym, entry_h, exit_h, exit_mode, dc):
    h = _hourly(sym); pip = PIP[sym]
    # direction from DISCOVER drift only (no confirm peek)
    disc = h[h["day"] < SPLIT]
    d_long = _trades(disc, entry_h, exit_h, +1, pip)
    direction = +1 if d_long.mean() > 0 else -1
    dname = "LONG" if direction == +1 else "SHORT"
    print(f"\n=== {sym}  enter {entry_h:02d}UTC → exit {exit_h:02d}UTC  dir={dname} "
          f"(from discover)  exit={exit_mode} ===")
    full = _trades(h, entry_h, exit_h, direction, pip)
    if dc:
        _stats(full[full.index < SPLIT], "DISCOVER 2015-21")
        mid = pd.Timestamp("2018-07-01")
        _stats(full[(full.index < mid)], "  sub 2015-18")
        _stats(full[(full.index >= mid) & (full.index < SPLIT)], "  sub 2018-21")
        _stats(full[full.index >= SPLIT], "CONFIRM 2022-26")
    _stats(full, "FULL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD", choices=list(PIP))
    ap.add_argument("--entry", type=int, default=None)
    ap.add_argument("--exit", type=int, default=None)
    ap.add_argument("--window", default=None, choices=list(WINDOWS))
    ap.add_argument("--exit-mode", default="time", choices=["time"])  # atr: future extension
    ap.add_argument("--orb", action="store_true", help="open-range-breakout entry instead of fixed-time")
    ap.add_argument("--dc", action="store_true")
    args = ap.parse_args()
    if args.window:
        e, x = WINDOWS[args.window]
    else:
        e, x = args.entry, args.exit
    if e is None or x is None:
        print("specify --entry/--exit or --window"); sys.exit(1)
    if args.orb:
        # ORB: range over [e, e+1), break-entry, exit at x
        print(f"\n=== {args.symbol}  ORB range {e:02d}-{e+1:02d}UTC → exit {x:02d}UTC (break direction) ===")
        full = _orb_trades(args.symbol, e, e + 1, x, PIP[args.symbol])
        if args.dc:
            _stats(full[full.index < SPLIT], "DISCOVER 2015-21")
            _stats(full[full.index >= SPLIT], "CONFIRM 2022-26")
        _stats(full, "FULL")
    else:
        run(args.symbol, e, x, args.exit_mode, args.dc)
    print("Done.")


if __name__ == "__main__":
    main()
