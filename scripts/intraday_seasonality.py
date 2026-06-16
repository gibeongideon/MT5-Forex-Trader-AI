"""intraday_seasonality.py — Phase A: map recurring hour-of-day patterns (descriptive).

For EURUSD/GBPUSD/USDJPY/XAUUSD on deep M15 (UTC, real spread). All lookahead-free; times
UTC with EAT=UTC+3 shown. NO model — pure intraday seasonality:
  1. hour-of-day drift   : mean hourly return + hit-rate + t-stat per UTC hour
  2. entry→exit matrix   : Sharpe of (enter at hour i close → exit at hour j close), net spread
  3. open-range breakout : follow-through of a session-open range break
This identifies candidate "buy/sell at hour X, exit at hour Y" patterns for Phase B.

Honest prior: hour structure was ~random-walk; expect most cells to be noise. The 24×24
matrix is EXPLORATORY (multiple comparisons) — pre-registered windows are confirmed in Phase B.

Usage:
    python scripts/intraday_seasonality.py --symbol EURUSD
    python scripts/intraday_seasonality.py            # all four
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.session_analysis import EAT_OFFSET

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}
COMM_PIPS = 0.5


def _load(sym: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"{sym}_M15_long.csv", index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    return d.sort_index()


def _hourly(d: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 → hourly bars (UTC) with mean spread (pips) per hour."""
    o = d["close"].resample("1h")
    h = pd.DataFrame({"close": o.last(), "spread": d["spread"].resample("1h").mean()}).dropna()
    h["ret"] = h["close"].pct_change()
    h["hour"] = h.index.hour
    return h


def hour_drift(h: pd.DataFrame, pip: float, sym: str):
    """Mean hourly return, hit-rate, t-stat by UTC hour (net of spread cost estimate)."""
    print(f"\n  ── HOUR-OF-DAY DRIFT — {sym}  (close-to-close hourly return) ──")
    print(f"  {'UTC':>3} {'EAT':>3} {'n':>5} {'mean_bp':>8} {'t-stat':>7} {'hit%':>6} {'spread_p':>8}")
    rows = []
    for hr in range(24):
        g = h[h["hour"] == hr]["ret"].dropna()
        if len(g) < 100:
            continue
        mean_bp = g.mean() * 1e4
        t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if g.std(ddof=1) > 0 else 0.0
        hit = (g > 0).mean() * 100
        spr = h[h["hour"] == hr]["spread"].mean()
        eat = (hr + EAT_OFFSET) % 24
        print(f"  {hr:02d}  {eat:02d}  {len(g):>5} {mean_bp:>+8.2f} {t:>+7.2f} {hit:>5.1f}% {spr:>8.2f}")
        rows.append((hr, mean_bp, t, hit))
    # flag |t|>3 (still exploratory across 24 tests)
    strong = [r for r in rows if abs(r[2]) > 3]
    if strong:
        print("   strong-drift hours (|t|>3, exploratory): " +
              ", ".join(f"{r[0]:02d}UTC({r[1]:+.1f}bp,t={r[2]:+.1f})" for r in strong))
    else:
        print("   no hour with |t|>3 — no standout directional drift")


def entry_exit_matrix(h: pd.DataFrame, pip: float, sym: str):
    """Sharpe of long (enter hour i close → exit hour j close), net of round-trip spread.
    Built lookahead-free from same-day i<j pairs; reports the best few cells (exploratory)."""
    # pivot to day × hour close
    h = h.copy()
    h["day"] = h.index.normalize()
    wide = h.pivot_table(index="day", columns="hour", values="close")
    sprd = h.pivot_table(index="day", columns="hour", values="spread")
    hours = [c for c in range(24) if c in wide.columns]
    rt_cost_bp = lambda i, j: ((sprd[i].mean() + sprd[j].mean()) * pip / wide[i].mean()
                               + 2 * COMM_PIPS * pip / wide[i].mean()) * 1e4
    best = []
    for i in hours:
        for j in hours:
            if j <= i:
                continue
            r = (wide[j] / wide[i] - 1.0).dropna()
            if len(r) < 200:
                continue
            net = r * 1e4 - rt_cost_bp(i, j)            # bp per trade, net
            sd = net.std(ddof=1)
            if sd < 1e-9:
                continue
            sh = net.mean() / sd * np.sqrt(252)         # daily → annualized
            best.append((sh, i, j, net.mean(), (net > 0).mean() * 100, len(net)))
    best.sort(key=lambda x: -abs(x[0]))
    print(f"\n  ── ENTRY→EXIT matrix top cells — {sym}  (long i→j; negate Sharpe = short) ──")
    print(f"  {'Sharpe':>7} {'in':>3} {'out':>3} {'EATin':>5} {'EATout':>6} {'net_bp':>7} {'hit%':>6} {'n':>5}")
    for sh, i, j, mbp, hit, n in best[:8]:
        print(f"  {sh:>+7.2f} {i:02d}  {j:02d}   {(i+EAT_OFFSET)%24:02d}    {(j+EAT_OFFSET)%24:02d}   "
              f"{mbp:>+7.2f} {hit:>5.1f}% {n:>5}")
    print("   (EXPLORATORY — 24×24 cells; confirm only pre-registered windows OOS in Phase B)")


def orb(d: pd.DataFrame, pip: float, sym: str, sessions):
    """Open-range breakout follow-through: range over [open, open+1h); does a break of that
    range in the next bars continue to session-window end? Reports continuation rate."""
    print(f"\n  ── OPEN-RANGE BREAKOUT follow-through — {sym} ──")
    d = d.copy(); d["day"] = d.index.normalize(); d["hour"] = d.index.hour
    for name, o_start, win_end, sess_end in sessions:
        cont, n = 0, 0
        for day, g in d.groupby("day"):
            rng = g[(g["hour"] >= o_start) & (g["hour"] < win_end)]
            post = g[(g["hour"] >= win_end) & (g["hour"] < sess_end)]
            if len(rng) < 2 or len(post) < 2:
                continue
            hi, lo = rng["high"].max(), rng["low"].min()
            entry = post["close"].iloc[0]
            brk_up = (post["high"] > hi).any()
            brk_dn = (post["low"] < lo).any()
            end = post["close"].iloc[-1]
            if brk_up and not brk_dn:
                cont += 1 if end > hi else 0; n += 1
            elif brk_dn and not brk_up:
                cont += 1 if end < lo else 0; n += 1
        rate = cont / n * 100 if n else float("nan")
        print(f"  {name:14s} range {o_start:02d}-{win_end:02d}UTC → exit {sess_end:02d}UTC: "
              f"break-continues {rate:.0f}%  (n={n})  [>55% = momentum, <45% = fade]")


def run(sym: str):
    d = _load(sym); pip = PIP[sym]
    print(f"\n{'='*84}\n  INTRADAY SEASONALITY — {sym}  ({len(d):,} M15 bars, UTC; EAT=UTC+{EAT_OFFSET})")
    print(f"  {d.index[0].date()} → {d.index[-1].date()}\n{'='*84}")
    h = _hourly(d)
    hour_drift(h, pip, sym)
    entry_exit_matrix(h, pip, sym)
    orb(d, pip, sym, [("London", 7, 8, 16), ("NewYork", 13, 14, 21), ("Tokyo", 0, 1, 8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(PIP))
    args = ap.parse_args()
    for s in ([args.symbol] if args.symbol else list(PIP)):
        run(s)
    print("\nDone.")


if __name__ == "__main__":
    main()
