"""
session_profile.py — Phase A: raw, model-free hour-of-day / session pattern study.

Unlike session_analysis.py (which profiles the leaky candle MODEL's predictions),
this reads RAW OHLCV(+spread) only and characterises the market structure of each
UTC hour and trading session: volatility, drift, trend-vs-mean-reversion, breakout
follow-through, real spread cost, and simple baseline-rule net expectancy.

All lookahead-free: a bar's descriptors use only that bar and earlier; outcome
metrics (follow-through, baseline rules) enter at a bar's close and measure
SUBSEQUENT bars, with exits via the same ATR triple-barrier as the live harness.

Data: use the DEEP files (data/{SYM}_M15_long.csv) — UTC, with a real `spread`
column. The short files (data/{SYM}_M15.csv) are broker-time with spread=0 and are
NOT valid for hour/cost analysis.

Usage:
    python scripts/session_profile.py --symbol EURUSD
    python scripts/session_profile.py --symbol USDJPY --discover-confirm
    python scripts/session_profile.py            # all available deep files
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.features.indicators import atr
from src.features.meta_labels import side_barrier_meta_label
from scripts.session_analysis import (
    EAT_OFFSET, SESSION_DEFS, session_mask, eat_range, _load_raw,
)

# pip size + pre-registered confirmatory windows (UTC) per symbol
SYMBOL_CFG = {
    "EURUSD": dict(deep="data/EURUSD_M15_long.csv", short="data/EURUSD_M15.csv", pip=1e-4),
    "GBPUSD": dict(deep="data/GBPUSD_M15_long.csv", short="data/GBPUSD_M15.csv", pip=1e-4),
    "USDJPY": dict(deep="data/USDJPY_M15_long.csv", short="data/USDJPY_M15.csv", pip=1e-2),
    "XAUUSD": dict(deep="data/XAUUSD_M15_long.csv", short="data/XAUUSD_M15.csv", pip=1e-1),
}
PRIMARY_WINDOWS = {
    "EURUSD": [("London+NY", 13, 17)],
    "GBPUSD": [("London+NY", 13, 17)],
    "XAUUSD": [("London+NY", 13, 17)],
    "USDJPY": [("Asian", 0, 6), ("Tokyo+London", 8, 9)],
}
COMM_PIPS = 0.5
TP_MULT = SL_MULT = 1.0
BARRIER_H = 8


# ── return / structure helpers (lookahead-free) ─────────────────────────────────

def _prep(df: pd.DataFrame, pip: float) -> pd.DataFrame:
    df = df.copy()
    df["r"]      = np.log(df["close"] / df["close"].shift(1))      # bar return
    df["ret_pip"] = (df["close"] - df["close"].shift(1)) / pip
    df["rng_pip"] = (df["high"] - df["low"]) / pip
    df["atr"]    = atr(df, 14)
    df["atr_pip"] = df["atr"] / pip
    gap = df.index.to_series().diff()
    df["gap_ok"] = (gap == pd.Timedelta(minutes=15))
    # contiguous-pair flag for lag-1 autocorr (last two gaps both 15min)
    df["pair_ok"] = df["gap_ok"] & df["gap_ok"].shift(1)
    return df


def _ac1(d: pd.DataFrame, mask: pd.Series) -> float:
    m = mask & d["pair_ok"]
    if m.sum() < 30:
        return float("nan")
    a = d.loc[m, "r"].values
    b = d["r"].shift(1).loc[m].values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _variance_ratio(d: pd.DataFrame, mask: pd.Series, q: int = 4) -> float:
    """VR(q) = Var(q-step ret)/(q*Var(1-step ret)) over the in-window subset.
    q-step ret valid only if the q-bar window is fully contiguous."""
    r1 = d.loc[mask & d["gap_ok"], "r"].dropna()
    if len(r1) < 50:
        return float("nan")
    logc = np.log(d["close"])
    rq = logc - logc.shift(q)
    contig = (d.index.to_series().diff(q) == pd.Timedelta(minutes=15 * q))
    rq = rq[mask & contig].dropna()
    v1 = r1.var(ddof=1)
    if len(rq) < 50 or v1 < 1e-18:
        return float("nan")
    return float(rq.var(ddof=1) / (q * v1))


def _followthrough(d: pd.DataFrame, mask: pd.Series) -> float:
    """P(next bar continues current bar's direction) for in-window entry bars."""
    cur = np.sign(d["close"] - d["open"])
    nxt = np.sign(d["close"].shift(-1) - d["close"])
    m = mask & (cur != 0) & d["gap_ok"].shift(-1).fillna(False)
    if m.sum() < 30:
        return float("nan")
    return float((cur[m] == nxt[m]).mean())


def _baseline_rule(d: pd.DataFrame, mask: pd.Series, pip: float,
                   mode: str, spread_pips: float) -> dict:
    """Net-of-cost expectancy of a simple rule, exits via ATR triple-barrier."""
    cur = np.sign(d["close"] - d["open"])
    if mode == "momentum":
        side = cur.where(mask, np.nan)
    elif mode == "revert":
        side = (-cur).where(mask, np.nan)
    else:
        return {}
    side = side.replace(0.0, np.nan)
    lab = side_barrier_meta_label(d["high"], d["low"], d["close"], side,
                                  d["atr"], TP_MULT, SL_MULT, BARRIER_H, pip)
    if len(lab) < 30:
        return dict(n=len(lab), win=float("nan"), exp_R=float("nan"))
    net = lab["pips"] - spread_pips - COMM_PIPS
    return dict(n=len(lab), win=float((net > 0).mean()),
                exp_R=float((net / lab["sl_pips"]).mean()))


# ── per-window report row ────────────────────────────────────────────────────────

def _window_row(d: pd.DataFrame, mask: pd.Series, pip: float, has_spread: bool) -> dict:
    sub = d[mask]
    n = len(sub)
    spread_pips = float(sub["spread"].mean()) if has_spread else 1.0
    ac1 = _ac1(d, mask)
    vr4 = _variance_ratio(d, mask, 4)
    mom = _baseline_rule(d, mask, pip, "momentum", spread_pips)
    rev = _baseline_rule(d, mask, pip, "revert", spread_pips)
    # KIND classification
    kind = "—"
    if not np.isnan(ac1) and not np.isnan(vr4):
        if ac1 > 0.03 and vr4 > 1.1 and mom.get("exp_R", -9) > 0:
            kind = "TREND"
        elif ac1 < -0.03 and vr4 < 0.9 and rev.get("exp_R", -9) > 0:
            kind = "REVERT"
    return dict(
        n=n, range_pip=sub["rng_pip"].mean(), vol=sub["r"].std(),
        drift_pip=sub["ret_pip"].mean(), abs_pip=sub["ret_pip"].abs().mean(),
        ac1=ac1, vr4=vr4, ft=_followthrough(d, mask),
        spread=spread_pips, cost_ratio=spread_pips / max(sub["rng_pip"].mean(), 1e-9),
        mom_exp=mom.get("exp_R", float("nan")), mom_win=mom.get("win", float("nan")),
        mom_n=mom.get("n", 0),
        rev_exp=rev.get("exp_R", float("nan")), kind=kind,
    )


def _print_hour_table(d, pip, has_spread, title):
    print(f"\n  ── HOURLY (UTC→EAT) — {title} ──")
    print(f"  {'UTC':>3} {'EAT':>3} {'Bars':>6} {'Rng':>6} {'Drift':>7} {'AC1':>6} "
          f"{'VR4':>5} {'FT%':>5} {'Spr':>5} {'MomExp':>7} {'MomWin':>6} {'KIND':>6}  Sess")
    hour = pd.Series(d.index.hour, index=d.index)
    for h in range(24):
        row = _window_row(d, hour == h, pip, has_spread)
        if row["n"] == 0:
            continue
        eat = (h + EAT_OFFSET) % 24
        tags = []
        if (h >= 22) or (h < 7): tags.append("SYD")
        if h < 9: tags.append("TOK")
        if 8 <= h < 17: tags.append("LON")
        if 13 <= h < 22: tags.append("NY")
        f = lambda v, p=1: (f"{v:.{p}f}" if not np.isnan(v) else " n/a")
        print(f"  {h:02d}  {eat:02d}  {row['n']:>6,} {f(row['range_pip']):>6} "
              f"{f(row['drift_pip'],2):>7} {f(row['ac1'],2):>6} {f(row['vr4'],2):>5} "
              f"{f(row['ft']*100,0):>5} {f(row['spread']):>5} {f(row['mom_exp'],3):>7} "
              f"{f(row['mom_win']*100,0):>6} {row['kind']:>6}  {','.join(tags)}")


def _print_session_table(d, pip, has_spread, title):
    print(f"\n  ── SESSIONS — {title} ──")
    print(f"  {'Session':<16} {'UTC':>11} {'EAT':>11} {'Bars':>7} {'Rng':>6} {'AC1':>6} "
          f"{'VR4':>5} {'Spr':>5} {'CostR':>6} {'MomExp':>7} {'MomN':>6} {'KIND':>6}")
    hour = pd.Series(d.index.hour, index=d.index)
    for name, u_open, u_close, midnight, _eat in SESSION_DEFS:
        mask = session_mask(hour, u_open, u_close, midnight)
        row = _window_row(d, mask, pip, has_spread)
        if row["n"] == 0:
            continue
        f = lambda v, p=1: (f"{v:.{p}f}" if not np.isnan(v) else " n/a")
        print(f"  {name:<16} {f'{u_open:02d}-{u_close:02d}':>11} "
              f"{eat_range(u_open,u_close,midnight):>11} {row['n']:>7,} "
              f"{f(row['range_pip']):>6} {f(row['ac1'],2):>6} {f(row['vr4'],2):>5} "
              f"{f(row['spread']):>5} {f(row['cost_ratio'],2):>6} "
              f"{f(row['mom_exp'],3):>7} {row['mom_n']:>6,} {row['kind']:>6}")


def _print_primary(d, sym, pip, has_spread, label):
    print(f"\n  ── PRE-REGISTERED WINDOWS — {sym} [{label}] ──")
    hour = pd.Series(d.index.hour, index=d.index)
    for name, u_open, u_close in PRIMARY_WINDOWS.get(sym, []):
        row = _window_row(d, session_mask(hour, u_open, u_close, u_close <= u_open),
                          pip, has_spread)
        f = lambda v, p=3: (f"{v:.{p}f}" if not np.isnan(v) else "n/a")
        print(f"  {name} ({u_open:02d}-{u_close:02d} UTC): bars={row['n']:,}  "
              f"AC1={f(row['ac1'],2)} VR4={f(row['vr4'],2)} kind={row['kind']}  "
              f"spread={f(row['spread'],2)}p  momentum net exp_R={f(row['mom_exp'])} "
              f"(win {f(row['mom_win']*100,0)}%, n={row['mom_n']:,})  "
              f"revert net exp_R={f(row['rev_exp'])}")


def run_symbol(sym: str, use_deep: bool, discover_confirm: bool):
    cfg = SYMBOL_CFG[sym]
    path = cfg["deep"] if use_deep else cfg["short"]
    if not Path(path).exists():
        print(f"  {sym}: {path} not found — skipping"); return
    d = _prep(_load_raw(path), cfg["pip"])
    has_spread = ("spread" in d.columns) and (d["spread"].abs().sum() > 0)
    print(f"\n{'='*96}\n  SESSION PROFILE — {sym}  ({'DEEP/UTC' if use_deep else 'SHORT/broker-time'})")
    print(f"  {len(d):,} bars  {d.index[0]} → {d.index[-1]}  "
          f"spread={'real' if has_spread else 'MISSING (cost=flat 1.0p)'}")
    print(f"{'='*96}")
    if not use_deep:
        print("  WARNING: short file is broker-time + spread=0 — hours/costs NOT reliable.")

    if discover_confirm and use_deep:
        split = pd.Timestamp("2022-01-01")
        for label, seg in [("DISCOVER 2015-2021", d[d.index < split]),
                           ("CONFIRM 2022-2026", d[d.index >= split])]:
            print(f"\n  ===== {label}  ({len(seg):,} bars) =====")
            _print_primary(seg, sym, cfg["pip"], has_spread, label)
            _print_session_table(seg, cfg["pip"], has_spread, f"{sym} {label}")
    else:
        _print_primary(d, sym, cfg["pip"], has_spread, "FULL")
        _print_session_table(d, cfg["pip"], has_spread, sym)
        _print_hour_table(d, cfg["pip"], has_spread, sym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG))
    ap.add_argument("--short", action="store_true", help="use short broker-time file (not recommended)")
    ap.add_argument("--discover-confirm", action="store_true")
    args = ap.parse_args()
    syms = [args.symbol] if args.symbol else list(SYMBOL_CFG)
    for s in syms:
        run_symbol(s, use_deep=not args.short, discover_confirm=args.discover_confirm)
    print("\nDone.")


if __name__ == "__main__":
    main()
