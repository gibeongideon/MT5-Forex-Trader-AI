"""XAUUSD next-bar fade (mean-reversion) backtest — flat vs anti-martingale vs martingale.

Signal (from win-rate study): fade extreme closes.
  close near bar LOW  (close_pos < lo_thr) -> LONG next bar
  close near bar HIGH (close_pos > hi_thr) -> SHORT next bar
Enter at next bar OPEN, exit at next bar CLOSE (one-bar hold), net of the broker's
time-varying spread (widens on volatile bars — the honest cost of trading them).

Sizing modes:
  flat   : constant stake (Sharpe-invariant baseline)
  anti   : ramp stake UP after wins (x step, capped), reset on loss  -> rides streaks
  marti  : DOUBLE stake after losses (capped), reset on win          -> demonstrates ruin

No-lookahead: signal from bar t, position taken on bar t+1 open->close.

Pre-registered:
    python scripts/v5_xau_fade_backtest.py --tf M15 --mode flat
Stress:
    --spread-mult 3      # 3x the broker spread
    --good-hours 8,20,22 # restrict to best session hours
    --eval-start 2021-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown, sortino_ratio

POINT = 0.01  # gold: spread column is in points; $ = points * 0.01


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def build_signal(d: pd.DataFrame, lo_thr: float, hi_thr: float,
                 good_hours: set | None) -> pd.Series:
    rng = (d.high - d.low).replace(0, np.nan)
    close_pos = (d.close - d.low) / rng
    dirn = pd.Series(0, index=d.index, dtype=float)
    dirn[close_pos < lo_thr] = 1.0    # long fade of a down-close
    dirn[close_pos > hi_thr] = -1.0   # short fade of an up-close
    if good_hours:
        dirn[~d.index.hour.isin(good_hours)] = 0.0
    return dirn


def run(d: pd.DataFrame, dirn: pd.Series, mode: str, cfg: dict) -> pd.DataFrame:
    o1, c1 = d.open.shift(-1), d.close.shift(-1)          # next bar OHLC
    if cfg.get("fixed_spread_usd") is not None:
        spread1 = pd.Series(cfg["fixed_spread_usd"], index=d.index)  # constant $ spread
    else:
        spread1 = d.spread.shift(-1) * POINT * cfg["spread_mult"]
    signed = dirn * (c1 - o1)                              # $ move in our direction
    gross_frac = signed / o1                               # per unit notional
    cost_frac = spread1 / o1                               # full spread paid round trip

    trade = dirn != 0
    idx = d.index[trade & c1.notna()]
    if cfg.get("eval_start"):
        idx = idx[idx >= pd.Timestamp(cfg["eval_start"])]

    # sequential sizing
    stakes, nets, wins = [], [], []
    mult = 1.0
    g = gross_frac.to_dict(); co = cost_frac.to_dict()
    step, cap = cfg["step"], cfg["cap"]
    for t in idx:
        gv = g[t] - co[t]           # net per-unit return this trade
        stake = mult
        net = stake * gv
        won = gv > 0
        stakes.append(stake); nets.append(net); wins.append(won)
        if mode == "flat":
            mult = 1.0
        elif mode == "anti":
            mult = min(mult * step, cap) if won else 1.0
        elif mode == "marti":
            mult = 1.0 if won else min(mult * 2.0, cap)
    return pd.DataFrame({"stake": stakes, "net": nets, "won": wins}, index=idx)


def summarize(tr: pd.DataFrame, d: pd.DataFrame, label: str) -> dict:
    # per-bar equity over the full timeline (flat between trades) for daily Sharpe
    net_bar = pd.Series(0.0, index=d.index)
    net_bar.loc[tr.index] = tr["net"].values
    net_bar = net_bar.loc[tr.index[0]:tr.index[-1]]
    equity = (1.0 + net_bar).cumprod()
    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    shp = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)
    net = tr["net"]
    wr = float(tr["won"].mean())
    gains = net[net > 0].sum(); losses = -net[net < 0].sum()
    pf = float(gains / losses) if losses > 0 else np.inf
    years = (tr.index[-1] - tr.index[0]).days / 365.25
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0
    yearly = (1.0 + net_bar).groupby(net_bar.index.year).prod() - 1.0
    return dict(label=label, sharpe=shp, ci=(ci_lo, ci_hi), sortino=sortino_ratio(equity),
                maxdd=max_drawdown(equity), pf=pf, wr=wr, trades=len(tr),
                tpy=len(tr) / years, total=total, cagr=cagr, years=years,
                max_stake=float(tr["stake"].max()), yearly=yearly, equity=equity)


def show(r: dict) -> None:
    lo, hi = r["ci"]
    print(f"\n=== {r['label']} ===")
    print(f"  trades {r['trades']} ({r['tpy']:.0f}/yr, {r['years']:.1f} yrs)   win {r['wr']*100:.1f}%   PF {r['pf']:.3f}")
    print(f"  Sharpe {r['sharpe']:.2f}  95% CI [{lo:.2f}, {hi:.2f}]   Sortino {r['sortino']:.2f}")
    print(f"  CAGR {r['cagr']*100:.2f}%   total {r['total']*100:.1f}%   maxDD {r['maxdd']:.1f}%   peak stake {r['max_stake']:.0f}x")
    yr = r["yearly"]
    print("  yearly: " + "  ".join(f"{y}:{v*100:+.0f}%" for y, v in yr.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="M15")
    ap.add_argument("--lo-thr", type=float, default=0.2)
    ap.add_argument("--hi-thr", type=float, default=0.8)
    ap.add_argument("--good-hours", default=None, help="comma list e.g. 8,20,22")
    ap.add_argument("--mode", default="all", choices=["flat", "anti", "marti", "all"])
    ap.add_argument("--step", type=float, default=1.5, help="anti-martingale up-step")
    ap.add_argument("--cap", type=float, default=8.0, help="max stake multiple")
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--fixed-spread-usd", type=float, default=None,
                    help="override time-varying spread with a constant $ spread (e.g. 0.14)")
    ap.add_argument("--eval-start", default=None)
    args = ap.parse_args()

    good = set(int(x) for x in args.good_hours.split(",")) if args.good_hours else None
    d = load(args.tf)
    dirn = build_signal(d, args.lo_thr, args.hi_thr, good)
    cfg = dict(step=args.step, cap=args.cap, spread_mult=args.spread_mult,
               fixed_spread_usd=args.fixed_spread_usd, eval_start=args.eval_start)
    print(f"[XAU fade] tf={args.tf} lo<{args.lo_thr} hi>{args.hi_thr} hours={good or 'ALL'} "
          f"spreadx{args.spread_mult} anti(step{args.step},cap{args.cap}x)"
          + (f" eval>={args.eval_start}" if args.eval_start else ""))
    modes = ["flat", "anti", "marti"] if args.mode == "all" else [args.mode]
    for m in modes:
        tr = run(d, dirn, m, cfg)
        show(summarize(tr, d, f"{m.upper()}"))


if __name__ == "__main__":
    main()
