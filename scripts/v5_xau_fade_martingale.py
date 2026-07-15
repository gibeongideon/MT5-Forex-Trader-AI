"""XAUUSD next-bar fade — capped LOSS-RECOVERY martingale experiment.

Extends scripts/v5_xau_fade_backtest.py with a designed, ruin-bounded martingale
whose explicit goal is the user's spec:

    "recover, on the next win, ALL losses accumulated in up to 4 subsequent
     trades; reset after that win. If still underwater after the 4th trade,
     give up the ladder and start fresh."

Same no-lookahead signal: fade extreme M15 closes (close near LOW -> long next
bar, close near HIGH -> short next bar), restricted to good session hours, enter
next-bar OPEN, exit next-bar CLOSE. Net of the honest broker spread (~$0.34).

Sizing engines
--------------
flat      constant base notional (edge-only baseline).
double4   CLASSIC martingale: after a loss, DOUBLE the stake; reset on any win;
          hard cap of K=4 escalation steps (=> max 2^4 = 16x), then force-reset.
          "double until a win recovers it, but never risk more than 4 doublings."
recover4  DEFICIT-TARGETED martingale (the 'tough', engineered one): after a
          loss, size the next stake so that a *typical* win return `w` clears the
          entire running deficit D **plus** one base unit of profit:
                stake = min( (D + base) / w , cap*base )
          Ladders up to K=4 trades; the moment cumulative ladder P&L >= 0 it
          resets flat; if the 4th trade still leaves it underwater it force-resets
          (realises the loss) so a single cold streak cannot compound forever.

Ruin model: equity in $, starts at --equity (10000). Base notional --base
($1000, matching the live paper bot). A trade's $ P&L = stake$ * net_return.
Equity that touches the --floor is BUST (trading stops; remaining bars skipped).

    python scripts/v5_xau_fade_martingale.py --good-hours 8,20,22           # net $0.34
    python scripts/v5_xau_fade_martingale.py --good-hours 8,20,22 --spread 0 # gross edge
    python scripts/v5_xau_fade_martingale.py --mc 500                        # ruin distribution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown

POINT = 0.01


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def build_signal(d, lo_thr, hi_thr, good):
    rng = (d.high - d.low).replace(0, np.nan)
    cp = (d.close - d.low) / rng
    dirn = pd.Series(0.0, index=d.index)
    dirn[cp < lo_thr] = 1.0
    dirn[cp > hi_thr] = -1.0
    if good:
        dirn[~d.index.hour.isin(good)] = 0.0
    return dirn


def trade_returns(d, dirn, spread_usd, spread_mult, eval_start):
    """Per-unit-notional NET return of each fired trade, chronological."""
    o1, c1 = d.open.shift(-1), d.close.shift(-1)
    if spread_usd is not None:
        spread1 = pd.Series(spread_usd, index=d.index)
    else:
        spread1 = d.spread.shift(-1) * POINT * spread_mult
    net_frac = (dirn * (c1 - o1) - spread1) / o1        # spread paid every trade
    trade = (dirn != 0) & c1.notna()
    r = net_frac[trade]
    if eval_start:
        r = r[r.index >= pd.Timestamp(eval_start)]
    return r


def simulate(r: np.ndarray, engine: str, base: float, equity0: float,
             floor: float, K: int, cap: float, w: float):
    """Sequential $ simulation. Returns (equity_curve, stakes, busted_at)."""
    eq = equity0
    curve = np.empty(len(r)); stakes = np.empty(len(r))
    k = 0                 # escalation step within current ladder
    ladder = 0.0          # cumulative $ P&L since last reset
    mult = 1.0            # double4 multiplier
    busted = -1
    for i, ri in enumerate(r):
        if eq <= floor:
            curve[i:] = eq; stakes[i:] = 0.0; busted = i; break
        # ---- choose stake ----
        if engine == "flat":
            stake = base
        elif engine == "double4":
            stake = base * mult
        elif engine == "recover4":
            deficit = max(0.0, -ladder)
            stake = base if k == 0 else min((deficit + base) / w, cap * base)
        else:
            raise ValueError(engine)
        stake = min(stake, max(0.0, eq - floor) / max(w, 1e-9))   # cant risk past bust
        pnl = stake * ri
        eq += pnl
        ladder += pnl
        curve[i] = eq; stakes[i] = stake
        # ---- ladder / reset logic ----
        won = ri > 0
        if engine == "double4":
            if won:
                mult, k, ladder = 1.0, 0, 0.0
            else:
                k += 1
                if k >= K:
                    mult, k, ladder = 1.0, 0, 0.0     # give up after K doublings
                else:
                    mult *= 2.0
        elif engine == "recover4":
            if ladder >= 0.0:                          # recovered (or a plain win)
                k, ladder = 0, 0.0
            elif k + 1 >= K:                           # exhausted the 4-trade ladder
                k, ladder = 0, 0.0                     # realise loss, start fresh
            else:
                k += 1
        else:  # flat
            pass
    else:
        pass
    return curve, stakes, busted


def stats(curve, stakes, r, base, equity0, label, busted):
    eq = pd.Series(curve)
    ret = eq.pct_change().fillna(eq.iloc[0] / equity0 - 1.0)
    # daily-ish aggregation not available (no timestamps here) -> per-trade Sharpe*sqrt(N/yr proxy)
    pt = ret.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe_trade = float(pt.mean() / pt.std()) if pt.std() > 0 else 0.0
    ci_lo, ci_hi = block_bootstrap_sharpe(pt.values) if len(pt) > 30 else (0, 0)
    wins = (r > 0).mean()
    end = curve[-1]
    peak_stake = stakes.max() / base
    return dict(label=label, end=end, ret=end / equity0 - 1.0, wr=float(wins),
                maxdd=max_drawdown(eq), sharpe_t=sharpe_trade, ci=(ci_lo, ci_hi),
                peak_stake=peak_stake, busted=busted, n=len(r))


def show(s):
    b = f"BUST@{s['busted']}" if s["busted"] >= 0 else "survived"
    lo, hi = s["ci"]
    print(f"\n=== {s['label']} ===")
    print(f"  trades {s['n']}   win {s['wr']*100:.1f}%   {b}")
    print(f"  end equity ${s['end']:,.0f}   total {s['ret']*100:+.1f}%   maxDD {s['maxdd']:.1f}%")
    print(f"  per-trade Sharpe {s['sharpe_t']:.3f}  CI[{lo:.3f},{hi:.3f}]   peak stake {s['peak_stake']:.0f}x base")


def monte_carlo(r, engines, args, w, n=500, block=200):
    """Bootstrap the trade sequence in blocks -> outcome distribution per engine."""
    rng = np.random.default_rng(7)
    N = len(r)
    out = {e: [] for e in engines}
    busts = {e: 0 for e in engines}
    for _ in range(n):
        # circular block bootstrap preserves short streaks (what kills martingales)
        idx = []
        while len(idx) < N:
            s0 = rng.integers(0, N)
            idx.extend(range(s0, s0 + block))
        seq = r[np.array(idx[:N]) % N]
        for e in engines:
            c, _, bust = simulate(seq, e, args.base, args.equity, args.floor,
                                  args.K, args.cap, w)
            out[e].append(c[-1])
            busts[e] += bust >= 0
    print(f"\n--- Monte-Carlo: {n} block-bootstrap paths ({block}-trade blocks) ---")
    print(f"{'engine':10} {'median':>12} {'mean':>12} {'p05':>10} {'p95':>12} {'bust%':>7}")
    for e in engines:
        a = np.array(out[e])
        print(f"{e:10} {np.median(a):12,.0f} {a.mean():12,.0f} "
              f"{np.percentile(a,5):10,.0f} {np.percentile(a,95):12,.0f} "
              f"{100*busts[e]/n:6.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="M15")
    ap.add_argument("--lo-thr", type=float, default=0.2)
    ap.add_argument("--hi-thr", type=float, default=0.8)
    ap.add_argument("--good-hours", default="8,20,22")
    ap.add_argument("--spread", type=float, default=0.34,
                    help="honest fixed $ spread; use 0 for the gross edge")
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--eval-start", default=None)
    ap.add_argument("--base", type=float, default=1000.0, help="base notional $")
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--floor", type=float, default=0.0, help="bust threshold $")
    ap.add_argument("--K", type=int, default=4, help="max ladder length (4 = spec)")
    ap.add_argument("--cap", type=float, default=50.0, help="recover4 max stake multiple")
    ap.add_argument("--mc", type=int, default=0, help="Monte-Carlo paths (0=off)")
    args = ap.parse_args()

    good = None if args.good_hours.lower() == "all" else {int(x) for x in args.good_hours.split(",")}
    d = load(args.tf)
    dirn = build_signal(d, args.lo_thr, args.hi_thr, good)
    r = trade_returns(d, dirn, args.spread if args.spread is not None else None,
                      args.spread_mult, args.eval_start).values
    w = float(np.median(r[r > 0])) if (r > 0).any() else 0.001   # typical win return

    print(f"[XAU fade martingale] tf={args.tf} lo<{args.lo_thr} hi>{args.hi_thr} "
          f"hours={good or 'ALL'} spread=${args.spread} K={args.K} cap={args.cap}x")
    print(f"  fired trades: {len(r)}   base win-rate {100*(r>0).mean():.1f}%   "
          f"typical win return w={w:.5f}  ({args.base*w:.2f}$ per base trade)")

    engines = ["flat", "double4", "recover4"]
    for e in engines:
        c, st, bust = simulate(r, e, args.base, args.equity, args.floor, args.K, args.cap, w)
        show(stats(c, st, r, args.base, args.equity, e, bust))

    if args.mc:
        monte_carlo(r, engines, args, w, n=args.mc)


if __name__ == "__main__":
    main()
