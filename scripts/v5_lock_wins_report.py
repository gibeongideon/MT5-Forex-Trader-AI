"""WHAT IF WE LOCKED IN WINS? — side-by-side report.

Answers the intuitive question "surely banking profits is safer?" by running the
SAME book two ways over the same 9.5 years and showing where the money actually
goes. Uses the live FundingPips 10K book (XAU + ETH + DJI) under Flex rules.

  HOLD-THROUGH : what the bot does today (exit only when the trend dies)
  LOCK WINS    : close everything once up >= X% since entry, re-enter after a dip

Reported per variant: final equity on a $10,000 account, the biggest single winning
run captured vs given away, worst drawdown, and the resulting challenge pass rate.
The point is to make the trade-off concrete rather than theoretical.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa
from v5_profit_take_overlay import overlay  # noqa

MODEL = vbc.MODELS["flex"]                       # FundingPips Flex
DIAL = MODEL["vol"] / vbc.TARGET_VOL
CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["ETH"], "eq_us": ["DJI"]}
START = 10000.0


def live_book():
    vbc.CLASSES = CLASSES
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (MODEL["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    return (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]


def best_run(r, window=60):
    """Largest gain captured in any rolling `window`-day stretch (the 'big winner')."""
    eq = (1 + r).cumprod()
    roll = eq / eq.shift(window) - 1
    return float(roll.max() * 100)


def describe(r, label):
    eq = (1 + r).cumprod()
    final = START * float(eq.iloc[-1])
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min() * 100)
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1) * 100
    r10 = (r * (0.10 / (r.std() * np.sqrt(252)))).values if r.std() > 0 else r.values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                    dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
    # how long to first reach +10% (phase-1 target) on this path
    hit = np.argmax(eq.values >= 1.10) if (eq.values >= 1.10).any() else -1
    months = (hit / 21.0) if hit > 0 else float("nan")
    return dict(label=label, final=final, sr=sr, cagr=cagr, dd=dd,
                best=best_run(r), passpct=fp["passpct"], med=fp["med_mo"],
                first10=months, inmkt=float((r != 0).mean()) * 100)


def main():
    base = live_book()
    rows = [describe(base, "HOLD-THROUGH (what the bot does)")]
    for tp, dip, wait in ((0.015, 0.005, 10), (0.015, 0.010, 10),
                          (0.030, 0.010, 15), (0.050, 0.020, 20)):
        rows.append(describe(overlay(base, tp, dip, wait),
                             f"LOCK WINS at +{tp*100:.1f}% (dip {dip*100:.1f}%)"))

    print("=" * 100)
    print("WHAT IF WE LOCKED IN WINS?   FundingPips 10K book (XAU+ETH+DJI), "
          f"{base.index[0]:%b %Y}-{base.index[-1]:%b %Y}")
    print("=" * 100)
    print(f"{'strategy':36s} {'$10k becomes':>13} {'CAGR%':>7} {'Sharpe':>7} "
          f"{'worstDD':>8} {'best 3mo run':>13} {'pass%':>7}")
    print("-" * 100)
    for s in rows:
        print(f"{s['label']:36s} {s['final']:13,.0f} {s['cagr']:7.2f} {s['sr']:7.2f} "
              f"{s['dd']:7.1f}% {s['best']:12.1f}% {s['passpct']:7.1f}")
    b = rows[0]
    print("\nCOST OF LOCKING WINS (vs hold-through):")
    for s in rows[1:]:
        print(f"  {s['label']:34s} ${s['final']-b['final']:+10,.0f}  "
              f"({(s['final']/b['final']-1)*100:+6.1f}%)   pass {s['passpct']-b['passpct']:+5.1f}pp   "
              f"biggest run {s['best']-b['best']:+.1f}pp")
    print(f"\nTime in market: hold-through {b['inmkt']:.0f}%  vs  "
          + ", ".join(f"{s['label'].split(' at ')[1] if ' at ' in s['label'] else s['label']} {s['inmkt']:.0f}%"
                      for s in rows[1:]))
    print("\nWHY: trend books earn from a few LARGE sustained runs. The 'best 3mo run'")
    print("column is the single biggest winning stretch each version captured — locking")
    print("wins truncates exactly that, and the small gains banked never make it back.")


if __name__ == "__main__":
    main()
