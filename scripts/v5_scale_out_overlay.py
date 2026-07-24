"""PARTIAL SCALE-OUT overlay — bank SOME profit, keep the rest running.

Distinct from v5_profit_take_overlay.py (which closed EVERYTHING and waited for a
dip — disproven, V5_FINDINGS §3e, because it misses the big trend runs). Here the
position is only REDUCED, so we stay exposed to the trend that is paying us.

  when gain-since-last-trim >= trim_at:  exposure *= (1 - trim_frac)   [bank part]
  exposure is restored to 1.0 when the book pulls back `reset_dd` from its peak
  (a new trend leg starts) or after `reset_days`.

Also covers the "bank and immediately re-enter the SAME size" idea: that is
economically just paying one round-trip spread with no change in exposure, so it
is modelled as a pure cost (roll_bp) rather than a strategy change.

Cost: each trim/restore crosses the spread on the traded fraction.
Measured FundingPips-side round trips: XAUmicro 2.2bp, ETH 5.4bp, DJI 1.4bp
-> ~3bp blended, used below.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

MODEL = vbc.MODELS["flex"]
DIAL = MODEL["vol"] / vbc.TARGET_VOL
CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["ETH"], "eq_us": ["DJI"]}
START = 10000.0
COST_BP = 3.0


def live_book():
    vbc.CLASSES = CLASSES
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (MODEL["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    return (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]


def scale_out(r, trim_at, trim_frac, reset_dd, reset_days, min_exposure=0.25):
    """Reduce exposure after gains; restore it when a new leg starts."""
    vals = r.values
    out = np.zeros(len(vals))
    expo = 1.0
    gain = 0.0            # gain since last trim
    peak = 1.0            # running peak of the notional book
    wouldbe = 1.0         # book equity at FULL exposure (for peak/reset logic)
    since_trim = 0
    cost = COST_BP / 1e4
    for i, x in enumerate(vals):
        out[i] = expo * x
        wouldbe *= (1 + x)
        peak = max(peak, wouldbe)
        gain = (1 + gain) * (1 + x) - 1
        since_trim += 1
        # 1) trim after a run
        if gain >= trim_at and expo > min_exposure:
            new = max(min_exposure, expo * (1 - trim_frac))
            out[i] -= abs(expo - new) * cost
            expo, gain, since_trim = new, 0.0, 0
        # 2) restore exposure when a new leg starts (pullback) or after a timeout
        elif expo < 1.0 and (wouldbe <= peak * (1 - reset_dd) or since_trim >= reset_days):
            out[i] -= abs(1.0 - expo) * cost
            expo, gain, since_trim, peak = 1.0, 0.0, 0, wouldbe
    return pd.Series(out, index=r.index)


def describe(r, label):
    eq = (1 + r).cumprod()
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min() * 100)
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1) * 100
    roll = eq / eq.shift(60) - 1
    r10 = (r * (0.10 / (r.std() * np.sqrt(252)))).values if r.std() > 0 else r.values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                    dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
    return dict(label=label, final=START * float(eq.iloc[-1]), sr=sr, cagr=cagr,
                dd=dd, best=float(roll.max() * 100), passpct=fp["passpct"],
                med=fp["med_mo"])


if __name__ == "__main__":
    base = live_book()
    rows = [describe(base, "HOLD-THROUGH (current bot)")]
    grid = [
        ("trim 25% @ +2%, reset -1%", 0.02, 0.25, 0.01, 40),
        ("trim 50% @ +2%, reset -1%", 0.02, 0.50, 0.01, 40),
        ("trim 25% @ +3%, reset -1.5%", 0.03, 0.25, 0.015, 40),
        ("trim 50% @ +3%, reset -1.5%", 0.03, 0.50, 0.015, 40),
        ("trim 33% @ +5%, reset -2%", 0.05, 0.33, 0.02, 60),
        ("trim 50% @ +5%, reset -2%", 0.05, 0.50, 0.02, 60),
        ("trim 25% @ +1.5%, reset -1%", 0.015, 0.25, 0.01, 30),
    ]
    for lab, ta, tf, rd, rdays in grid:
        rows.append(describe(scale_out(base, ta, tf, rd, rdays), lab))

    print("=" * 104)
    print("PARTIAL SCALE-OUT: bank some, keep the rest running   "
          f"(FundingPips 10K book, {base.index[0]:%b %Y}-{base.index[-1]:%b %Y})")
    print("=" * 104)
    print(f"{'variant':32s} {'$10k becomes':>13} {'CAGR%':>7} {'Sharpe':>7} "
          f"{'worstDD':>8} {'best 3mo':>9} {'pass%':>7} {'med_mo':>7}")
    print("-" * 104)
    for s in rows:
        print(f"{s['label']:32s} {s['final']:13,.0f} {s['cagr']:7.2f} {s['sr']:7.2f} "
              f"{s['dd']:7.1f}% {s['best']:8.1f}% {s['passpct']:7.1f} {s['med']:7.1f}")
    b = rows[0]
    print("\nvs HOLD-THROUGH:")
    for s in rows[1:]:
        flag = "BETTER" if (s['passpct'] > b['passpct'] and s['final'] > b['final']) else ""
        print(f"  {s['label']:32s} ${s['final']-b['final']:+10,.0f} "
              f"({(s['final']/b['final']-1)*100:+6.1f}%)  Sharpe {s['sr']-b['sr']:+.3f}  "
              f"DD {s['dd']-b['dd']:+.1f}pp  pass {s['passpct']-b['passpct']:+5.1f}pp  {flag}")
    print("\nNOTE: 'bank and re-enter the SAME size' is not shown as a strategy because it")
    print("does not change exposure at all — it is just one extra round-trip spread")
    print("(~3bp). It converts floating P/L to balance but leaves risk identical.")
