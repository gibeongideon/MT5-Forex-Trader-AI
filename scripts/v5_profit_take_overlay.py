"""PROFIT-TAKE + RE-ENTRY overlay on the challenge book — does banking gains help?

Idea under test: when the book is up >= X% since the last entry, CLOSE everything
(bank it), then RE-ENTER once price has pulled back ("low enough to rise again"),
or after a timeout so we don't miss a runaway trend.

Why it might help a CHALLENGE even if it hurts Sharpe: floating profit counts
against the firm's daily-loss line, so unrealized gains are fragile. Converting
them to balance protects the account and raises the daily anchor.
Why it might hurt: trend-following earns from a few huge winners; capping them
truncates the right tail that pays for all the small losses.

Modelled at DAILY resolution on the book's own return stream:
  state IN  : accumulate gain since entry; if >= take_pct -> go OUT at that close
  state OUT : track the "would-be" book equity; re-enter when it has dipped
              dip_pct from the exit level, or after max_wait days.

CAVEAT: a real +1.5% day would trigger INTRADAY; daily bars can only exit at the
close of the day that crossed it, so realised take-profit levels are approximate.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

MODEL = vbc.MODELS["ftmo"]                 # FTMO 2-Step: +10/+5, daily 5%, max 10%
DIAL = MODEL["vol"] / vbc.TARGET_VOL
CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]}


def base_book():
    vbc.CLASSES = CLASSES
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (MODEL["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    return (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]


def overlay(r: pd.Series, take_pct, dip_pct, max_wait, cost_bp=2.0):
    """Apply profit-take + dip re-entry. Returns the realised daily stream."""
    out = np.zeros(len(r))
    vals = r.values
    state = "IN"
    gain = 0.0          # compounded gain since entry, while IN
    wouldbe = 1.0       # 'would-be' equity while OUT (tracks the strategy)
    peak_out = 1.0      # level at exit (and running peak while OUT)
    wait = 0
    cost = cost_bp / 1e4
    for i, x in enumerate(vals):
        if state == "IN":
            out[i] = x
            gain = (1 + gain) * (1 + x) - 1
            if gain >= take_pct:                 # bank it
                out[i] -= cost                   # exit cost
                state, gain, wouldbe, peak_out, wait = "OUT", 0.0, 1.0, 1.0, 0
        else:                                    # OUT: earn nothing, watch
            out[i] = 0.0
            wouldbe *= (1 + x)
            peak_out = max(peak_out, wouldbe)
            wait += 1
            dipped = wouldbe <= peak_out * (1 - dip_pct)
            if dipped or wait >= max_wait:       # re-enter
                out[i] -= cost
                state, gain = "IN", 0.0
    return pd.Series(out, index=r.index)


def stats(r, label):
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1) * 100
    r10 = (r * (0.10 / (r.std() * np.sqrt(252)))).values if r.std() > 0 else r.values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                    dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
    return dict(label=label, sr=sr, cagr=cagr, dd=dd,
                passpct=fp["passpct"], fday=fp["fail_day"], fdd=fp["fail_dd"],
                med=fp["med_mo"], intrade=float((r != 0).mean()) * 100)


if __name__ == "__main__":
    base = base_book()
    print(f"FTMO book (XAU+BTC+NDX), eval 2017+, {len(base)} days\n")
    rows = [stats(base, "BASELINE hold-through")]
    grid = [(0.015, 0.005, 10), (0.015, 0.010, 10), (0.015, 0.005, 5),
            (0.015, 0.000, 1),                      # take profit, re-enter next day
            (0.010, 0.005, 10), (0.020, 0.005, 10), (0.030, 0.005, 10),
            (0.015, 0.020, 20)]
    for tp, dip, wait in grid:
        lab = f"take {tp*100:.1f}% / dip {dip*100:.1f}% / wait {wait}d"
        rows.append(stats(overlay(base, tp, dip, wait), lab))
    print(f"{'variant':34s} {'Sharpe':>7} {'CAGR%':>7} {'maxDD%':>7} "
          f"{'pass%':>7} {'fDay':>6} {'fDD':>6} {'med_mo':>7} {'%inMkt':>7}")
    for s in rows:
        print(f"{s['label']:34s} {s['sr']:+7.2f} {s['cagr']:7.1f} {s['dd']:7.1f} "
              f"{s['passpct']:7.1f} {s['fday']:6.1f} {s['fdd']:6.1f} {s['med']:7.1f} "
              f"{s['intrade']:7.1f}")
