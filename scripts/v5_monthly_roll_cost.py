"""FTMO SWING: what does a forced 'close every position at least once a month' cost?

Swing accounts allow weekend AND overnight holding (so the drop-crypto restructure
in FUNDED-STAGE-PLAN.md is NOT needed) but require positions to be closed at least
monthly. Our trend book holds winners for months, so it must ROLL: close and
immediately re-open the same exposure.

Economically a same-session roll is NOT a signal change — it is purely one extra
round-trip cost per position per month. This measures that drag.

Two readings of the rule, both covered:
  (a) INACTIVITY rule (just need >=1 trade a month) -> the book already trades
      3-4x/month, so cost is ZERO. Nothing to do.
  (b) MAX HOLDING PERIOD (every position closed monthly) -> forced roll, costed here.

Real measured spreads (FTMO, 2026-07-20):
  XAUUSD $0.45 @ ~4019 = 1.1bp/crossing | BTCUSD $20 @ ~64000 = 3.1bp
  US100.cash 1.95 @ ~28700 = 0.7bp
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

MODEL = vbc.MODELS["ftmo"]
DIAL = MODEL["vol"] / vbc.TARGET_VOL
CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]}

# round-trip cost per sleeve, in basis points of notional (measured spreads x2)
ROUNDTRIP_BP = {"XAUCHAMP": 2.2, "BTC": 6.2, "NDX": 1.4}
ROLL_DAYS = 21          # ~1 trading month


def book_with_roll(roll_days, extra_mult=1.0):
    """Rebuild the book charging a forced round-trip on every sleeve every
    `roll_days`. extra_mult lets us stress the assumed spread."""
    vbc.CLASSES = CLASSES
    live, nets = {}, {}
    for members in CLASSES.values():
        for sym in members:
            lp, nt = vbc._load_asset(sym)
            nets[sym] = nt.loc["2016-01-01":]
            live[sym] = lp
    al = pd.DataFrame(nets).loc["2016-01-01":]

    if roll_days:
        # charge each sleeve a round trip every roll_days (only while exposed)
        for sym in al.columns:
            bp = ROUNDTRIP_BP.get(sym, 3.0) * extra_mult / 1e4
            idx = np.arange(len(al))
            hit = (idx % roll_days) == 0
            # scale by how exposed the sleeve typically is (proxy: |ret| > 0)
            al[sym] = al[sym] - np.where(hit, bp, 0.0)

    a = {s: vbc.TARGET_VOL / (al[s].std() * np.sqrt(252)) for s in al.columns}
    cls_stream, b_cls = {}, {}
    for cls, members in CLASSES.items():
        comp = sum(a[m] * al[m].fillna(0.0) for m in members) / len(members)
        b_cls[cls] = vbc.TARGET_VOL / (comp.std() * np.sqrt(252))
        cls_stream[cls] = b_cls[cls] * comp
    cl = pd.DataFrame(cls_stream).dropna()
    port = sum(cl[c] for c in cl.columns) / len(cl.columns)
    g = vbc.TARGET_VOL / (port.std() * np.sqrt(252))
    book = DIAL * g * port
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (MODEL["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    return (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]


def stats(r, label):
    sr = float(r.mean() / r.std() * np.sqrt(252))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1) * 100
    r10 = (r * (0.10 / (r.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                    dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
    return dict(label=label, sr=sr, cagr=cagr, dd=dd,
                passpct=fp["passpct"], med=fp["med_mo"])


if __name__ == "__main__":
    print("FTMO book (XAU+BTC+NDX) — cost of a forced monthly close+reopen\n")
    rows = [stats(book_with_roll(0), "no roll (weekend+overnight OK)")]
    rows.append(stats(book_with_roll(ROLL_DAYS), "monthly roll (measured spreads)"))
    rows.append(stats(book_with_roll(ROLL_DAYS, 3.0), "monthly roll, spreads x3 (stress)"))
    rows.append(stats(book_with_roll(10), "roll every 10d (stress)"))
    print(f"{'variant':36s} {'Sharpe':>7} {'CAGR%':>7} {'maxDD%':>7} {'pass%':>7} {'med_mo':>7}")
    for s in rows:
        print(f"{s['label']:36s} {s['sr']:+7.2f} {s['cagr']:7.2f} {s['dd']:7.1f} "
              f"{s['passpct']:7.1f} {s['med']:7.1f}")
    b = rows[0]
    print("\nDrag vs no-roll:")
    for s in rows[1:]:
        print(f"  {s['label']:34s} Sharpe {s['sr']-b['sr']:+.3f}, "
              f"CAGR {s['cagr']-b['cagr']:+.2f}pp, pass {s['passpct']-b['passpct']:+.1f}pp")
