"""SPRINT ANALYSIS — how fast can we get funded if we accept high risk?

A prop challenge is an ASYMMETRIC bet: downside is capped at the entry fee, upside
is a funded account. So the objective is NOT risk-adjusted return — it is
"cheapest/fastest expected route to a funded account", where failing is survivable
and retryable.

Key benchmark: for a DRIFTLESS random walk between +target and -maxloss, the
probability of hitting the target first is maxloss/(target+maxloss). For Flex
(+10% / -12%) that is 12/22 = 54.5% with NO edge at all. Edge and the DAILY loss
limit move it from there.

This sweeps the vol dial from conservative to reckless and reports, per dial:
  pass%, how you fail (daily vs drawdown), median months to pass,
  expected attempts to eventually pass, expected fees, expected calendar time.
"""
from __future__ import annotations
import sys
import numpy as np

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

MODEL = vbc.MODELS["flex"]          # P1 +10% / P2 +6%, daily 4%, max 12%
TARGET_VOL = vbc.TARGET_VOL
FEE = 60.0                          # ~USD entry fee for a 10K Flex challenge

# the live 10K book
CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["ETH"], "eq_us": ["DJI"]}
DIALS = [0.07, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60]


def risk_label(passpct, faildd, faildaily):
    if passpct >= 95:  return "LOW"
    if passpct >= 85:  return "MODERATE"
    if passpct >= 70:  return "HIGH"
    if passpct >= 50:  return "VERY HIGH"
    return "EXTREME"


def main():
    # build the book once at the reference dial, then rescale the return stream
    vbc.CLASSES = CLASSES
    W, book, live = vbc.build(dial=1.0)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s = (TARGET_VOL / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    dd_s = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vt = (book * (vol_s * dd_s).shift(1)).dropna().loc["2017-01-01":]
    sr = float(vt.mean() / vt.std() * np.sqrt(252))
    r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values   # normalized to 10% vol
    print(f"Book: XAU + ETH + DJI (live 10K book)   eval Sharpe {sr:+.2f}")
    print(f"Rules: FLEX  P1 +{MODEL['p1']*100:.0f}% / P2 +{MODEL['p2']*100:.0f}%, "
          f"daily -{MODEL['daily']*100:.0f}%, max -{MODEL['maxloss']*100:.0f}%   "
          f"(fee assumed ${FEE:.0f})\n")
    print(f"{'vol':>5} {'pass%':>6} {'failDay%':>9} {'failDD%':>8} {'med_mo':>7} "
          f"{'attempts':>9} {'E[fees]$':>9} {'E[months]':>10}  risk")
    print("-" * 82)
    for dial in DIALS:
        k = dial / 0.10          # r10 is a 10%-vol stream; k scales it to `dial`
        s = vbc.fp_sim(r10, k, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                       dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
        p = max(s["passpct"], 0.1) / 100.0
        attempts = 1.0 / p
        e_fees = attempts * FEE
        # failures resolve faster than passes; approximate a failed run at ~1/3 median
        e_months = s["med_mo"] + (attempts - 1) * s["med_mo"] / 3.0
        print(f"{dial*100:4.0f}% {s['passpct']:6.1f} {s['fail_day']:9.1f} "
              f"{s['fail_dd']:8.1f} {s['med_mo']:7.1f} {attempts:9.2f} "
              f"{e_fees:9.0f} {e_months:10.1f}  {risk_label(s['passpct'], s['fail_dd'], s['fail_day'])}")

    print("\nReference: a ZERO-EDGE random walk between +10% and -12% passes "
          f"{12/22*100:.1f}% of the time (ignoring the daily limit).")


if __name__ == "__main__":
    main()
