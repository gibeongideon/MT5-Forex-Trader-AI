"""Compare the deployed 6-class FundingPips basket vs XAU-FOCUSED books
(XAU core + a few uncorrelated adds) on the exact FP 2-Step pass sim.

Reuses the basket engine (v5_basket_challenge): champion long-only recipe per
asset, equal-class risk, portfolio vol-target + dd-scaler, then challenge_lab.fp_sim.
We just swap the CLASSES dict to define each candidate book.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

M = vbc.MODELS["standard"]
DIAL = M["vol"] / vbc.TARGET_VOL


def eval_book(classes: dict) -> dict:
    vbc.CLASSES = classes
    W, book, live = vbc.build(dial=DIAL)
    # apply causal vol-target x dd scaler as a time series (matches live)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s = (M["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    dd_s = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vtbook = (book * (vol_s * dd_s).shift(1)).dropna()

    def sh(d, s):
        d = d.loc[s:]
        return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    eq = (1 + vtbook.loc["2017-01-01":]).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    r10 = (vtbook * (0.10 / (vtbook.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=M["p1"], p2=M["p2"],
                    dayloss=M["daily"], maxloss=M["maxloss"])
    return dict(sr17=sh(vtbook, "2017-01-01"), sr21=sh(vtbook, "2021-01-01"),
                dd=dd, passpct=fp["passpct"], fail_day=fp["fail_day"],
                fail_dd=fp["fail_dd"], med=fp["med_mo"], q75=fp["q75_mo"],
                nassets=sum(len(v) for v in classes.values()))


BOOKS = {
    "6-class basket (DEPLOYED)": {
        "eq_us": ["SPX", "NDX", "DJI"], "eq_eu": ["DAX", "FTSE", "STOXX"],
        "eq_ap": ["NIKKEI", "ASX"], "crypto": ["BTC", "ETH"],
        "xau": ["XAUCHAMP"], "metal": ["SILVER"]},
    "XAU only": {"xau": ["XAUCHAMP"]},
    "XAU + BTC (50/50)": {"xau": ["XAUCHAMP"], "crypto": ["BTC"]},
    "XAU + BTC + NDX (eq 1/3)": {
        "xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]},
    "XAU-tilt 50% + BTC/NDX 25%": {
        "xau": ["XAUCHAMP"], "div": ["BTC", "NDX"]},
    "XAU + BTC + NDX + SILVER (eq 1/4)": {
        "xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"],
        "metal": ["SILVER"]},
}


def main():
    # correlation of the core sleeves (daily net, 2017+)
    print("=== correlation of core sleeves (daily net, 2017+) ===")
    nets = {}
    for s in ("XAUCHAMP", "BTC", "NDX", "SILVER", "SPX"):
        _, nt = vbc._load_asset(s)
        nets[s] = nt.loc["2017-01-01":]
    C = pd.DataFrame(nets).dropna().corr()
    print(C.round(2).to_string())

    print(f"\n=== FP 2-Step STANDARD @7% vol (realistic day_safety=1.5, eval 2017+) ===")
    print(f"{'book':34s} {'#a':>3} {'SR17':>5} {'SR21':>5} {'maxDD':>6} "
          f"{'pass%':>6} {'failDay':>7} {'failDD':>6} {'med_mo':>6}")
    rows = {}
    for name, cls in BOOKS.items():
        r = eval_book(cls)
        rows[name] = r
        print(f"{name:34s} {r['nassets']:3d} {r['sr17']:+5.2f} {r['sr21']:+5.2f} "
              f"{r['dd']:6.1f} {r['passpct']:6.1f} {r['fail_day']:7.1f} "
              f"{r['fail_dd']:6.1f} {r['med']:6.1f}")


if __name__ == "__main__":
    main()
