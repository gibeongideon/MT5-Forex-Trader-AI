"""Find the best FundingPips-Flex book that is actually SIZEABLE on a $10K account.

Constraint discovered 2026-07-20: on $10K the researched XAU+BTC+NDX book cannot be
traded — min-lot notionals (gold $4,020, BTC $640) round the target lots to ZERO.
Only small-contract instruments work. This script searches the sizeable universe.

Sizeable at 10K (FTMO/prop CFD specs, min-lot notional):
  EU50 $62 | US500 $75 | AUS200 $88 | UK100 $106 | ETHUSD $186
  (marginal: GER40 $249, US100 $287, US30 $522 | dead: BTC $640, XAG $2840, XAU $4020)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

MODEL = "flex"                      # FundingPips 2-Step Flex (10%/6%, 4% daily, 12% max)
M = vbc.MODELS[MODEL]
DIAL = M["vol"] / vbc.TARGET_VOL


def eval_book(classes: dict) -> dict:
    vbc.CLASSES = classes
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s = (M["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    dd_s = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vt = (book * (vol_s * dd_s).shift(1)).dropna()

    def sh(d, s):
        d = d.loc[s:]
        return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    eq = (1 + vt.loc["2017-01-01":]).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=M["p1"], p2=M["p2"],
                    dayloss=M["daily"], maxloss=M["maxloss"])
    return dict(sr17=sh(vt, "2017-01-01"), sr21=sh(vt, "2021-01-01"), dd=dd,
                passpct=fp["passpct"], fail_day=fp["fail_day"], fail_dd=fp["fail_dd"],
                med=fp["med_mo"], W=W)


BOOKS = {
    "[ref] XAU+BTC+NDX (NOT 10K-sizeable)": {
        "xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]},
    "SPX + ETH": {"eq_us": ["SPX"], "crypto": ["ETH"]},
    "NDX + ETH": {"eq_us": ["NDX"], "crypto": ["ETH"]},
    "SPX + STOXX + ETH": {"eq_us": ["SPX"], "eq_eu": ["STOXX"], "crypto": ["ETH"]},
    "NDX + STOXX + ETH": {"eq_us": ["NDX"], "eq_eu": ["STOXX"], "crypto": ["ETH"]},
    "SPX + ASX + ETH": {"eq_us": ["SPX"], "eq_ap": ["ASX"], "crypto": ["ETH"]},
    "SPX + STOXX + ASX + ETH": {"eq_us": ["SPX"], "eq_eu": ["STOXX"],
                                "eq_ap": ["ASX"], "crypto": ["ETH"]},
    "NDX + FTSE + ASX + ETH": {"eq_us": ["NDX"], "eq_eu": ["FTSE"],
                               "eq_ap": ["ASX"], "crypto": ["ETH"]},
    "ETH only": {"crypto": ["ETH"]},
    "SPX only": {"eq_us": ["SPX"]},
}


def main():
    print("=== correlations of 10K-sizeable sleeves (daily net, 2017+) ===")
    nets = {}
    for s in ("SPX", "NDX", "STOXX", "ASX", "FTSE", "ETH"):
        _, nt = vbc._load_asset(s)
        nets[s] = nt.loc["2017-01-01":]
    print(pd.DataFrame(nets).dropna().corr().round(2).to_string())

    print(f"\n=== FundingPips FLEX (P1 10%/P2 6%, daily 4%, max 12%) @7% vol, eval 2017+ ===")
    print(f"{'book':38s} {'SR17':>5} {'SR21':>5} {'maxDD':>6} {'pass%':>6} "
          f"{'fDay':>5} {'fDD':>5} {'med_mo':>6}")
    out = {}
    for name, cls in BOOKS.items():
        r = eval_book(cls)
        out[name] = r
        print(f"{name:38s} {r['sr17']:+5.2f} {r['sr21']:+5.2f} {r['dd']:6.1f} "
              f"{r['passpct']:6.1f} {r['fail_day']:5.1f} {r['fail_dd']:5.1f} {r['med']:6.1f}")
    return out


if __name__ == "__main__":
    main()
