"""BOOK-LEVEL SPEED SWEEP — does a faster 3-asset book finish the challenge sooner?

Earlier work swept speed on XAU alone (V5_FINDINGS 3f: intraday loses, H4 wins).
This sweeps the speed of the DEPLOYED MULTI-ASSET book — the thing that actually
trades — and reports the metric that matters for a challenge: MEDIAN MONTHS TO PASS.

Speed is applied to every sleeve at once (gold on H4, the rest on D1), by scaling
the champion's EWMAC pairs / breakout windows. 'slow' == the current live recipe.

Reported per speed: Sharpe, drawdown, pass%, and median/p75 months to complete.
Faster is only better if it passes SOONER *without* giving up pass probability.
"""
from __future__ import annotations
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

# speed multipliers applied to the champion's lookbacks (1.0 = current live book)
SPEEDS = {
    "3.0x faster": 1 / 3.0,
    "2.0x faster": 1 / 2.0,
    "1.5x faster": 1 / 1.5,
    "LIVE (champion)": 1.0,
    "1.5x slower": 1.5,
}
BOOKS = {
    "FTMO 100K  (XAU+BTC+NDX)": ({"xau": ["XAUCHAMP"], "crypto": ["BTC"],
                                  "eq_us": ["NDX"]}, "ftmo"),
    "FundingPips 10K (XAU+ETH+DJI)": ({"xau": ["XAUCHAMP"], "crypto": ["ETH"],
                                       "eq_us": ["DJI"]}, "flex"),
}


def patched_recipe(mult):
    """champ_recipe_lo with all lookbacks scaled by `mult` (<1 = faster)."""
    def recipe(close):
        ep = tuple((max(2, int(f * mult)), max(3, int(s * mult)))
                   for f, s in ((16, 64), (32, 128), (64, 256)))
        bw = tuple(max(4, int(w * mult)) for w in (10, 20, 40))
        ew = vbc.ewmac_fc(close, ep)
        bk = vbc.breakout_fc(close, bw)
        return (0.5 * (vbc._conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0))) * 0.8 + 0.15)
                + 0.5 * (vbc._conc(bk) * 0.8 + 0.15)).clip(0, 2)
    return recipe


def evaluate(classes, model_name, mult):
    M = vbc.MODELS[model_name]
    dial = M["vol"] / vbc.TARGET_VOL
    orig = vbc.champ_recipe_lo
    vbc.champ_recipe_lo = patched_recipe(mult)
    vbc.CLASSES = classes
    try:
        W, book, live = vbc.build(dial=dial)
    finally:
        vbc.champ_recipe_lo = orig
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (M["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vt = (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]
    sr = float(vt.mean() / vt.std() * np.sqrt(252))
    eq = (1 + vt).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, dial, day_safety=1.5, p1=M["p1"], p2=M["p2"],
                    dayloss=M["daily"], maxloss=M["maxloss"])
    # split-sample robustness (the check that killed H4/med)
    h1 = vt.loc[:"2020-12-31"]; h2 = vt.loc["2021-01-01":]
    sr1 = float(h1.mean() / h1.std() * np.sqrt(252)) if h1.std() > 0 else 0.0
    sr2 = float(h2.mean() / h2.std() * np.sqrt(252)) if h2.std() > 0 else 0.0
    return dict(sr=sr, dd=dd, passpct=fp["passpct"], med=fp["med_mo"],
                q75=fp["q75_mo"], fday=fp["fail_day"], sr1=sr1, sr2=sr2)


if __name__ == "__main__":
    for label, (classes, model) in BOOKS.items():
        print("\n" + "=" * 96)
        print(f"{label}   model={model.upper()}   (median months = time to finish)")
        print("=" * 96)
        print(f"{'speed':>17} {'Sharpe':>7} {'maxDD':>7} {'pass%':>7} {'MEDIAN':>8} "
              f"{'p75':>7} {'failDay':>8} {'17-20':>7} {'21-26':>7}")
        for name, mult in SPEEDS.items():
            r = evaluate(classes, model, mult)
            star = "  <-- live" if name.startswith("LIVE") else ""
            print(f"{name:>17} {r['sr']:+7.2f} {r['dd']:6.1f}% {r['passpct']:7.1f} "
                  f"{r['med']:7.1f}mo {r['q75']:6.1f}mo {r['fday']:7.1f}% "
                  f"{r['sr1']:+7.2f} {r['sr2']:+7.2f}{star}")
    print("\nA faster book only wins if MEDIAN drops AND pass% holds AND both")
    print("half-samples (17-20 / 21-26) stay healthy — the test H4/med failed.")
