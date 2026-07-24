"""INSTRUMENT SEARCH — find a 4th sleeve that improves the FTMO book.

The biggest proven win in this project has been ADDING UNCORRELATED ASSETS
(single-XAU 1.06 -> XAU+BTC+NDX 1.71). Signal-speed tuning is exhausted
(V5_FINDINGS 3f/book sweep). So this searches every instrument we have D1 history
for, keeping only those FTMO actually offers, and asks: does adding it as a 4th
sleeve raise the challenge pass rate / cut the median time?

A candidate must clear three bars:
  1. positive standalone champion-recipe Sharpe (it must carry its own weight)
  2. low correlation to the existing sleeves (that is the whole point)
  3. the 4-asset book must actually beat the 3-asset book on pass% AND median,
     and hold up in BOTH half-samples (the check that killed H4/med)
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

M = vbc.MODELS["ftmo"]
DIAL = M["vol"] / vbc.TARGET_VOL
BASE = {"xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]}

# instruments FTMO actually offers (verified live on account 1514025597), mapped
# to the local D1 file we can backtest. FX excluded: comprehensively dead post-2016
# (V5_FINDINGS / challenge-lab), so not re-tested here.
FTMO_TRADEABLE = {
    "SILVER": "XAGUSD", "WTI": "USOIL", "BRENT": "UKOIL", "NATGAS": "NATGAS.cash",
    "SPX": "US500.cash", "DJI": "US30.cash", "DAX": "GER40.cash",
    "FTSE": "UK100.cash", "STOXX": "EU50.cash", "NIKKEI": "JP225.cash",
    "ASX": "AUS200.cash", "ETH": "ETHUSD", "LTC": "LTCUSD", "SOL": "SOLUSD",
    "AVA": "AVAUSD", "PLAT": "XPTUSD", "PALL": "XPDUSD", "COPPER": "XCUUSD",
    "CORN": "CORN.c", "WHEAT": "WHEAT.c", "SUGAR": "SUGAR.c", "COFFEE": "COFFEE.c",
    "COTTON": "COTTON.c", "SOY": "SOYBEAN.c",
}


def book_stats(classes):
    vbc.CLASSES = classes
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (M["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vt = (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]
    sr = float(vt.mean() / vt.std() * np.sqrt(252))
    eq = (1 + vt).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=M["p1"], p2=M["p2"],
                    dayloss=M["daily"], maxloss=M["maxloss"])
    h1, h2 = vt.loc[:"2020-12-31"], vt.loc["2021-01-01":]
    return dict(sr=sr, dd=dd, passpct=fp["passpct"], med=fp["med_mo"],
                sr1=float(h1.mean() / h1.std() * np.sqrt(252)),
                sr2=float(h2.mean() / h2.std() * np.sqrt(252)))


if __name__ == "__main__":
    # sleeve-level screen: standalone Sharpe + correlation to the base book
    base_nets = {}
    for s in ("XAUCHAMP", "BTC", "NDX"):
        _, nt = vbc._load_asset(s)
        base_nets[s] = nt.loc["2017-01-01":]
    base_df = pd.DataFrame(base_nets)

    print("STEP 1 — sleeve screen (standalone Sharpe + corr to XAU/BTC/NDX)\n")
    print(f"{'instrument':11s} {'FTMO symbol':13s} {'SR':>6} {'rXAU':>6} {'rBTC':>6} "
          f"{'rNDX':>6} {'maxcorr':>8}  verdict")
    cands = []
    for eng, ftmo_sym in sorted(FTMO_TRADEABLE.items()):
        try:
            _, nt = vbc._load_asset(eng)
        except Exception:
            print(f"{eng:11s} {ftmo_sym:13s}   (no local D1 data)")
            continue
        nt = nt.loc["2017-01-01":]
        j = pd.concat([base_df, nt.rename("cand")], axis=1).dropna()
        if len(j) < 500:
            print(f"{eng:11s} {ftmo_sym:13s}   (history too short: {len(j)}d)")
            continue
        sr = float(j["cand"].mean() / j["cand"].std() * np.sqrt(252))
        cor = {k: float(j["cand"].corr(j[k])) for k in ("XAUCHAMP", "BTC", "NDX")}
        mx = max(abs(v) for v in cor.values())
        ok = sr > 0.30 and mx < 0.45
        if ok:
            cands.append((eng, ftmo_sym, sr, mx))
        print(f"{eng:11s} {ftmo_sym:13s} {sr:+6.2f} {cor['XAUCHAMP']:+6.2f} "
              f"{cor['BTC']:+6.2f} {cor['NDX']:+6.2f} {mx:8.2f}  "
              f"{'CANDIDATE' if ok else ''}")

    print(f"\nSTEP 2 — add each candidate as a 4th sleeve (FTMO rules, 7% vol)\n")
    base = book_stats(BASE)
    print(f"{'book':28s} {'Sharpe':>7} {'maxDD':>7} {'pass%':>7} {'median':>8} "
          f"{'17-20':>7} {'21-26':>7}")
    print(f"{'BASE XAU+BTC+NDX':28s} {base['sr']:+7.2f} {base['dd']:6.1f}% "
          f"{base['passpct']:7.1f} {base['med']:7.1f}mo {base['sr1']:+7.2f} {base['sr2']:+7.2f}")
    results = []
    for eng, sym, sr, mx in cands:
        cls = {k: list(v) for k, v in BASE.items()}
        cls[f"x_{eng.lower()}"] = [eng]          # its own class = equal risk weight
        try:
            r = book_stats(cls)
        except Exception as e:
            print(f"  +{eng:26s} ERROR {e}"); continue
        results.append((eng, sym, r))
        better = (r["passpct"] >= base["passpct"] and r["med"] <= base["med"]
                  and min(r["sr1"], r["sr2"]) > 0.8)
        print(f"{'+ ' + eng:28s} {r['sr']:+7.2f} {r['dd']:6.1f}% {r['passpct']:7.1f} "
              f"{r['med']:7.1f}mo {r['sr1']:+7.2f} {r['sr2']:+7.2f}"
              f"{'   <-- IMPROVES' if better else ''}")

    print("\nAn addition must raise pass% AND cut median AND stay healthy in BOTH")
    print("half-samples. Anything else is noise or regime luck.")
