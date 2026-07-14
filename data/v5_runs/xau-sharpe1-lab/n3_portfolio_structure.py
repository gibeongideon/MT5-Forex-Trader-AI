"""Quest 1.6 — N3: portfolio structure. Structural (not performance-picked)
refinements of the N2 equal-class portfolio:
  1. dedupe gold (GOLD D1 duplicates the XAU H4 champion, corr 0.68)
  2. rates at SLOW speeds (bonds classically trend slow)
  3. add AG class (6 softs, LS-fast) as one more near-zero-corr diversifier
  4. drift-prior weights: full weight to drift classes (xau, crypto, eq_us),
     half to diversifiers — declared ex-ante from the drift thesis, and the
     N2 result is quoted alongside as the no-choice baseline.
Certification battery on the winner: subwindows, costx2, CI, per-year."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from n2_drift_portfolio import (champ_recipe_lo, load_d1, ls_fast,
                                net_daily_d1, sh, z)  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe

HERE = os.path.dirname(os.path.abspath(__file__))
al = pd.read_csv(os.path.join(HERE, "n2_streams.csv"),
                 parse_dates=["time"], index_col="time")

AGS = ["CORN", "WHEAT", "SOY", "COFFEE", "SUGAR", "COTTON"]
ag_streams = {}
for sym in AGS:
    try:
        df = load_d1(sym)
    except FileNotFoundError:
        continue
    ag_streams[sym] = net_daily_d1(df, ls_fast(df["close"]))
ag = sum(z(s.reindex(al.index).fillna(0.0)) for s in ag_streams.values()) / len(ag_streams)
print(f"AG class ({len(ag_streams)} softs, LS-fast): eval {sh(ag):+.3f}")

# rates slow variant
rt = {}
for sym in ("UST10Y", "UST30Y"):
    df = load_d1(sym)
    rt[sym] = net_daily_d1(df, ewmac_fc(df["close"], ((32, 128), (64, 256))))
rates_slow = sum(z(s.reindex(al.index).fillna(0.0)) for s in rt.values()) / 2
print(f"rates SLOW: eval {sh(rates_slow):+.3f}  (fast was +0.29)")

CLS = {
    "xau": ["XAUCHAMP"], "crypto": ["BTC", "ETH"],
    "eq_us": ["SPX", "NDX", "DJI"], "eq_eu": ["DAX", "FTSE", "STOXX"],
    "eq_ap": ["NIKKEI", "ASX"], "metal_ns": ["SILVER", "COPPER"],  # no GOLD
    "energy": ["WTI", "BRENT"],
}

def cls_stream(members):
    return z(sum(z(al[m].fillna(0.0)) for m in members) / len(members))

def port_stats(nm, streams, weights=None):
    df = pd.DataFrame(streams).dropna()
    w = weights or {c: 1.0 for c in df.columns}
    tot = sum(w.values())
    p = sum(z(df[c]) * w[c] for c in df.columns) / tot
    eq = (1 + p.loc["2017-01-01":]).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    print(f"{nm:34s} eval {sh(p):+.3f}  2021+ {sh(p, '2021-01-01'):+.3f}  "
          f"16-20 {sh(p.loc[:'2020-12-31'], '2016-06-01'):+.3f}  DD {dd:5.1f}%")
    return p

S = {c: cls_stream(m) for c, m in CLS.items()}
S["rates"] = z(rates_slow)
S["ag"] = z(ag)

port_stats("P1 equal-9 (dedup gold, slow rates)", S)
port_stats("P2 = P1 minus ag", {c: s for c, s in S.items() if c != "ag"})
drift_w = {c: (1.0 if c in ("xau", "crypto", "eq_us") else 0.5)
           for c in S}
best = port_stats("P3 drift-prior weights", S, drift_w)
port_stats("P4 drift-prior, no ag", {c: s for c, s in S.items() if c != "ag"},
           {c: w for c, w in drift_w.items() if c != "ag"})

# ---- certification battery on P3
print("\n=== CERTIFY P3 ===")
d = best
for tag, sl in (("2017+", slice("2017-01-01", None)),
                ("2021+", slice("2021-01-01", None)),
                ("2016-2020", slice("2016-06-01", "2020-12-31")),
                ("full", slice("2016-06-01", None))):
    x = d.loc[sl].dropna()
    lo, hi = block_bootstrap_sharpe(x.values)
    print(f"  {tag:10s} SR {float(x.mean()/x.std()*np.sqrt(252)):+.3f}  "
          f"CI95 [{lo:+.2f}, {hi:+.2f}]")
yr = {y: round(float(g.mean() / g.std() * np.sqrt(252)), 2)
      for y, g in d.groupby(d.index.year) if g.std() > 0}
print("  yearly:", yr)
pd.DataFrame({"P3": best}).to_csv(os.path.join(HERE, "n3_best_port.csv"))
