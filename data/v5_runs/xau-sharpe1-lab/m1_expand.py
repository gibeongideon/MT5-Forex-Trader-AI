"""Quest 2.1 — M1: more orthogonal books.
 a) wider membership: PLAT/PALL (metals), NATGAS/HEATOIL/GASOIL (energy)
 b) speed-split trend: FAST vs SLOW EWMAC as separate books per drift class
 c) cross-sectional momentum (xsmom) across the drift universe
 d) index dip-buy: 5d z < -1 in a 200d uptrend -> long 5d (indices only)
 e) turn-of-month: long indices last 4 + first 3 trading days of month
All streams saved causally-scaled-ready (raw net returns) to m1_streams.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from n2_drift_portfolio import (champ_recipe_lo, load_d1, ls_fast,
                                net_daily_d1, sh)  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = ["SPX", "NDX", "DJI", "DAX", "FTSE", "STOXX", "NIKKEI", "ASX"]
DRIFT_ALL = IDX + ["BTC", "ETH", "GOLD", "SILVER"]

streams = {}

# a) new members
for sym, style in (("PLAT", "lo"), ("PALL", "lo"), ("NATGAS", "ls"),
                   ("HEATOIL", "ls"), ("GASOIL", "ls")):
    try:
        df = load_d1(sym)
    except FileNotFoundError:
        print(f"{sym}: no data")
        continue
    fc = champ_recipe_lo(df["close"]) if style == "lo" else ls_fast(df["close"])
    s = net_daily_d1(df, fc)
    streams[f"{sym}"] = s
    print(f"a) {sym:8s} {style:2s}  eval {sh(s):+.3f}  full {sh(s, '2015-06-01'):+.3f}")

# b) speed-split: FAST-only and SLOW-only champion variants on core drift assets
def champ_speed(close, speeds):
    def norm(s): return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
    def conc(s, p=1.5): return norm(s.clip(lower=0.0) ** p)
    ew = ewmac_fc(close, speeds)
    bk = breakout_fc(close, (10, 20, 40))
    return (0.5 * (conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0))) * 0.8 + 0.15)
            + 0.5 * (conc(bk) * 0.8 + 0.15)).clip(0, 2)

for sym in ("SPX", "BTC", "GOLD"):
    df = load_d1(sym)
    fastb = net_daily_d1(df, champ_speed(df["close"], ((4, 16), (8, 32))))
    slowb = net_daily_d1(df, champ_speed(df["close"], ((64, 256),)))
    base = net_daily_d1(df, champ_recipe_lo(df["close"]))
    c = pd.DataFrame({"f": fastb, "s": slowb, "m": base}).dropna().corr()
    print(f"b) {sym}: fast {sh(fastb):+.2f} slow {sh(slowb):+.2f} mid {sh(base):+.2f} "
          f"corr(f,m) {c.loc['f','m']:.2f} corr(s,m) {c.loc['s','m']:.2f}")
    streams[f"{sym}_fastb"] = fastb
    streams[f"{sym}_slowb"] = slowb

# c) xsmom across drift universe: rank vol-adjusted 3m returns monthly
closes = {}
for sym in DRIFT_ALL:
    closes[sym] = load_d1(sym)["close"]
cl = pd.DataFrame(closes).ffill()
ret = cl.pct_change(fill_method=None)
vol = ret.ewm(halflife=42, min_periods=60).std() * np.sqrt(252)
mom = (cl.shift(5) / cl.shift(63) - 1.0) / (vol.shift(1) + 1e-9)
rank = mom.rank(axis=1, pct=True)
sig = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
sig[rank >= 2 / 3] = 1.0
sig[rank <= 1 / 3] = -1.0
sig = sig.resample("ME").last().reindex(cl.index, method="ffill").shift(1)
pos = (sig * (0.10 / np.sqrt(len(DRIFT_ALL))) / vol.shift(1)).replace(
    [np.inf, -np.inf], np.nan).fillna(0.0)
spread = pd.DataFrame({s: load_d1(s)["spread_px"] for s in DRIFT_ALL}).ffill()
net = ((pos.shift(1) * ret).sum(axis=1)
       - ((pos - pos.shift(1)).abs() * (spread / cl)).sum(axis=1)).fillna(0.0)
net = net[net.index.dayofweek < 5]
streams["XSMOM"] = net
print(f"c) XSMOM drift-universe: eval {sh(net):+.3f}  full {sh(net, '2015-06-01'):+.3f}")

# d) index dip-buy (5d z<-1, 200d uptrend, hold 5d)
z5 = (cl[IDX] / cl[IDX].shift(5) - 1.0) / (ret[IDX].rolling(60).std().shift(1) * np.sqrt(5))
up = cl[IDX].shift(1) > cl[IDX].shift(200)
raw_sig = ((z5 < -1.0) & up).astype(float)
hold = raw_sig.rolling(5, min_periods=1).max()  # hold 5 days after trigger
posd = (hold * (0.10 / np.sqrt(len(IDX))) / vol[IDX].shift(1)).replace(
    [np.inf, -np.inf], np.nan).fillna(0.0)
netd = ((posd.shift(1) * ret[IDX]).sum(axis=1)
        - ((posd - posd.shift(1)).abs() * (spread[IDX] / cl[IDX])).sum(axis=1)).fillna(0.0)
netd = netd[netd.index.dayofweek < 5]
streams["DIPBUY"] = netd
print(f"d) index dip-buy: eval {sh(netd):+.3f}  full {sh(netd, '2015-06-01'):+.3f}")

# e) turn-of-month long indices (last 4 + first 3 trading days)
bd = cl[IDX].dropna(how="all").index
day_of_month = pd.Series(bd, index=bd).groupby([bd.year, bd.month]).cumcount()
days_in_month = day_of_month.groupby([bd.year, bd.month]).transform("max")
tom = ((day_of_month <= 2) | (day_of_month >= days_in_month - 3)).astype(float)
post = pd.DataFrame({s: tom for s in IDX}, index=bd)
post = (post * (0.10 / np.sqrt(len(IDX))) / vol[IDX].shift(1).reindex(bd)).replace(
    [np.inf, -np.inf], np.nan).fillna(0.0)
nett = ((post.shift(1) * ret[IDX].reindex(bd)).sum(axis=1)
        - ((post - post.shift(1)).abs() * (spread[IDX].reindex(bd) / cl[IDX].reindex(bd))).sum(axis=1)).fillna(0.0)
nett = nett[nett.index.dayofweek < 5]
streams["TOM"] = nett
print(f"e) turn-of-month: eval {sh(nett):+.3f}  full {sh(nett, '2015-06-01'):+.3f}")

pd.DataFrame(streams).to_csv(os.path.join(HERE, "m1_streams.csv"))
