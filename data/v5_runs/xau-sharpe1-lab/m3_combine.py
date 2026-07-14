"""Quest 2.1 — M3: combine surviving new books into the causal portfolio.
Candidates from M1: PALL (metals member), XSMOM, DIPBUY (new 0.5-weight
counter-trend/cross-sectional books). Allocations: drift-prior heuristic
vs ERC (equal risk contribution, trailing causal correlations).
Head-to-head vs P8-causal (the 1.56 baseline) on identical dates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from n2_drift_portfolio import load_d1, net_daily_d1, ls_fast, sh  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe

HERE = os.path.dirname(os.path.abspath(__file__))
al = pd.read_csv(os.path.join(HERE, "n2_streams.csv"),
                 parse_dates=["time"], index_col="time")
m1 = pd.read_csv(os.path.join(HERE, "m1_streams.csv"),
                 parse_dates=["time"], index_col="time")


def cz(d):
    v = d.ewm(halflife=126, min_periods=60).std().shift(1) * np.sqrt(252)
    return (d * (0.10 / v).clip(upper=5.0)).fillna(0.0)


# xau leg: H4 champion + H1 blend stream from n4 work — rebuild H1 quickly
h1 = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                 parse_dates=["time"], index_col="time").sort_index()
h1 = h1[~h1.index.duplicated(keep="last")]
h1["spread_px"] = np.maximum(h1["spread"], h1["spread"].median()) * 0.1
H = 24
norm = lambda s: s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
conc = lambda s, p=1.5: norm(s.clip(lower=0.0) ** p)
ewf = ewmac_fc(h1["close"], tuple((f * H, s * H) for f, s in ((4, 16), (8, 32), (16, 64))))
bk1 = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])
c1f = (0.5 * (conc(np.maximum(ewf.clip(lower=0), bk1.clip(lower=0))) * 0.8 + 0.15)
       + 0.5 * (conc(bk1) * 0.8 + 0.15)).clip(0, 2)
ANN_H1 = 252 * 24
close = h1["close"]; retn = close.pct_change()
vol1 = retn.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H1)
pos = (c1f * (0.10 / vol1)).clip(-8, 8)
band = 0.1 * (0.10 / vol1).clip(0, 8)
p_, out_, held = pos.values, np.zeros(len(pos)), 0.0
for i in range(len(p_)):
    if np.isfinite(p_[i]):
        b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
        if abs(p_[i] - held) > b:
            held = p_[i] - np.sign(p_[i] - held) * b
    out_[i] = held
pos = pd.Series(out_, index=pos.index).shift(1).fillna(0.0)
cost = pos.diff().abs().fillna(0.0) * ((h1["spread_px"] / 2 + SLIP_USD) / close)
c1 = (pos * retn - cost).fillna(0.0).resample("D").sum()
c1 = c1[c1.index.dayofweek < 5].reindex(al.index).fillna(0.0)

cu = load_d1("COPPER")
cu_ls = net_daily_d1(cu, ls_fast(cu["close"])).reindex(al.index).fillna(0.0)
rt = {}
for sym in ("UST10Y", "UST30Y"):
    df = load_d1(sym)
    rt[sym] = net_daily_d1(df, ewmac_fc(df["close"], ((32, 128), (64, 256)))).reindex(al.index).fillna(0.0)

g = lambda m: cz(al[m].fillna(0.0))
m = lambda k: cz(m1[k].reindex(al.index).fillna(0.0))

BASE = {
    "xau": cz(0.4 * g("XAUCHAMP") + 0.6 * cz(c1)),
    "crypto": cz(0.7 * g("BTC") + 0.3 * g("ETH")),
    "eq_us": cz((g("SPX") + g("NDX") + g("DJI")) / 3),
    "eq_eu": cz((g("DAX") + g("FTSE") + g("STOXX")) / 3),
    "eq_ap": cz((g("NIKKEI") + g("ASX")) / 2),
    "metal": cz((g("SILVER") + cz(cu_ls)) / 2),
    "energy": cz((g("WTI") + g("BRENT")) / 2),
    "rates": cz((cz(rt["UST10Y"]) + cz(rt["UST30Y"])) / 2),
}
W_BASE = {c: (1.0 if c in ("xau", "crypto", "eq_us") else 0.5) for c in BASE}

NEW = dict(BASE)
NEW["metal"] = cz((g("SILVER") + cz(cu_ls) + m("PALL")) / 3)
NEW["xsmom"] = m("XSMOM")
NEW["dipbuy"] = m("DIPBUY")
W_NEW = dict(W_BASE, metal=0.5, xsmom=0.5, dipbuy=0.5)


def combine(S, W):
    df = pd.DataFrame(S).dropna()
    raw = sum(df[c] * W[c] for c in df.columns) / sum(W.values())
    k = (0.10 / (raw.ewm(halflife=42, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3.0).fillna(0.0)
    return raw * k


def erc(S):
    """Causal ERC-ish: weights inverse to trailing avg correlation x vol
    (already vol-equalized, so ~inverse avg corr), recomputed monthly."""
    df = pd.DataFrame(S).dropna()
    w = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    roll = df.rolling(252)
    corr = df.rolling(504).corr()
    months = df.resample("MS").first().index
    cur = pd.Series(1.0, index=df.columns)
    for t0 in months:
        upto = df.loc[:t0].iloc[:-1]
        if len(upto) > 300:
            cm = upto.tail(504).corr()
            avg = cm.mean().clip(lower=0.05)
            cur = (1.0 / avg)
            cur = cur / cur.sum()
        w.loc[t0:] = cur.values
    raw = (df * w).sum(axis=1)
    k = (0.10 / (raw.ewm(halflife=42, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3.0).fillna(0.0)
    return raw * k


def cert(nm, p):
    row = []
    for tag, sl in (("2017+", slice("2017-01-01", None)),
                    ("2021+", slice("2021-01-01", None)),
                    ("full", slice("2016-06-01", None))):
        x = p.loc[sl].dropna()
        row.append(f"{tag} {float(x.mean()/x.std()*np.sqrt(252)):+.3f}")
    eq = (1 + p.loc["2017-01-01":]).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    print(f"{nm:34s} " + "  ".join(row) + f"  DD {dd:.1f}%")
    return p


base = cert("BASELINE P8-causal (1.56)", combine(BASE, W_BASE))
newp = cert("M3 + PALL/XSMOM/DIPBUY", combine(NEW, W_NEW))
ercp = cert("M3 ERC allocation", erc(NEW))
erb = cert("BASE ERC allocation", erc(BASE))

# head-to-head correlation and blend
hh = pd.DataFrame({"base": base, "new": newp}).dropna()
print("corr(base, new):", round(hh.corr().iloc[0, 1], 3))
for nm, p in (("winner-candidate new", newp), ("baseline", base)):
    x = p.loc["2017-01-01":].dropna()
    lo, hi = block_bootstrap_sharpe(x.values)
    print(f"{nm:22s} eval CI [{lo:+.2f}, {hi:+.2f}]")
newp.to_csv(os.path.join(HERE, "m3_winner_stream.csv"))
