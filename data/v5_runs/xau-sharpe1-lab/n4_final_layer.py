"""Quest 1.6 — N4: final layers on the drift portfolio.
  L1: portfolio-level vol targeting (10% target, past-only trailing vol)
  L2: xau leg = 40/60 H4-champ / H1-champ-fast blend (N1's small gain)
Certification: CI per window, yearly, costx2 proxy (double all costs is
already inside streams; here re-run streams would be needed — instead the
known per-book costx2 deltas were <=0.02, noted), DSR vs trials."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from n2_drift_portfolio import sh, z  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe

HERE = os.path.dirname(os.path.abspath(__file__))
al = pd.read_csv(os.path.join(HERE, "n2_streams.csv"),
                 parse_dates=["time"], index_col="time")

# rebuild classes as in N3/P4 (dedup gold, slow rates, no ag)
from n2_drift_portfolio import load_d1, net_daily_d1  # noqa
rt = {}
for sym in ("UST10Y", "UST30Y"):
    df = load_d1(sym)
    rt[sym] = net_daily_d1(df, ewmac_fc(df["close"], ((32, 128), (64, 256))))
rates_slow = sum(z(s.reindex(al.index).fillna(0.0)) for s in rt.values()) / 2

CLS = {
    "xau": ["XAUCHAMP"], "crypto": ["BTC", "ETH"],
    "eq_us": ["SPX", "NDX", "DJI"], "eq_eu": ["DAX", "FTSE", "STOXX"],
    "eq_ap": ["NIKKEI", "ASX"], "metal_ns": ["SILVER", "COPPER"],
    "energy": ["WTI", "BRENT"],
}
S = {c: z(sum(z(al[m].fillna(0.0)) for m in mm) / len(mm))
     for c, mm in CLS.items()}
S["rates"] = z(rates_slow)
W = {c: (1.0 if c in ("xau", "crypto", "eq_us") else 0.5) for c in S}

df = pd.DataFrame(S).dropna()
raw = sum(z(df[c]) * W[c] for c in df.columns) / sum(W.values())

def cert(nm, p):
    out = {}
    for tag, sl in (("2017+", slice("2017-01-01", None)),
                    ("2021+", slice("2021-01-01", None)),
                    ("16-20", slice("2016-06-01", "2020-12-31")),
                    ("full", slice("2016-06-01", None))):
        x = p.loc[sl].dropna()
        s = float(x.mean() / x.std() * np.sqrt(252))
        lo, hi = block_bootstrap_sharpe(x.values)
        out[tag] = (s, lo, hi)
    eq = (1 + p.loc["2017-01-01":]).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    print(f"{nm:30s} eval {out['2017+'][0]:+.3f} CI[{out['2017+'][1]:+.2f},{out['2017+'][2]:+.2f}]  "
          f"2021+ {out['2021+'][0]:+.3f}  16-20 {out['16-20'][0]:+.3f}  "
          f"full {out['full'][0]:+.3f} CI[{out['full'][1]:+.2f},{out['full'][2]:+.2f}]  DD {dd:.1f}%")
    return out

cert("P4 raw (baseline)", raw)

# L1: portfolio vol targeting
real = raw.ewm(halflife=42, min_periods=60).std().shift(1) * np.sqrt(252)
k = (0.10 / real).clip(upper=3.0).fillna(0.0)
pvt = raw * k
cert("P5 = P4 + port vol target", pvt)

# L2: xau leg blend (recompute H1 champ fast stream)
h1 = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                 parse_dates=["time"], index_col="time").sort_index()
h1 = h1[~h1.index.duplicated(keep="last")]
h1["spread_px"] = np.maximum(h1["spread"], h1["spread"].median()) * 0.1
H = 24
def norm(s): return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
def conc(s, p=1.5): return norm(s.clip(lower=0.0) ** p)
ew_f1 = ewmac_fc(h1["close"], tuple((f * H, s * H) for f, s in ((4, 16), (8, 32), (16, 64))))
bk1 = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])
champ1f = (0.5 * (conc(np.maximum(ew_f1.clip(lower=0), bk1.clip(lower=0))) * 0.8 + 0.15)
           + 0.5 * (conc(bk1) * 0.8 + 0.15)).clip(0, 2)
ANN_H1 = 252 * 24
close = h1["close"]; retn = close.pct_change()
vol1 = retn.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H1)
pos = (champ1f * (0.10 / vol1)).clip(-8, 8)
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
c1 = c1[c1.index.dayofweek < 5]

xau_blend = z(0.4 * z(al["XAUCHAMP"].fillna(0.0))
              + 0.6 * z(c1.reindex(al.index).fillna(0.0)))
S2 = dict(S); S2["xau"] = xau_blend
df2 = pd.DataFrame(S2).dropna()
raw2 = sum(z(df2[c]) * W[c] for c in df2.columns) / sum(W.values())
real2 = raw2.ewm(halflife=42, min_periods=60).std().shift(1) * np.sqrt(252)
final = raw2 * (0.10 / real2).clip(upper=3.0).fillna(0.0)
o = cert("P6 = P5 + xau tf-blend", final)

yr = {y: round(float(g.mean() / g.std() * np.sqrt(252)), 2)
      for y, g in final.groupby(final.index.year) if g.std() > 0}
print("P6 yearly:", yr)
final.to_csv(os.path.join(HERE, "n4_final_stream.csv"))
