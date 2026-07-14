"""Campaign 2: hunt a positive FX-majors stream + better SILVER sleeve.
FX trend is dead 2016+ (-0.36). Try: per-pair scan, xsmom, D1 mean-reversion,
carry (if rates available); SILVER champion-recipe (max ewmac/bko, conc^1.5,
long-only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from challenge_lab import *  # noqa

START = "2016-01-01"

# --- per-pair D1 fast trend 2016+ (diagnostic)
for s in FX:
    d = d1_sleeve([s], SPEEDS_FAST_D1).loc[START:]
    print(f"pair {s}: SR {sharpe(d):+.2f}")

# --- cross-sectional momentum across 7 majors (rank 3m/12m return, top-bottom)
dfs = {s: load_d1(s) for s in FX}
close = pd.DataFrame({s: d["close"] for s, d in dfs.items()}).ffill()
spread = pd.DataFrame({s: d["spread"] for s, d in dfs.items()}).ffill()
ret = close.pct_change(fill_method=None)
# express every pair as USD-quote direction consistency doesn't matter for xs rank
for lb, skip, nm in ((252, 21, "xs12m"), (63, 5, "xs3m")):
    r = close.shift(skip) / close.shift(lb) - 1.0
    def rank_row(row):
        v = row.dropna()
        out = pd.Series(0.0, index=row.index)
        if len(v) < 4: return out
        lo, hi = v.quantile(1/3), v.quantile(2/3)
        out[row >= hi] = 1.0; out[row <= lo] = -1.0
        return out
    sig = r.apply(rank_row, axis=1)
    sigma = ret.shift(1).ewm(halflife=42, min_periods=60).std() * np.sqrt(252)
    pos = (sig * (0.10 / np.sqrt(7)) / sigma).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    pos = buffer_band_causal(pos, 0.3)
    net = ((pos.shift(1) * ret).sum(axis=1)
           - ((pos - pos.shift(1)).abs() * (spread / close)).sum(axis=1)).fillna(0.0)
    log(f"FX-{nm}", stats(net.loc[START:]))

# --- D1 mean reversion: fade 3-day move against 200d trend direction
z = (close / close.shift(3) - 1.0) / (ret.rolling(60).std().shift(1) * np.sqrt(3))
trend = np.sign(close.shift(1) / close.shift(200) - 1.0)
sig = (-np.sign(z) * (z.abs() > 1.0)).where(lambda s: (s * trend) >= 0, 0.0)
sigma = ret.shift(1).ewm(halflife=42, min_periods=60).std() * np.sqrt(252)
pos = (sig * (0.10 / np.sqrt(7)) / sigma).replace([np.inf,-np.inf], np.nan).fillna(0.0)
net = ((pos.shift(1) * ret).sum(axis=1)
       - ((pos - pos.shift(1)).abs() * (spread / close)).sum(axis=1)).fillna(0.0)
log("FX-meanrev-trendgated", stats(net.loc[START:]))

# --- FX carry (if rates file exists)
try:
    from src.v5.levers import load_rates, carry_signal
    rates = load_rates()
    from src.cta.signals import fx_carry
    from src.cta.universe import FX_PAIRS
    car = fx_carry(close.index, rates, FX_PAIRS, FX)
    pos = (car * (0.10 / np.sqrt(7)) / sigma).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    pos = buffer_band_causal(pos, 0.3)
    net = ((pos.shift(1) * ret).sum(axis=1)
           - ((pos - pos.shift(1)).abs() * (spread / close)).sum(axis=1)).fillna(0.0)
    log("FX-carry", stats(net.loc[START:]))
except Exception as e:
    print("carry skipped:", e)

# --- SILVER champion recipe (max ewmac/bko, conc^1.5, long-only, resting tilt)
sv = load_d1("SILVER")
c = sv["close"]
def norm(s): return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
def conc(s, p): return norm(s.clip(lower=0.0) ** p)
def bko(cl, windows):
    combined = None
    for n_ in windows:
        hi = cl.rolling(n_, min_periods=n_//2).max(); lo = cl.rolling(n_, min_periods=n_//2).min()
        rng_ = (hi - lo).replace(0.0, np.nan)
        raw = ((cl - (hi+lo)/2) / rng_ * 4.0).ewm(span=max(2, n_//4)).mean()
        fc = (raw * (1.0/raw.abs().expanding(min_periods=60).mean().shift(1))).clip(-4, 4)
        combined = fc if combined is None else combined + fc
    return (combined / len(windows) * 1.2).clip(-2, 2)
ew = ewmac_panel(c.to_frame("S"), ((16,64),(32,128),(64,256)))["S"]
bk = bko(c, (10, 20, 40))
champ_ag = (0.5*(conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0)), 1.5)*0.8 + 0.15)
            + 0.5*(conc(bk, 1.5)*0.8 + 0.15)).clip(0, 2)
retS = c.pct_change()
sigmaS = retS.shift(1).ewm(halflife=42, min_periods=60).std() * np.sqrt(252)
posS = (champ_ag * 0.10 / sigmaS).replace([np.inf,-np.inf], np.nan).fillna(0.0)
posS = buffer_band_causal(posS.to_frame("S"), 0.1)["S"]
netS = (posS.shift(1)*retS - (posS.diff().abs().fillna(0))*(sv["spread"]/c)).fillna(0.0)
log("SILVER-champrecipe-LO", stats(netS.loc[START:]))
log("SILVER-champrecipe-LO-2008+", stats(netS.loc["2008-06-01":]))
netS.loc[START:].to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "silver_champ_stream.csv"))
