"""Quest 1.6 — N2: the champion recipe generalized. Thesis: the recipe
(long-only conc^1.5 max(trend, breakout) + resting tilt) is 'drift asset +
trend timing'. Apply it to EVERY structural-drift asset with D1 data and
combine with class-level risk parity + the H4 XAU champion.

Rates have no reliable drift since 2022 -> LS fast trend as diversifier.
Costs: D1 CSVs carry spread = cost_bps model in price units (pip=1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))

DRIFT = {  # asset -> class
    "SPX": "eq_us", "NDX": "eq_us", "DJI": "eq_us",
    "DAX": "eq_eu", "FTSE": "eq_eu", "STOXX": "eq_eu",
    "NIKKEI": "eq_ap", "ASX": "eq_ap",
    "BTC": "crypto", "ETH": "crypto",
    "GOLD": "metal", "SILVER": "metal", "COPPER": "metal",
}
RATES = ["UST10Y", "UST30Y"]
ENERGY = ["WTI", "BRENT"]  # LS trend (no reliable drift)


def load_d1(sym):
    df = pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = df["spread"].clip(lower=df["spread"].median())
    return df


def norm(s):
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def conc(s, p=1.5):
    return norm(s.clip(lower=0.0) ** p)


def champ_recipe_lo(close):
    ew = ewmac_fc(close, ((16, 64), (32, 128), (64, 256)))
    bk = breakout_fc(close, (10, 20, 40))
    return (0.5 * (conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0))) * 0.8 + 0.15)
            + 0.5 * (conc(bk) * 0.8 + 0.15)).clip(0, 2)


def ls_fast(close):
    return ewmac_fc(close, ((8, 32), (16, 64), (32, 128), (64, 256)))


def net_daily_d1(df, fc, buffer_frac=0.1, vol_hl=42, target_vol=0.10):
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=vol_hl, min_periods=20).std() * np.sqrt(252)
    pos = (fc * (target_vol / vol)).clip(-8, 8)
    avg = (target_vol / vol).clip(0, 8)
    band = buffer_frac * avg
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (df["spread_px"] / close)
    net = (pos * ret - cost).fillna(0.0)
    d = net.resample("D").sum()
    return d[d.index.dayofweek < 5]


def sh(d, start="2017-01-01"):
    x = d.loc[start:].dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def z(d):
    sd = d.std() * np.sqrt(252)
    return d * (0.10 / sd) if sd > 0 else d


# ---- per-asset streams
streams, classes = {}, {}
for sym, cls in DRIFT.items():
    try:
        df = load_d1(sym)
    except FileNotFoundError:
        print(f"{sym}: no data, skipped")
        continue
    s = net_daily_d1(df, champ_recipe_lo(df["close"]))
    streams[sym], classes[sym] = s, cls
    print(f"{sym:8s} [{cls:6s}] LO-recipe  eval {sh(s):+.3f}  full {sh(s, '2015-06-01'):+.3f}")
for sym in RATES + ENERGY:
    try:
        df = load_d1(sym)
    except FileNotFoundError:
        continue
    s = net_daily_d1(df, ls_fast(df["close"]))
    streams[sym], classes[sym] = s, ("rates" if sym in RATES else "energy")
    print(f"{sym:8s} [{classes[sym]:6s}] LS-fast    eval {sh(s):+.3f}  full {sh(s, '2015-06-01'):+.3f}")

# ---- H4 XAU champion stream
h4 = load_h4()
D = 6
MID4 = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
b4 = ewmac_fc(h4["close"], MID4)
k4 = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
champ4 = (0.5 * (conc(np.maximum(b4.clip(lower=0), k4.clip(lower=0))) * 0.8 + 0.15)
          + 0.5 * (conc(k4) * 0.8 + 0.15)).clip(0, 2)
close = h4["close"]; ret = close.pct_change()
vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H4)
pos = (champ4 * (0.10 / vol)).clip(-8, 8)
band = 0.1 * (0.10 / vol).clip(0, 8)
p_, out, held = pos.values, np.zeros(len(pos)), 0.0
for i in range(len(p_)):
    if np.isfinite(p_[i]):
        b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
        if abs(p_[i] - held) > b:
            held = p_[i] - np.sign(p_[i] - held) * b
    out[i] = held
pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
cost = pos.diff().abs().fillna(0.0) * ((h4["spread_px"] / 2 + SLIP_USD) / close)
nx = (pos * ret - cost).fillna(0.0).resample("D").sum()
streams["XAUCHAMP"], classes["XAUCHAMP"] = nx[nx.index.dayofweek < 5], "xau"
print(f"{'XAUCHAMP':8s} [xau   ] H4-champ   eval {sh(streams['XAUCHAMP']):+.3f}")

# ---- class composites (equal risk within class), then portfolio
al = pd.DataFrame(streams).loc["2016-01-01":]
cls_streams = {}
for cls in sorted(set(classes.values())):
    members = [s for s, c in classes.items() if c == cls]
    comp = sum(z(al[m].fillna(0.0)) for m in members) / len(members)
    cls_streams[cls] = z(comp.dropna())
    print(f"class {cls:7s} ({len(members)}): eval {sh(cls_streams[cls]):+.3f}")

cl = pd.DataFrame(cls_streams).dropna()
print("\nclass correlations:")
print(cl.corr().round(2))

port = sum(z(cl[c]) for c in cl.columns) / len(cl.columns)
print(f"\nPORTFOLIO equal-class ({len(cl.columns)} classes): "
      f"eval {sh(port):+.3f}  2021+ {sh(port, '2021-01-01'):+.3f}  "
      f"full {sh(port, '2016-06-01'):+.3f}")
eq = (1 + port.loc['2017-01-01':]).cumprod()
print(f"eval maxDD {float((eq/eq.cummax()-1).min()*100):.1f}%")
al.to_csv(os.path.join(HERE, "n2_streams.csv"))
