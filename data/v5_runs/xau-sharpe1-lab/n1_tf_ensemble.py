"""Quest 1.6 — N1: timeframe ensemble. Extract daily net streams from the
H4 champion, an H1 champion-recipe book, and the H1 multi-speed runner-up;
measure correlations; sweep blends."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ANN_H1 = 252 * 24


def load_h1():
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df


def net_daily(df, fc, ann, buffer_frac=0.1, vol_hl=42, target_vol=0.10,
              spread_mult=1.0):
    """Daily net returns of the continuous engine (mirror of xau_lab.run)."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=vol_hl, min_periods=20).std() * np.sqrt(ann)
    pos = (fc * (target_vol / vol)).clip(-8, 8)
    if buffer_frac > 0:
        avg = (target_vol / vol).clip(0, 8)
        band = buffer_frac * avg
        p, out, held = pos.values, np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)
    pos = pos.shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * \
        ((df["spread_px"] * spread_mult / 2 + SLIP_USD) / close)
    net = (pos * ret - cost).fillna(0.0)
    d = net.resample("D").sum()
    return d[d.index.dayofweek < 5]


def sh(d, start="2017-01-01"):
    x = d.loc[start:]
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def report(nm, d):
    e17, e21, full = sh(d), sh(d, "2021-01-01"), sh(d, "2015-06-01")
    eq = (1 + d).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    print(f"{nm:34s} eval {e17:+.3f}  2021+ {e21:+.3f}  full {full:+.3f}  DD {dd:6.1f}%")
    return e17


def norm(s):
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def conc(s, p):
    return norm(s.clip(lower=0.0) ** p)


# ---------------- H4 champion (reference)
h4 = load_h4()
D = 6
MID4 = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
b4 = ewmac_fc(h4["close"], MID4)
k4 = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
champ4 = (0.5 * (conc(np.maximum(b4.clip(lower=0), k4.clip(lower=0)), 1.5) * 0.8 + 0.15)
          + 0.5 * (conc(k4, 1.5) * 0.8 + 0.15)).clip(0, 2)
A = net_daily(h4, champ4, ANN_H4)

# ---------------- H1 books
h1 = load_h1()
H = 24
FAST1 = tuple((f * H, s * H) for f, s in ((4, 16), (8, 32), (16, 64)))
MID1 = tuple((f * H, s * H) for f, s in ((16, 64), (32, 128), (64, 256)))
ew_f1 = ewmac_fc(h1["close"], FAST1)
ew_m1 = ewmac_fc(h1["close"], MID1)
bk1 = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])

# runner-up from camp8_final (memory): 0.6 fast + 0.2 mid + 0.2 bko, long-flat
runner = (0.6 * ew_f1 + 0.2 * ew_m1 + 0.2 * bk1).clip(-2, 2).clip(lower=0.0)
B = net_daily(h1, runner, ANN_H1, buffer_frac=0.35, vol_hl=21 * 24)

# H1 champion-recipe (same construction as H4 champion)
champ1 = (0.5 * (conc(np.maximum(ew_m1.clip(lower=0), bk1.clip(lower=0)), 1.5) * 0.8 + 0.15)
          + 0.5 * (conc(bk1, 1.5) * 0.8 + 0.15)).clip(0, 2)
C = net_daily(h1, champ1, ANN_H1, buffer_frac=0.1)

# H1 champion on FAST ewmac legs (H1 favours faster speeds per camp8)
champ1f = (0.5 * (conc(np.maximum(ew_f1.clip(lower=0), bk1.clip(lower=0)), 1.5) * 0.8 + 0.15)
           + 0.5 * (conc(bk1, 1.5) * 0.8 + 0.15)).clip(0, 2)
Cf = net_daily(h1, champ1f, ANN_H1, buffer_frac=0.1)

al = pd.DataFrame({"A_h4champ": A, "B_h1runner": B, "C_h1champ": C,
                   "Cf_h1champfast": Cf}).dropna()
print("correlations (daily, aligned):")
print(al.corr().round(2))
print()
for nm in al.columns:
    report(nm, al[nm])
print()

def z(d):
    return d * (0.10 / (d.std() * np.sqrt(252)))

best = None
for wa in (0.3, 0.4, 0.5, 0.6, 0.7):
    for other in ("B_h1runner", "Cf_h1champfast"):
        bl = wa * z(al["A_h4champ"]) + (1 - wa) * z(al[other])
        e = report(f"BLEND {wa:.0%}A + {1-wa:.0%}{other[:4]}", bl)
        if best is None or e > best[0]:
            best = (e, wa, other)
# three-way
for wa, wb in ((0.4, 0.3), (0.34, 0.33), (0.5, 0.25)):
    bl = wa * z(al["A_h4champ"]) + wb * z(al["B_h1runner"]) \
        + (1 - wa - wb) * z(al["Cf_h1champfast"])
    report(f"BLEND3 {wa}/{wb}/{round(1-wa-wb,2)}", bl)
print("\nbest 2-way:", best)
