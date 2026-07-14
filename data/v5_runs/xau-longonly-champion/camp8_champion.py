"""Campaign 8: champion selection — combine blend + concentration, then
full stress battery (costs, delay, buffers, subwindows, CI) on finalists."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe  # noqa

h4 = load_h4()
D = 6
MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
base = ewmac_fc(h4["close"], MID)
bko_f = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
L = lambda s: s.clip(lower=0.0)
maxewbko = np.maximum(L(base), L(bko_f))

def norm(s):
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))

def conc(s, p):
    return norm(s.clip(lower=0.0) ** p)

# candidate forecasts
CANDS = {
    # blend of leaders (camp7 D)
    "blend": (0.5 * (maxewbko + 0.15).clip(0, 2)
              + 0.5 * (L(bko_f) + 0.15).clip(0, 2)).clip(0, 2),
    # concentration on max (camp7 E)
    "conc1.5": (conc(maxewbko, 1.5) * 0.8 + 0.15).clip(0, 2),
    # blend + concentration
    "blend-conc1.5": (0.5 * (conc(maxewbko, 1.5) * 0.8 + 0.15)
                      + 0.5 * (conc(bko_f, 1.5) * 0.8 + 0.15)).clip(0, 2),
    # exponent variants on max
    "conc1.25": (conc(maxewbko, 1.25) * 0.8 + 0.15).clip(0, 2),
    "conc2.0": (conc(maxewbko, 2.0) * 0.8 + 0.15).clip(0, 2),
}

for nm, fc in CANDS.items():
    for buf in (0.1, 0.2):
        m = run(h4, fc, ann=ANN_H4, buffer_frac=buf)
        log_result(f"S8-{nm}-buf{buf}", {"buf": buf}, m)

# ---- full battery on the two finalists
def battery(nm, fc, buf):
    print(f"\n=== BATTERY {nm} (buf={buf}) ===")
    for tag, kw in [("base", {}), ("costx2", dict(spread_mult=2.0)),
                    ("costx3", dict(spread_mult=3.0)), ("delay2", dict(delay=2))]:
        m = run(h4, fc, ann=ANN_H4, buffer_frac=buf, **kw)
        print(f"  {tag:8s} eval {m['sharpe_eval']:+.3f} full {m['sharpe_full']:+.3f} "
              f"DD {m['dd_eval']:5.1f}% CAGR {m['cagr_eval']:+5.1f}% turn {m['turnover_yr']:5.1f}")
    # exact-engine daily CI per window
    close = h4["close"]; ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc * (0.10 / vol)).clip(-8, 8)
    avg = (0.10 / vol).clip(0, 8)
    band = buf * avg
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0) * ((h4["spread_px"] / 2 + SLIP_USD) / close)
    net = (pos * ret - cost).fillna(0.0)
    eq = (1 + net).cumprod()
    for tag, sl in [("2017+", slice("2017-01-01", None)),
                    ("2021+", slice("2021-01-01", None)),
                    ("2015-2020", slice(None, "2020-12-31")),
                    ("full", slice(None, None))]:
        e = eq.loc[sl]
        d = e.resample("D").last().pct_change(fill_method=None).dropna()
        sh = d.mean() / d.std() * np.sqrt(252)
        lo, hi = block_bootstrap_sharpe(d.values)
        print(f"  CI {tag:10s} SR {sh:+.3f}  CI95 [{lo:+.2f}, {hi:+.2f}]")
    # yearly Sharpe consistency
    yr = net.resample("YE").apply(
        lambda x: (x.resample("D").sum().pipe(
            lambda d: d.mean() / d.std() * np.sqrt(252)) if x.std() > 0 else 0.0))
    print("  yearly:", {i.year: round(float(v), 2) for i, v in yr.items()})

battery("blend", CANDS["blend"], 0.1)
battery("blend-conc1.5", CANDS["blend-conc1.5"], 0.1)
battery("conc1.5", CANDS["conc1.5"], 0.2)
