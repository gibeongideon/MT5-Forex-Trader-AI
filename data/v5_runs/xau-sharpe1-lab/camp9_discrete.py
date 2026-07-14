"""Campaign 9: validate the champion long-only signal through the LIVE-style
discrete trade engine (src.v5.xau_trend.run_trades: ATR trail, next-bar
fills, spread+slippage, 1%-risk lot sizing) by monkeypatching xau_signal.

Champion forecast: 50/50 blend of concentrated (p=1.5) long-only
max(EWMAC-mid, breakout-fast) and concentrated breakout-fast, +0.15 resting
long tilt. Discrete engine enters long when forecast >= 0.5; never shorts
(forecast is never <= -0.5); exits via 2x ATR trail, re-enters if still >= 0.5.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa

sys.path.insert(0, ROOT)
import src.v5.xau_trend as xt
from src.cta.bootstrap import block_bootstrap_sharpe

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

champ = (0.5 * (conc(maxewbko, 1.5) * 0.8 + 0.15)
         + 0.5 * (conc(bko_f, 1.5) * 0.8 + 0.15)).clip(0, 2)

xt.xau_signal = lambda close: champ.reindex(close.index).fillna(0.0)

EVAL = "2017-01-01"
for exit_mode in ("trail", "flip"):
    for extra in ({}, {"sl_atr": 3.0, "trail_atr": 3.0}):
        res = xt.run_trades(h4, equity0=3000.0, exit_mode=exit_mode,
                            flip_mode="confidence", params=extra or None)
        eq = res["equity"].loc[EVAL:].dropna()
        d = eq.resample("D").last().pct_change(fill_method=None).dropna()
        sh = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
        lo, hi = block_bootstrap_sharpe(d.values)
        dd = float((eq / eq.cummax() - 1).min() * 100)
        tr = res["trades"]
        tr = tr[tr["close_time"] >= EVAL]
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (float(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100
        tag = f"{exit_mode}{'-atr3' if extra else ''}"
        print(f"DISCRETE {tag:12s} SR {sh:+.3f} CI[{lo:+.2f},{hi:+.2f}] "
              f"DD {dd:5.1f}% CAGR {cagr:+5.1f}% trades {len(tr)} "
              f"win {(tr['pnl'] > 0).mean() * 100:.0f}%")
