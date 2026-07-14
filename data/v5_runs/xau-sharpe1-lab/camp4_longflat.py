"""Campaign 4: long-flat robustness (2008+ bear-market check) + refinement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa

h4 = load_h4()
d1 = load_d1()
D = 6
MID_D = ((16, 64), (32, 128), (64, 256))
MID = tuple((f * D, s * D) for f, s in MID_D)

# --- D1 2008+ : does long-flat survive the 2011-2015 gold bear?
base_d1 = ewmac_fc(d1["close"], MID_D)
bko_d1 = breakout_fc(d1["close"], (10, 20, 40))
ens_d1 = (0.7 * base_d1 + 0.3 * bko_d1).clip(-2, 2)
for k, tag in ((1.0, "longshort"), (0.0, "longflat")):
    fc = ens_d1.where(ens_d1 > 0, ens_d1 * k)
    m = run(d1, fc, ann=ANN_D1)
    log_result(f"d1-ens-{tag}-2008", {}, m)
# buy-hold D1 2008+ benchmark
m = run(d1, pd.Series(1.0, index=d1.index), ann=ANN_D1)
log_result("d1-longonly-voltarget-2008", {}, m)

# bear-window slice: report equity stats within 2011-09..2015-12 manually
close = d1["close"]
ret = close.pct_change()
vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
for tag, fc in (("longflat", ens_d1.where(ens_d1 > 0, 0.0)),
                ("buyhold", pd.Series(1.0, index=d1.index))):
    pos = (fc * (0.10 / vol)).clip(-8, 8).shift(1).fillna(0)
    cost = pos.diff().abs().fillna(0) * ((d1["spread_px"] / 2 + 0.10) / close)
    net = (pos * ret - cost).loc["2011-09":"2015-12"].dropna()
    sh = net.mean() / net.std() * np.sqrt(252)
    eqb = (1 + net).cumprod()
    dd = (eqb / eqb.cummax() - 1).min() * 100
    print(f"BEAR 2011-09..2015-12 {tag:9s} Sharpe {sh:+.2f} total {(eqb.iloc[-1]-1)*100:+.1f}% DD {dd:.1f}%")

# --- H4 long-flat refinements
base = ewmac_fc(h4["close"], MID)
bko = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
tsm = tsmom_fc(h4["close"], [d * D for d in (21, 63, 126, 252)])

variants = {
    "ewmac-only": base,
    "ens73": (0.7 * base + 0.3 * bko),
    "ens532": (0.5 * base + 0.3 * bko + 0.2 * tsm),
    "ens433": (0.4 * base + 0.3 * bko + 0.3 * tsm),
}
for name, f in variants.items():
    fc = f.clip(-2, 2).where(f > 0, 0.0)
    m = run(h4, fc, ann=ANN_H4)
    log_result(f"h4-lf-{name}", {}, m)
    m = run(h4, fc, ann=ANN_H4, buffer_frac=0.3)
    log_result(f"h4-lf-{name}-buf0.3", {}, m)

# --- long-flat with floor exposure (never fully flat, ride drift): floor 0.3
fc = (0.7 * base + 0.3 * bko).clip(-2, 2)
fc = fc.where(fc > 0.3, 0.3)
m = run(h4, fc, ann=ANN_H4)
log_result("h4-lf-floor0.3", {}, m)

# --- target vol sweep on best long-flat (risk knob, Sharpe should be ~flat)
fc = (0.7 * base + 0.3 * bko).clip(-2, 2).where(lambda s: s > 0, 0.0)
for tv in (0.15, 0.20):
    m = run(h4, fc, ann=ANN_H4, target_vol=tv)
    log_result(f"h4-lf-ens73-tv{tv}", {"target_vol": tv}, m)
