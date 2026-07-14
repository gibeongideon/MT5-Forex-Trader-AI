"""Campaign 7: sizing/vol-estimator refinements + regime-timing on the
long-only leaders (max-ew-bko rest0.15 ~0.967 eval; bkof rest0.15 0.983)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa

h4 = load_h4()
D = 6
MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
SLOW_D1 = ((32, 128), (64, 256))
base = ewmac_fc(h4["close"], MID)
bko_f = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
L = lambda s: s.clip(lower=0.0)
maxewbko = np.maximum(L(base), L(bko_f))
final = (maxewbko + 0.15).clip(0, 2)
bkofinal = (L(bko_f) + 0.15).clip(0, 2)

# A) vol-estimator halflife sweep (42 bars = 1wk is jumpy; Carver ~36d=216bars)
for hl in (126, 216, 378):
    for buf in (0.1, 0.2):
        m = run(h4, final, ann=ANN_H4, vol_hl=hl, buffer_frac=buf)
        log_result(f"S7-max-hl{hl}-buf{buf}", {"vol_hl": hl, "buf": buf}, m)

# B) conservative dual-vol sizing: vol = max(fast, slow) estimate
close = h4["close"]; ret = close.pct_change()
vfast = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H4)
vslow = ret.ewm(halflife=252, min_periods=60).std() * np.sqrt(ANN_H4)
vmax = np.maximum(vfast, vslow)
pos = (final * (0.10 / vmax)).clip(-8, 8)
# reuse engine by passing fc already divided: emulate via custom run
def run_pos(pos, buffer_frac=0.0, spread_mult=1.0, delay=1):
    if buffer_frac > 0:
        avg = pos.abs().rolling(500, min_periods=50).mean().shift(1).fillna(0)
        band = buffer_frac * avg
        p, out, held = pos.values, np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)
    pos = pos.shift(delay).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * ((h4["spread_px"] * spread_mult / 2 + SLIP_USD) / close)
    net = (pos * ret - cost).fillna(0.0)
    eq = (1 + net).cumprod()
    out = {}
    for tag, sl in (("full", slice(None, None)), ("eval", slice(EVAL_START, None))):
        e = eq.loc[sl]; e = e / e.iloc[0]
        d = e.resample("D").last().pct_change(fill_method=None).dropna()
        out[f"sharpe_{tag}"] = round(float(d.mean() / d.std() * np.sqrt(252)), 3)
        out[f"dd_{tag}"] = round(float((e / e.cummax() - 1).min() * 100), 1)
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        out[f"cagr_{tag}"] = round((float(e.iloc[-1]) ** (1 / yrs) - 1) * 100, 1)
    out["turnover_yr"] = round(float(pos.diff().abs().sum() /
                               ((pos.index[-1] - pos.index[0]).days / 365.25)), 1)
    out["avg_abs_pos"] = round(float(pos.abs().mean()), 2)
    return out

for buf in (0.0, 0.2):
    log_result(f"S7-max-dualvol-buf{buf}", {"vol": "max(42,252)"}, run_pos(pos, buffer_frac=buf))

# C) D1 slow-trend regime gate on fast H4 entries
d1 = h4["close"].resample("D").last().dropna()
slow_d1 = ewmac_fc(d1, SLOW_D1).shift(1).reindex(h4.index, method="ffill")
gate = (slow_d1 > 0).astype(float)
fc = (L(bko_f) * gate + 0.15).clip(0, 2)
m = run(h4, fc, ann=ANN_H4, buffer_frac=0.2)
log_result("S7-bkof-d1gate-buf0.2", {}, m)

# D) blend of the two leaders
blend = (0.5 * final + 0.5 * bkofinal).clip(0, 2)
for buf in (0.1, 0.2):
    m = run(h4, blend, ann=ANN_H4, buffer_frac=buf)
    log_result(f"S7-blend-leaders-buf{buf}", {}, m)

# E) forecast concentration: fc^1.5 (normalized past-only)
conc = (maxewbko ** 1.5)
scal = 1.0 / conc.abs().expanding(min_periods=120).mean().shift(1)
conc = (conc * scal * 0.8 + 0.15).clip(0, 2)
m = run(h4, conc, ann=ANN_H4, buffer_frac=0.2)
log_result("S7-max-conc1.5-buf0.2", {}, m)
