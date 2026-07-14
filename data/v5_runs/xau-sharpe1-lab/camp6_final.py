"""Campaign 6: final grid on the long-only max-combo — buffer x resting
tilt — plus proper CI/stress computed from the SAME engine (run(...)),
so headline numbers are apples-to-apples with the rest of results.csv."""
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
ts = tsmom_fc(h4["close"], [d * D for d in (21, 63, 126, 252)])
L = lambda s: s.clip(lower=0.0)

maxewbko = np.maximum(L(base), L(bko_f))

# --- grid: resting tilt x buffer
for rest in (0.0, 0.15, 0.25):
    for buf in (0.1, 0.2, 0.3):
        fc = (maxewbko + rest).clip(0, 2)
        m = run(h4, fc, ann=ANN_H4, buffer_frac=buf)
        log_result(f"LO-max-rest{rest}-buf{buf}", {"rest": rest, "buf": buf}, m)

# --- three-leg max with buffer
fc3 = (np.maximum(maxewbko, L(ts)) + 0.15).clip(0, 2)
for buf in (0.2, 0.3):
    m = run(h4, fc3, ann=ANN_H4, buffer_frac=buf)
    log_result(f"LO-max3-rest0.15-buf{buf}", {"buf": buf}, m)

# --- bko-fast standalone buffered + tilt
for buf in (0.2, 0.3):
    m = run(h4, (L(bko_f) + 0.15).clip(0, 2), ann=ANN_H4, buffer_frac=buf)
    log_result(f"LO-bkof-rest0.15-buf{buf}", {"buf": buf}, m)


# --- exact-engine net series for CI / windows (mirror of xau_lab.run)
def net_series(fc, buffer_frac=0.0, spread_mult=1.0, delay=1, vol_hl=42,
               target_vol=0.10, max_lev=8.0):
    close = h4["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=vol_hl, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc * (target_vol / vol)).clip(-max_lev, max_lev)
    if buffer_frac > 0:
        avg = (target_vol / vol).clip(0, max_lev)
        band = buffer_frac * avg
        p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)
    pos = pos.shift(delay).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * ((h4["spread_px"] * spread_mult / 2 + SLIP_USD) / close)
    return (pos * ret - cost).fillna(0.0)


def report_ci(nm, fc, **kw):
    net = net_series(fc, **kw)
    eq = (1 + net).cumprod()
    for tag, sl in [("2017+", slice("2017-01-01", None)),
                    ("2021+", slice("2021-01-01", None)),
                    ("2015-2020", slice(None, "2020-12-31")),
                    ("full", slice(None, None))]:
        e = eq.loc[sl]
        d = e.resample("D").last().pct_change(fill_method=None).dropna()
        sh = d.mean() / d.std() * np.sqrt(252)
        lo, hi = block_bootstrap_sharpe(d.values)
        print(f"CI {nm:34s} {tag:10s} SR {sh:+.3f}  CI95 [{lo:+.2f}, {hi:+.2f}]")


final = (maxewbko + 0.15).clip(0, 2)
report_ci("FINAL max-rest0.15-buf0.2", final, buffer_frac=0.2)
report_ci("FINAL costx2", final, buffer_frac=0.2, spread_mult=2.0)
report_ci("FINAL delay2", final, buffer_frac=0.2, delay=2)
