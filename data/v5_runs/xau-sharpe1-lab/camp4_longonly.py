"""Campaign 4: refine the long-only discovery.

Campaign 3 found zeroing shorts lifts the trend ensemble to eval 0.931
(full 0.809) vs 0.757 long/short. Refine: family x weights, long-only with
resting tilt (hold small long when trend flat), tsmom third leg.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa

h4 = load_h4()
D = 6
MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
SLOW = tuple((f * D, s * D) for f, s in ((32, 128), (64, 256)))
FAST = tuple((f * D, s * D) for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)))

base = ewmac_fc(h4["close"], MID)
bko_f = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
bko_s = breakout_fc(h4["close"], [d * D for d in (40, 80, 160)])
ts = tsmom_fc(h4["close"], [d * D for d in (21, 63, 126, 252)])

# --- single families, long-only
for nm, fc in [("ewmac-mid", base),
               ("ewmac-slow", ewmac_fc(h4["close"], SLOW)),
               ("ewmac-fast", ewmac_fc(h4["close"], FAST)),
               ("bko-fast", bko_f), ("bko-slow", bko_s), ("tsmom", ts)]:
    m = run(h4, fc.clip(lower=0.0), ann=ANN_H4)
    log_result(f"LO-{nm}", {"long_only": True}, m)

# --- ensemble weight grid, long-only
for w in (0.5, 0.6, 0.7, 0.8):
    ens = (w * base + (1 - w) * bko_f).clip(-2, 2)
    m = run(h4, ens.clip(lower=0.0), ann=ANN_H4)
    log_result(f"LO-ens-ew{w}-bkof", {"w": w}, m)

# --- 3-leg ensemble long-only
ens3 = (0.5 * base + 0.25 * bko_f + 0.25 * ts).clip(-2, 2)
m = run(h4, ens3.clip(lower=0.0), ann=ANN_H4)
log_result("LO-ens3-ew50-bko25-ts25", {}, m)

# --- long-only + resting long tilt (hold small long when trend flat/neg)
ens = (0.7 * base + 0.3 * bko_f).clip(-2, 2)
for tilt in (0.15, 0.25, 0.4):
    fc = np.maximum(ens, 0.0) + tilt
    m = run(h4, fc.clip(0, 2), ann=ANN_H4)
    log_result(f"LO-ens-resting{tilt}", {"resting": tilt}, m)

# --- max(trend, breakout) long-only: enter on either confirmation
fc = np.maximum(base.clip(lower=0), bko_f.clip(lower=0))
m = run(h4, fc, ann=ANN_H4)
log_result("LO-max-ew-bko", {}, m)

# --- buffered best (execution-friendly turnover)
ens = (0.7 * base + 0.3 * bko_f).clip(-2, 2).clip(lower=0.0)
for b in (0.2, 0.4):
    m = run(h4, ens, ann=ANN_H4, buffer_frac=b)
    log_result(f"LO-ens-buf{b}", {"buffer": b}, m)
