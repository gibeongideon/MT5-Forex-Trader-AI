"""Campaign 2: signal families on H4/D1 + ensembles with EWMAC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xau_lab import *  # noqa

h4 = load_h4()
d1 = load_d1()
D = 6

MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
FAST = tuple((f * D, s * D) for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)))

base = ewmac_fc(h4["close"], MID)

# --- individual families on H4
fams = {
    "bko-fast": breakout_fc(h4["close"], [d * D for d in (10, 20, 40)]),
    "bko-mid": breakout_fc(h4["close"], [d * D for d in (20, 40, 80)]),
    "bko-slow": breakout_fc(h4["close"], [d * D for d in (40, 80, 160)]),
    "tsmom": tsmom_fc(h4["close"], [d * D for d in (21, 63, 126, 252)]),
    "accel": accel_fc(h4["close"], MID),
    "skew120d": skew_fc(h4["close"].pct_change(), 120 * D),
}
for name, fc in fams.items():
    m = run(h4, fc, ann=ANN_H4)
    log_result(f"h4-{name}", {}, m)

# --- ensembles: ewmac-mid + each family, 50/50 and 70/30
for name, fc in fams.items():
    for w in (0.5, 0.7):
        ens = (w * base + (1 - w) * fc).clip(-2, 2)
        m = run(h4, ens, ann=ANN_H4)
        log_result(f"h4-ens-ewmid{int(w*100)}-{name}", {"w_ewmac": w}, m)

# --- three-way: ewmac + breakout-mid + tsmom
ens3 = ((base + fams["bko-mid"] + fams["tsmom"]) / 3 * 1.25).clip(-2, 2)
m = run(h4, ens3, ann=ANN_H4)
log_result("h4-ens3-ewmac-bko-tsmom", {}, m)

# --- D1 breakout for reference
fc = breakout_fc(d1["close"], (20, 40, 80, 160))
m = run(d1, fc, ann=ANN_D1)
log_result("d1-bko-20-160", {}, m)
