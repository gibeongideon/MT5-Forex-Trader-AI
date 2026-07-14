"""Campaign 1: EWMAC speed sets x timeframe x buffering, vol-targeted."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xau_lab import *  # noqa

h4 = load_h4()
d1 = load_d1()

D = 6  # H4 bars per day
SETS = {
    "vfast": ((2, 8), (4, 16), (8, 32)),
    "fast": ((8, 32), (16, 64), (32, 128), (64, 256)),
    "mid": ((16, 64), (32, 128), (64, 256)),
    "slow": ((32, 128), (64, 256)),
    "all6": ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)),
}

for name, spd in SETS.items():
    # D1 (2008+)
    fc = ewmac_fc(d1["close"], spd)
    m = run(d1, fc, ann=ANN_D1)
    log_result(f"d1-ewmac-{name}", {"speeds": spd, "tf": "D1"}, m)
    # H4 (2015+), daily-equivalent speeds x6
    spd_h4 = tuple((f * D, s * D) for f, s in spd)
    fc = ewmac_fc(h4["close"], spd_h4)
    m = run(h4, fc, ann=ANN_H4)
    log_result(f"h4-ewmac-{name}", {"speeds": spd_h4, "tf": "H4"}, m)

# buffering sweep on the two most promising bases (fast/mid H4+D1)
for tf, df, ann, mult in (("d1", d1, ANN_D1, 1), ("h4", h4, ANN_H4, 6)):
    for name in ("fast", "mid"):
        spd = tuple((f * mult, s * mult) for f, s in SETS[name])
        fc = ewmac_fc(df["close"], spd)
        for buf in (0.3, 0.6):
            m = run(df, fc, ann=ann, buffer_frac=buf)
            log_result(f"{tf}-ewmac-{name}-buf{buf}", {"speeds": spd, "buffer": buf}, m)

# no-vol-target control (raw forecast, fixed leverage 1x per unit forecast)
for name in ("fast", "slow"):
    fc = ewmac_fc(d1["close"], SETS[name])
    m = run(d1, fc, ann=ANN_D1, target_vol=0.10, vol_hl=10000)  # ~static vol
    log_result(f"d1-ewmac-{name}-novoltarget", {"speeds": SETS[name]}, m)
