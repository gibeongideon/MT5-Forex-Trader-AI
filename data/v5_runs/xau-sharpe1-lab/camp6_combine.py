"""Campaign 6: H1 combined book — multi-speed EWMAC + breakout long-flat,
overnight-drift tilt overlay, buffering. Then candidates for stress."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa

def load_h1():
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df

h1 = load_h1()
ANN_H1 = 252 * 24
H = 24

FASTER = tuple((f * H, s * H) for f, s in ((4, 16), (8, 32), (16, 64)))
MID = tuple((f * H, s * H) for f, s in ((16, 64), (32, 128), (64, 256)))

ew_fast = ewmac_fc(h1["close"], FASTER)
ew_mid = ewmac_fc(h1["close"], MID)
bko = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])

# overnight tilt: +1 exposure-unit forecast during server hours 20-23,0-3
hr = h1.index.hour
overnight = pd.Series(np.where((hr >= 20) | (hr < 4), 1.0, 0.0), index=h1.index)

cands = {
    "ewfast": ew_fast,
    "ewfast-mid": 0.6 * ew_fast + 0.4 * ew_mid,
    "ewfast-bko": 0.7 * ew_fast + 0.3 * bko,
    "ewfast-mid-bko": 0.5 * ew_fast + 0.25 * ew_mid + 0.25 * bko,
}
for name, f in cands.items():
    lf = f.clip(-2, 2).where(f > 0, 0.0)
    for buf in (0.2, 0.3, 0.4):
        m = run(h1, lf, ann=ANN_H1, buffer_frac=buf)
        log_result(f"h1-lf-{name}-buf{buf}", {}, m)

# overnight tilt overlay on the best base (ewfast) at weights 0.2/0.4
lf = ew_fast.clip(-2, 2).where(ew_fast > 0, 0.0)
for w in (0.2, 0.4):
    fc = (lf + w * overnight).clip(0, 2.5)
    m = run(h1, fc, ann=ANN_H1, buffer_frac=0.3)
    log_result(f"h1-lf-ewfast-ovn{w}-buf0.3", {"ovn_w": w}, m)

# overnight-only stream for reference (is the drift net-positive after costs?)
m = run(h1, overnight, ann=ANN_H1)
log_result("h1-overnight-only", {}, m)

# long-short control of best combo (how much do shorts cost on H1 too?)
f = cands["ewfast-mid-bko"].clip(-2, 2)
m = run(h1, f, ann=ANN_H1, buffer_frac=0.3)
log_result("h1-LS-ewfast-mid-bko-buf0.3", {}, m)
