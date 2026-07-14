"""Campaign 8: final plateau check around champion + discrete-engine mapping.
Discrete = hysteresis long-flat: go long when fc >= enter, flat when fc <= exit.
That is exactly implementable in the live bot's BUY/close framework."""
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
ew_fast = ewmac_fc(h1["close"], tuple((f*H, s*H) for f, s in ((4,16),(8,32),(16,64))))
ew_mid = ewmac_fc(h1["close"], tuple((f*H, s*H) for f, s in ((16,64),(32,128),(64,256))))
bko = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])
tsm = tsmom_fc(h1["close"], [d * H for d in (21, 63, 126, 252)])

# --- plateau grid around champion B
grids = {
    "w622": 0.6*ew_fast + 0.2*ew_mid + 0.2*bko,
    "w532": 0.5*ew_fast + 0.3*ew_mid + 0.2*bko,
    "w5221": 0.5*ew_fast + 0.2*ew_mid + 0.2*bko + 0.1*tsm,
    "w4222": 0.4*ew_fast + 0.2*ew_mid + 0.2*bko + 0.2*tsm,
}
for name, f in grids.items():
    lf = f.clip(-2, 2).where(f > 0, 0.0)
    for hl in (21, 42):
        m = run(h1, lf, ann=ANN_H1, buffer_frac=0.35, vol_hl=hl)
        log_result(f"h1-lf-{name}-hl{hl}", {"vol_hl": hl}, m)

# --- discrete hysteresis long-flat (live-bot-compatible)
close = h1["close"]; ret = close.pct_change()
vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H1)
f = (0.5*ew_fast + 0.25*ew_mid + 0.25*bko).clip(-2, 2)
for enter, exit_ in ((0.5, 0.0), (0.5, -0.25), (0.75, 0.0), (0.25, -0.25)):
    sig = f.values
    state = np.zeros(len(sig))
    on = False
    for i in range(len(sig)):
        if np.isfinite(sig[i]):
            if not on and sig[i] >= enter:
                on = True
            elif on and sig[i] <= exit_:
                on = False
        state[i] = 1.0 if on else 0.0
    pos_fc = pd.Series(state, index=h1.index)
    m = run(h1, pos_fc, ann=ANN_H1, buffer_frac=0.5)
    log_result(f"h1-DISCRETE-lf-e{enter}-x{exit_}", {"enter": enter, "exit": exit_}, m)
    m = run(h1, pos_fc, ann=ANN_H1, buffer_frac=0.5, spread_mult=2.0)
    log_result(f"h1-DISCRETE-lf-e{enter}-x{exit_}-costx2", {}, m)
