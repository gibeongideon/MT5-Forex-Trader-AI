"""Campaign 3: structural overlays on best ensemble — short damping, long
tilt, seasonality, vol floor. Gold 2017+ has strong secular drift; test how
much of the trend book's loss comes from shorts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa

h4 = load_h4()
D = 6
MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))

base = ewmac_fc(h4["close"], MID)
bko = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
ens = (0.7 * base + 0.3 * bko).clip(-2, 2)

# --- short damping: scale negative forecasts by k
for k in (0.0, 0.25, 0.5, 0.75):
    fc = ens.where(ens > 0, ens * k)
    m = run(h4, fc, ann=ANN_H4)
    log_result(f"h4-ens-shortdamp{k}", {"short_scale": k}, m)

# --- long tilt added to forecast
for tilt in (0.25, 0.5):
    m = run(h4, ens, ann=ANN_H4, long_tilt=tilt)
    log_result(f"h4-ens-tilt{tilt}", {"tilt": tilt}, m)

# --- combo: damp 0.5 + tilt 0.25
fc = ens.where(ens > 0, ens * 0.5)
m = run(h4, fc, ann=ANN_H4, long_tilt=0.25)
log_result("h4-ens-damp0.5-tilt0.25", {}, m)

# --- vol floor (avoid over-levering in calm regimes before vol spikes)
for q in (0.3, 0.5):
    m = run(h4, ens, ann=ANN_H4, vol_floor_q=q)
    log_result(f"h4-ens-volfloor{q}", {"vol_floor_q": q}, m)

# --- pure long-only benchmark (buy & hold vol-targeted) for context
m = run(h4, pd.Series(1.0, index=h4.index), ann=ANN_H4)
log_result("h4-longonly-voltarget", {}, m)

# --- seasonality: gold strong Dec-Feb + Aug-Sep (known pattern); overlay
mon = h4.index.month
seas = pd.Series(np.where(np.isin(mon, [12, 1, 2, 8, 9]), 0.5, 0.0), index=h4.index)
fc = (ens + seas).clip(-2, 2)
m = run(h4, fc, ann=ANN_H4)
log_result("h4-ens-seasonal-tilt", {}, m)
