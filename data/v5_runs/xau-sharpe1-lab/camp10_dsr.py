"""Campaign 10: Deflated Sharpe / PSR on the champion (multiple-testing guard)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import inspect
from xau_lab import *  # noqa
sys.path.insert(0, ROOT)
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, probabilistic_sharpe_ratio

print("dsr sig:", inspect.signature(deflated_sharpe_ratio))
print("psr sig:", inspect.signature(probabilistic_sharpe_ratio))

def load_h1():
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df

h1 = load_h1(); H = 24; ANN = 252 * 24
ew_fast = ewmac_fc(h1["close"], tuple((f*H, s*H) for f, s in ((4,16),(8,32),(16,64))))
ew_mid = ewmac_fc(h1["close"], tuple((f*H, s*H) for f, s in ((16,64),(32,128),(64,256))))
bko = breakout_fc(h1["close"], [d*H for d in (10, 20, 40)])
f = (0.6*ew_fast + 0.2*ew_mid + 0.2*bko).clip(-2, 2)
lf = f.where(f > 0, 0.0)
close = h1["close"]; ret = close.pct_change()
vol = ret.ewm(halflife=21, min_periods=20).std() * np.sqrt(ANN)
pos = (lf * (0.10 / vol)).clip(-8, 8)
avg = (0.10 / vol).clip(0, 8); band = 0.35 * avg
p = pos.values.copy(); held = 0.0; out = np.zeros_like(p)
for i in range(len(p)):
    if np.isfinite(p[i]):
        b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
        if abs(p[i] - held) > b:
            held = p[i] - np.sign(p[i] - held) * b
    out[i] = held
pos = pd.Series(out, index=h1.index).shift(1).fillna(0)
cost = pos.diff().abs().fillna(0) * ((h1["spread_px"] / 2 + 0.10) / close)
net = (pos * ret - cost).fillna(0.0)
daily = (1 + net).cumprod().resample("D").last().pct_change(fill_method=None).dropna()
d17 = daily.loc["2017":].values

for n_trials in (50, 100, 200):
    try:
        print(f"DSR n_trials={n_trials}:", deflated_sharpe_ratio(d17, n_trials=n_trials))
    except TypeError:
        print(f"DSR n_trials={n_trials}:", deflated_sharpe_ratio(d17, n_trials))
try:
    print("PSR vs 0:", probabilistic_sharpe_ratio(d17, benchmark_sharpe=0.0))
except TypeError:
    print("PSR vs 0:", probabilistic_sharpe_ratio(d17, 0.0))
