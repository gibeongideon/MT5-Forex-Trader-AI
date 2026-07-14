"""Campaign 9: certify champion h1-lf-w622-hl21 (0.6 ewfast + 0.2 ewmid +
0.2 breakout, long-flat, buffer 0.35, vol-target 10% hl21)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa
sys.path.insert(0, ROOT)
from src.cta.bootstrap import block_bootstrap_sharpe

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
f = (0.6*ew_fast + 0.2*ew_mid + 0.2*bko).clip(-2, 2)
lf = f.where(f > 0, 0.0)

print("=== CHAMPION: long-flat 0.6*EWMAC(4-16..16-64d) + 0.2*EWMAC(mid) + 0.2*BKO(10/20/40d), buf 0.35, tv10% hl21 ===")
for tag, kw in (("base", {}), ("costx2", {"spread_mult": 2}), ("costx3", {"spread_mult": 3}),
                ("delay2", {"delay": 2}), ("delay3", {"delay": 3}),
                ("tv15", {"target_vol": 0.15}), ("buf0.25", {"buffer_frac": 0.25}),
                ("hl42", {"vol_hl": 42})):
    kw2 = {"buffer_frac": 0.35, "vol_hl": 21, **kw}
    if "buffer_frac" in kw: kw2["buffer_frac"] = kw["buffer_frac"]
    m = run(h1, lf, ann=ANN_H1, **kw2)
    log_result(f"CHAMP-{tag}", kw, m)

# CI + yearly on base
close = h1["close"]; ret = close.pct_change()
vol = ret.ewm(halflife=21, min_periods=20).std() * np.sqrt(ANN_H1)
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
eq = (1 + net).cumprod()
daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
for lab, s in (("full", daily), ("2017+", daily.loc["2017":]),
               ("2021+", daily.loc["2021":]), ("2023+", daily.loc["2023":])):
    sh = s.mean() / s.std() * np.sqrt(252)
    lo, hi = block_bootstrap_sharpe(s.values)
    print(f"{lab:6s} Sharpe {sh:+.3f} CI95 [{lo:+.2f},{hi:+.2f}]")
yearly = daily.groupby(daily.index.year).apply(
    lambda s: s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0)
print("yearly:", {int(k): round(v, 2) for k, v in yearly.items()})
print("avg leverage:", round(float(pos.abs().mean()), 2),
      "| max leverage:", round(float(pos.abs().max()), 2),
      "| pct time in market:", round(float((pos > 0).mean()) * 100, 1))
