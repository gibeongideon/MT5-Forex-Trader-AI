"""Campaign 7: stress battery on top long-flat candidates.
costx2, delay2, subperiods, yearly table, block-bootstrap CI."""
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
FASTER = tuple((f * H, s * H) for f, s in ((4, 16), (8, 32), (16, 64)))
MID = tuple((f * H, s * H) for f, s in ((16, 64), (32, 128), (64, 256)))

ew_fast = ewmac_fc(h1["close"], FASTER)
ew_mid = ewmac_fc(h1["close"], MID)
bko = breakout_fc(h1["close"], [d * H for d in (10, 20, 40)])

CANDS = {
    "A-ewfast-buf0.4": (ew_fast, 0.4),
    "B-ewfast-mid-bko-buf0.3": (0.5 * ew_fast + 0.25 * ew_mid + 0.25 * bko, 0.3),
}

for name, (f, buf) in CANDS.items():
    lf = f.clip(-2, 2).where(f > 0, 0.0)
    print(f"\n=== {name} ===")
    for tag, kw in (("base", {}), ("costx2", {"spread_mult": 2.0}),
                    ("costx3", {"spread_mult": 3.0}), ("delay2", {"delay": 2}),
                    ("delay3", {"delay": 3})):
        m = run(h1, lf, ann=ANN_H1, buffer_frac=buf, **kw)
        log_result(f"stress-{name}-{tag}", kw, m)

    # subperiods + yearly + CI on the base config
    close = h1["close"]; ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H1)
    pos = (lf * (0.10 / vol)).clip(-8, 8)
    # apply same buffer logic as lab
    avg = (0.10 / vol).clip(0, 8); band = buf * avg
    p = pos.values.copy(); held = 0.0; out = np.zeros_like(p)
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0)
    cost = pos.diff().abs().fillna(0) * ((h1["spread_px"] / 2 + 0.10) / close)
    net = (pos * ret - cost).fillna(0.0)
    eq = (1 + net).cumprod()
    daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
    for lab, s in (("2017+", daily.loc["2017":]), ("2021+", daily.loc["2021":]),
                   ("2023+", daily.loc["2023":])):
        sh = s.mean() / s.std() * np.sqrt(252)
        lo, hi = block_bootstrap_sharpe(s.values)
        print(f"  {lab:6s} Sharpe {sh:+.3f} CI95 [{lo:+.2f},{hi:+.2f}]")
    yearly = daily.groupby(daily.index.year).apply(
        lambda s: s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0)
    print("  yearly:", {int(k): round(v, 2) for k, v in yearly.items()})
