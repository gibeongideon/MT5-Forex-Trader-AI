"""Quest 1.6 — N6: Sharpe vs capital under real lot quantization.

For each leg: effective exposure fraction of equity = engine position
(10%-vol book) x class weight x within-class split x top-level vol-target
multiplier. At equity E the tradable exposure is rounded to whole
multiples of the instrument's minimum lot (in units), then net returns are
recomputed on the QUANTIZED exposure incl. costs on quantized turnover.

Min-lot assumptions (HFM standard CFDs; XAU verified on cent terminal;
others published specs — re-verify before live): see MINUNITS below.
Rates class excluded (untradable at HFM).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from xau_lab import *  # noqa
from n2_drift_portfolio import load_d1, champ_recipe_lo, ls_fast  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe

HERE = os.path.dirname(os.path.abspath(__file__))

# instrument -> (class, style, min units per lot-step)
LEGS = {
    "XAUUSD": ("xau", "champ_h4", 1.0),        # 1 oz (standard acct)
    "BTC": ("crypto", "lo", 0.01),
    "ETH": ("crypto", "lo", 0.01),
    "SPX": ("eq_us", "lo", 0.1),               # 0.1 index units
    "NDX": ("eq_us", "lo", 0.05),
    "DJI": ("eq_us", "lo", 0.01),
    "DAX": ("eq_eu", "lo", 0.01),
    "FTSE": ("eq_eu", "lo", 0.01),
    "STOXX": ("eq_eu", "lo", 0.1),
    "NIKKEI": ("eq_ap", "lo", 0.01),
    "ASX": ("eq_ap", "lo", 0.01),
    "SILVER": ("metal", "lo", 50.0),           # 50 oz
    "COPPER": ("metal", "ls_fast", 250.0),     # ~$1.2k step
    "WTI": ("energy", "ls_fast", 10.0),        # 10 bbl
    "BRENT": ("energy", "ls_fast", 10.0),
}
CLASS_W = {"xau": 1.0, "crypto": 1.0, "eq_us": 1.0,
           "eq_eu": 0.5, "eq_ap": 0.5, "metal": 0.5, "energy": 0.5}
WITHIN = {"crypto": {"BTC": 0.7, "ETH": 0.3}}  # others equal split
TOTW = sum(CLASS_W.values())  # 5.0 (rates dropped)


def positions_frac(close, fc, ann, buffer_frac=0.1, vol_hl=42, tv=0.10):
    ret = close.pct_change()
    vol = ret.ewm(halflife=vol_hl, min_periods=20).std() * np.sqrt(ann)
    pos = (fc * (tv / vol)).clip(-8, 8)
    band = buffer_frac * (tv / vol).clip(0, 8)
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return pd.Series(out, index=close.index)


# ---- build daily exposure fraction + price + cost per leg
legs = {}
for sym, (cls, style, minu) in LEGS.items():
    if style == "champ_h4":
        df = load_h4()
        fc = None
        D = 6
        norm = lambda s: s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
        conc = lambda s, q=1.5: norm(s.clip(lower=0.0) ** q)
        b4 = ewmac_fc(df["close"], tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256))))
        k4 = breakout_fc(df["close"], [d * D for d in (10, 20, 40)])
        fc = (0.5 * (conc(np.maximum(b4.clip(lower=0), k4.clip(lower=0))) * 0.8 + 0.15)
              + 0.5 * (conc(k4) * 0.8 + 0.15)).clip(0, 2)
        pos = positions_frac(df["close"], fc, ANN_H4)
        px = df["close"]
        cost_frac = (df["spread_px"] / 2 + SLIP_USD) / px
    else:
        df = load_d1(sym)
        fc = champ_recipe_lo(df["close"]) if style == "lo" else ls_fast(df["close"])
        pos = positions_frac(df["close"], fc, 252)
        px = df["close"]
        cost_frac = df["spread_px"] / px
    # resample to daily
    d = pd.DataFrame({"pos": pos, "px": px, "cf": cost_frac})
    dd = d.resample("D").last().ffill()
    dd["ret"] = dd["px"].pct_change(fill_method=None)
    dd = dd[dd.index.dayofweek < 5]
    w = CLASS_W[cls] / TOTW * WITHIN.get(cls, {}).get(
        sym, 1.0 / sum(1 for s, (c, _, _) in LEGS.items() if c == cls))
    legs[sym] = dict(df=dd.loc["2016-01-01":], w=w, minu=minu)

idx = None
for L in legs.values():
    idx = L["df"].index if idx is None else idx.intersection(L["df"].index)

# raw (unquantized) portfolio for the top-level vol multiplier
raw = sum(L["df"].loc[idx, "pos"].shift(1) * L["df"].loc[idx, "ret"] * L["w"]
          for L in legs.values()).fillna(0.0)
k_top = (0.10 / (raw.ewm(halflife=42, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3.0).fillna(0.0)


def run_at_equity(E):
    tot = pd.Series(0.0, index=idx)
    for sym, L in legs.items():
        d = L["df"].loc[idx]
        expo = d["pos"] * L["w"] * k_top          # fraction of equity
        if np.isfinite(E):
            units = expo * E / d["px"]
            step = L["minu"]
            units_q = np.round(units / step) * step
            expo_q = units_q * d["px"] / E
        else:
            expo_q = expo
        gross = expo_q.shift(1) * d["ret"]
        cost = expo_q.diff().abs().fillna(0.0) * d["cf"]
        tot = tot + (gross - cost).fillna(0.0)
    return tot


print(f"{'equity':>10} {'SR eval':>8} {'SR 2021+':>9} {'evalDD':>7}")
for E in (np.inf, 200_000, 100_000, 50_000, 25_000, 10_000, 5_000):
    p = run_at_equity(E)
    x = p.loc["2017-01-01":]
    sr = float(x.mean() / x.std() * np.sqrt(252))
    x21 = p.loc["2021-01-01":]
    sr21 = float(x21.mean() / x21.std() * np.sqrt(252))
    eq = (1 + x).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    lab = "unquantized" if not np.isfinite(E) else f"${E:,.0f}"
    print(f"{lab:>10} {sr:>+8.3f} {sr21:>+9.3f} {dd:>6.1f}%")
