"""FundingPips challenge lab — build a FX-majors + GOLD + SILVER book that
passes the 2-Step Standard (8% -> 5%, daily 5%, max 10% static, unlimited
time) with median total time < 2 years.

Sleeves (all lookahead-free, repo conventions):
  A  XAU champion H4 long-only conc^1.5 blend (the deployed 360542 signal)
  B  FX majors D1 EWMAC trend (7 USD pairs), cluster inv-vol, buffered
  C  SILVER D1 trend (LS / long-only variants)
  D  GOLD D1 trend (reference / substitute when H4 unavailable)

Combined daily net stream -> FundingPips simulator (block bootstrap).
Results appended to results.csv. Run from repo root:
  conda run -n envmt5 python data/v5_runs/challenge-lab/challenge_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/rock/Desktop/2026_Projects/Trader36/MT5")
sys.path.insert(0, str(ROOT))
LAB = ROOT / "data" / "v5_runs" / "xau-sharpe1-lab"
sys.path.insert(0, str(LAB))

from src.v5.h4_cta import buffer_band_causal  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
ANN_H4 = int(252 * 6)
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# ------------------------------------------------------------------ loaders


def load_d1(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"data/{sym}_D1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread"] = df["spread"].clip(lower=df["spread"].median())
    return df


def load_xau_h4() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/XAUUSD_H4_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df


# ------------------------------------------------------------------ signals


def ewmac_panel(close: pd.DataFrame, speeds, cap=2.0) -> pd.DataFrame:
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in speeds:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(speeds) * 1.3).clip(-cap, cap)


SPEEDS_FAST_D1 = ((8, 32), (16, 64), (32, 128), (64, 256))
SPEEDS_SLOW_D1 = ((32, 128), (64, 256))

# ------------------------------------------------------------------ engines


def d1_sleeve(symbols, speeds=SPEEDS_FAST_D1, target_vol=0.10, buffer_frac=0.4,
              long_only=None, spread_mult=1.0, vol_hl=42) -> pd.Series:
    """D1 panel trend sleeve -> daily net return series (10% vol target).
    long_only: optional list of symbols whose forecasts are clipped >= 0."""
    dfs = {s: load_d1(s) for s in symbols}
    close = pd.DataFrame({s: d["close"] for s, d in dfs.items()}).ffill()
    spread = pd.DataFrame({s: d["spread"] for s, d in dfs.items()}).ffill()
    ret = close.pct_change(fill_method=None)
    sig = ewmac_panel(close, speeds)
    if long_only:
        for s in long_only:
            sig[s] = sig[s].clip(lower=0.0)
    sigma = ret.shift(1).ewm(halflife=vol_hl, min_periods=60).std() * np.sqrt(252)
    n = len(symbols)
    pos = (sig * (target_vol / np.sqrt(n)) / sigma).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    pos = buffer_band_causal(pos, buffer_frac)
    pos_lag = pos.shift(1)
    gross = (pos_lag * ret).sum(axis=1)
    cost = ((pos - pos.shift(1)).abs() * (spread * spread_mult / close)).sum(axis=1)
    return (gross - cost).fillna(0.0)


def xau_champ_sleeve(spread_mult=1.0, buffer_frac=0.1,
                     target_vol=0.10) -> pd.Series:
    """Deployed champion signal, continuous engine, daily net returns."""
    h4 = load_xau_h4()
    fc = champion_signal(h4["close"])
    close = h4["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc * (target_vol / vol)).clip(-8, 8)
    band = buffer_frac * (target_vol / vol).clip(0, 8)
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * \
        ((h4["spread_px"] * spread_mult / 2 + 0.10) / close)
    net = (pos * ret - cost).fillna(0.0)
    d = net.resample("D").sum()
    return d[d.index.dayofweek < 5]


# ------------------------------------------------------------------ metrics


def sharpe(d: pd.Series) -> float:
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0


def stats(d: pd.Series, tag="") -> dict:
    eq = (1 + d).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    yrs = max(len(d) / 252, 1e-9)
    cagr = (float(eq.iloc[-1]) ** (1 / yrs) - 1) * 100
    return dict(tag=tag, sharpe=round(sharpe(d), 3), maxdd=round(dd, 1),
                cagr=round(cagr, 1), worst_day=round(float(d.min()) * 100, 2),
                ann_vol=round(float(d.std() * np.sqrt(252)) * 100, 1))


def log(name, m):
    hdr = not RESULTS.exists()
    pd.DataFrame([{"name": name, **m}]).to_csv(RESULTS, mode="a",
                                               header=hdr, index=False)
    print(f"{name:38s} SR {m['sharpe']:+.3f} DD {m['maxdd']:6.1f}% "
          f"CAGR {m['cagr']:+6.1f}% vol {m['ann_vol']:4.1f}% "
          f"worst {m['worst_day']:+.2f}%")


# --------------------------------------------------- FundingPips simulator


def fp_sim(r: np.ndarray, k: float, nsim=4000, block=20, maxd=2520,
           p1=0.08, p2=0.05, dayloss=0.05, maxloss=0.10, day_safety=1.0,
           seed=7):
    """2-Step Standard. Phase transition = fresh account (return space).
    day_safety > 1 scales the daily-return breach check to proxy intraday
    floating excursions beyond close-to-close (e.g. 1.5)."""
    rng = np.random.default_rng(seed)
    n = len(r)
    passed = fail_day = fail_dd = 0
    days_list = []
    for _ in range(nsim):
        idx = []
        while len(idx) < maxd:
            s = rng.integers(0, n)
            L = min(rng.geometric(1 / block), maxd - len(idx))
            idx.extend([(s + j) % n for j in range(L)])
        x = k * r[np.array(idx)]
        eq, base, tgt, day, ok, ph2, isday = 1.0, 1.0, 1 + p1, 0, None, False, False
        for d in x:
            day += 1
            if d * day_safety < -dayloss:
                ok, isday = False, True
                break
            eq *= (1 + d)
            if eq < base * (1 - maxloss):
                ok, isday = False, False
                break
            if eq >= base * tgt:
                if not ph2:
                    ph2, base, tgt = True, eq, 1 + p2
                else:
                    ok = True
                    break
        if ok is True:
            passed += 1
            days_list.append(day)
        elif ok is False:
            fail_day += isday
            fail_dd += not isday
    med = int(np.median(days_list)) if days_list else -1
    q75 = int(np.percentile(days_list, 75)) if days_list else -1
    return dict(passpct=round(passed / nsim * 100, 1),
                fail_day=round(fail_day / nsim * 100, 1),
                fail_dd=round(fail_dd / nsim * 100, 1),
                med_days=med, med_mo=round(med / 21, 1), q75_mo=round(q75 / 21, 1))


def print_sim(name, r, ks, **kw):
    print(f"--- FP 2-Step sim: {name} (n={len(r)}) {kw or ''}")
    for k in ks:
        s = fp_sim(r, k, **kw)
        print(f"  k={k:4.2f} vol~{k*10:4.1f}%  pass {s['passpct']:5.1f}%  "
              f"failDay {s['fail_day']:4.1f}%  failDD {s['fail_dd']:4.1f}%  "
              f"median {s['med_mo']:5.1f}mo  p75 {s['q75_mo']:5.1f}mo")
