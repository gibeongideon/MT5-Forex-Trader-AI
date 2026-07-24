"""MULTI-TIMEFRAME XAU TREND SWEEP — M30 / H1 / H4 x speed x real spreads.

Re-opens the fast-trend question now that we have TIGHT-SPREAD gold access:
  FundingPips XAUUSDmicro = $0.12   (measured 2026-07-20)
  FTMO        XAUUSD      = $0.45
  HFM cent    XAUUSDc     = $0.36   (where fast trend previously DIED, V5_FINDINGS 3c)

"Sensitive to trend changes" = shorter lookbacks + a tighter ATR trail, so the book
exits faster when a trend rolls over. Both are swept here.

Honest engine: the DISCRETE lot/stop simulator (src.v5.xau_trend.run_trades) with the
long-only champion recipe patched in — NOT the vectorized approximation, which
previously flattered fast configs by ~2x (V5_FINDINGS 3c).

    python scripts/v5_multi_tf_trend.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/rock/Desktop/2026_Projects/Trader36/MT5")
sys.path.insert(0, str(ROOT))

import src.v5.xau_trend as xt  # noqa: E402

EVAL = "2017-01-01"
CONF = {"low": 0.5, "med": 1.0, "high": 1.5}
BPD = {"M30": 48, "H1": 24, "H4": 6}          # bars per trading day

# speed sets, expressed in DAYS then converted to bars per timeframe
SPEEDS = {                        # (ewmac day-pairs, breakout day-windows)
    "ultra": ([(0.3, 1.2), (0.6, 2.4)], [0.5, 1.0]),
    "vfast": ([(0.7, 2.8), (1.4, 5.6)], [1.0, 2.0]),
    "fast":  ([(1.5, 6.0), (3.0, 12.0)], [2.5, 5.0]),
    "med":   ([(4.0, 16.0), (8.0, 32.0)], [6.0, 12.0]),
    "slow":  ([(16, 64), (32, 128), (64, 256)], [10, 20, 40]),   # = champion
}


def load(tf):
    if tf == "H4":
        h1 = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv",
                         parse_dates=["time"], index_col="time").sort_index()
        h1 = h1[~h1.index.duplicated(keep="last")]
        df = h1.resample("4h", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        return df
    f = {"M30": "data/XAUUSD_M30_long.csv", "H1": "data/XAUUSD_H1_long.csv"}[tf]
    df = pd.read_csv(ROOT / f, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close"]]


def ewmac_fc(close, pairs, cap=2.0):
    ret = close.pct_change()
    pv = close * ret.ewm(span=36, min_periods=20).std()
    comb = None
    for f, s in pairs:
        f, s = max(2, int(f)), max(3, int(s))
        raw = (close.ewm(span=f, min_periods=f).mean()
               - close.ewm(span=s, min_periods=s).mean()) / pv
        sc = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * sc).clip(-cap * 2, cap * 2)
        comb = fc if comb is None else comb + fc
    return (comb / len(pairs)).clip(-cap, cap)


def breakout_fc(close, windows, cap=2.0):
    comb = None
    for n in windows:
        n = max(4, int(n))
        hi = close.rolling(n, min_periods=n // 2).max()
        lo = close.rolling(n, min_periods=n // 2).min()
        mid = (hi + lo) / 2.0
        rng = (hi - lo).replace(0.0, np.nan)
        raw = ((close - mid) / rng * 4.0).ewm(span=max(2, n // 4)).mean()
        sc = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * sc).clip(-cap * 2, cap * 2)
        comb = fc if comb is None else comb + fc
    return (comb / len(windows)).clip(-cap, cap)


def champion_signal(tf, speed):
    bpd = BPD[tf]
    epairs, bwins = SPEEDS[speed]
    ep = [(f * bpd, s * bpd) for f, s in epairs]
    bw = [w * bpd for w in bwins]

    def _norm(s):
        return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))

    def fn(close):
        ew = ewmac_fc(close, ep)
        bk = breakout_fc(close, bw)
        mx = np.maximum(ew.clip(lower=0.0), bk.clip(lower=0.0))
        return (0.5 * (_norm(mx.clip(lower=0) ** 1.5) * 0.8 + 0.15)
                + 0.5 * (_norm(bk.clip(lower=0) ** 1.5) * 0.8 + 0.15)).clip(0, 2)
    return fn


def run(tf, speed, spread_usd, trail_atr=3.0, equity=100000.0):
    df = load(tf)[["open", "high", "low", "close"]].copy()
    df["spread"] = spread_usd / 0.1          # engine: spread(pips) * 0.1 = USD
    orig = xt.xau_signal
    xt.xau_signal = champion_signal(tf, speed)
    try:
        res = xt.run_trades(df, equity0=equity, exit_mode="trail",
                            flip_mode="confidence",
                            params=dict(conf_risk_scale=CONF, risk_frac=0.01,
                                        slippage_pips=0.2, spread_cost_mult=1.0,
                                        entry_delay_bars=1, enter_thresh=0.5,
                                        flip_thresh=1.0, sl_atr=trail_atr,
                                        trail_atr=trail_atr))
    finally:
        xt.xau_signal = orig
    eq = res["equity"].loc[EVAL:].dropna()
    tr = res["trades"]
    if len(tr) == 0 or "pnl" not in tr.columns:
        return None
    tr = tr[tr["close_time"] >= EVAL]
    d = eq.resample("D").last().pct_change(fill_method=None).dropna()
    if d.std() == 0 or len(tr) == 0:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    per_year = {int(y): round(float(g.mean() / g.std() * np.sqrt(252)), 2)
                for y, g in d.groupby(d.index.year) if g.std() > 0}
    return dict(sharpe=float(d.mean() / d.std() * np.sqrt(252)),
                dd=float((eq / eq.cummax() - 1).min() * 100),
                tr_mo=len(tr) / (yrs * 12),
                win=float((tr["pnl"] > 0).mean() * 100),
                worst_yr=min(per_year.values()) if per_year else 0.0,
                per_year=per_year)


if __name__ == "__main__":
    print("MULTI-TIMEFRAME XAU TREND (discrete engine, long-only champion recipe, "
          f"eval {EVAL}+)")
    for spread, tag in ((0.12, "FundingPips XAUUSDmicro $0.12"),
                        (0.45, "FTMO XAUUSD $0.45"),
                        (0.36, "HFM cent XAUUSDc $0.36")):
        print("\n" + "=" * 92)
        print(f"SPREAD {tag}")
        print("=" * 92)
        print(f"{'TF':>4} {'speed':>6} {'Sharpe':>7} {'maxDD%':>8} {'trades/mo':>10} "
              f"{'win%':>6} {'worst yr':>9}")
        for tf in ("M30", "H1", "H4"):
            for speed in ("ultra", "vfast", "fast", "med", "slow"):
                r = run(tf, speed, spread)
                if r is None:
                    print(f"{tf:>4} {speed:>6}    (no trades)")
                    continue
                print(f"{tf:>4} {speed:>6} {r['sharpe']:+7.2f} {r['dd']:8.1f} "
                      f"{r['tr_mo']:10.1f} {r['win']:6.1f} {r['worst_yr']:+9.2f}")
