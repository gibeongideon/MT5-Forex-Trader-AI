"""single_instrument_strategies.py — five SIMPLE single-instrument formulations, honest test.

Different problem formulations from the (dead) directional-prediction class. All on deep
Dukascopy M15 → D1 (real per-bar spread, 2015-2026), vol-targeted, net of real cost, with
discover(2015-21)/confirm(2022-26) + block-bootstrap 95% CI + GO gate. TEXTBOOK params only
(no in-sample optimization) — the discipline is what makes the verdict trustworthy.

  1. TREND   — Carver EWMAC continuous trend forecast (reuse src/cta/signals.ewmac)
  2. VOLMAN  — is volatility predictable? + does vol-targeting the trend add value?
  3. BREAK   — Donchian channel breakout (20-day entry / 10-day exit), classic Turtle
  4. CARRY   — rate-differential carry (FRED 3M), structural, +trend-filtered variant
  5. REVERT  — short-term mean-reversion (20-day z-score, ±1.5 enter / 0.5 exit)

GO = confirm net Sharpe >= +0.5 AND bootstrap CI lower bound > 0 AND discover Sharpe > 0.
Cardinal rule: any Sharpe >> 1 or hit ~100% => audit.

Usage:
    python scripts/single_instrument_strategies.py                 # all 4 instruments, all 5
    python scripts/single_instrument_strategies.py --symbol XAUUSD
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.signals import ewmac
from src.cta.bootstrap import block_bootstrap_sharpe
from scripts.backtest_champion_baseline import _load_raw

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}
# pair → (base ccy, quote ccy) for carry = r_base - r_quote (long pair earns this)
CCY = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
       "USDJPY": ("USD", "JPY"), "XAUUSD": (None, None)}  # gold: no yield → no carry sleeve
COMM_PIPS = 0.5
SPLIT = pd.Timestamp("2022-01-01")
VOL_TARGET = 0.15      # 15% annualized per-strategy target
POS_CAP = 4.0


# ───────────────────────────── data ─────────────────────────────
def resample_d1(sym: str) -> pd.DataFrame:
    d = _load_raw(ROOT / "data" / f"{sym}_M15_long.csv")
    o = d.resample("1D", label="left", closed="left")
    df = pd.DataFrame({
        "open": o["open"].first(), "high": o["high"].max(), "low": o["low"].min(),
        "close": o["close"].last(), "spread": o["spread"].mean(),
    }).dropna(subset=["open"])
    return df[df.index.dayofweek < 5]


def load_rates() -> pd.DataFrame:
    r = pd.read_csv(ROOT / "data" / "rates_3m.csv", index_col=0, parse_dates=True)
    return r.sort_index()


def bars_per_year(idx) -> float:
    span = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / span if span > 0 else 252.0


# ───────────────────────── sizing + cost ────────────────────────
def vol_scale(pos_raw: pd.Series, ret: pd.Series, bpy: float) -> pd.Series:
    """Scale a directional signal in [-1,1] to VOL_TARGET using PAST-ONLY instrument vol
    (Carver position = signal * cash_vol_target / instrument_vol). Lookahead-free (shifted)."""
    inst_vol = (ret.ewm(span=36, min_periods=20).std() * np.sqrt(bpy)).shift(1)
    scalar = (VOL_TARGET / inst_vol).replace([np.inf, -np.inf], np.nan)
    return (pos_raw * scalar).clip(-POS_CAP, POS_CAP).fillna(0.0)


def net_returns(pos: pd.Series, ret: pd.Series, cost_rate: pd.Series,
                extra_ret: pd.Series | None = None) -> pd.Series:
    """Daily net return: yesterday's position earns today's return (+ optional carry),
    minus turnover * one-way cost. Position uses .shift(1) → strictly lookahead-free."""
    held = pos.shift(1).fillna(0.0)
    gross = held * ret
    if extra_ret is not None:
        gross = gross + held * extra_ret
    turn = (pos - pos.shift(1)).abs().fillna(0.0)
    return (gross - turn * cost_rate).dropna()


# ──────────────────────────── strategies ────────────────────────
def sig_trend(df: pd.DataFrame, sym: str) -> pd.Series:
    f = ewmac(df[["close"]].rename(columns={"close": sym}))[sym]   # ~±20, target 10
    return (f / 10.0).clip(-1.5, 1.5)


def sig_breakout(df: pd.DataFrame, n_in: int = 20, n_out: int = 10) -> pd.Series:
    hi_in = df["high"].rolling(n_in).max().shift(1)
    lo_in = df["low"].rolling(n_in).min().shift(1)
    hi_out = df["high"].rolling(n_out).max().shift(1)
    lo_out = df["low"].rolling(n_out).min().shift(1)
    pos, state = [], 0
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    hin, lin, hout, lout = hi_in.values, lo_in.values, hi_out.values, lo_out.values
    for i in range(len(df)):
        if np.isnan(hin[i]):
            pos.append(0); continue
        if state == 0:
            if h[i] > hin[i]: state = 1
            elif l[i] < lin[i]: state = -1
        elif state == 1:
            if l[i] < lout[i]: state = -1 if (l[i] < lin[i]) else 0
        elif state == -1:
            if h[i] > hout[i]: state = 1 if (h[i] > hin[i]) else 0
        pos.append(state)
    return pd.Series(pos, index=df.index, dtype=float)


def sig_revert(df: pd.DataFrame, n: int = 20, enter: float = 1.5, exit_: float = 0.5) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=1)
    z = ((df["close"] - ma) / sd).shift(1)            # signal known at prior close
    pos, state = [], 0
    for zz in z.values:
        if np.isnan(zz):
            pos.append(0); continue
        if state == 0:
            if zz > enter: state = -1                 # overbought → short
            elif zz < -enter: state = 1               # oversold → long
        elif state == 1 and zz > -exit_:
            state = 0
        elif state == -1 and zz < exit_:
            state = 0
        pos.append(state)
    return pd.Series(pos, index=df.index, dtype=float)


def carry_diff(sym: str, idx) -> pd.Series | None:
    base, quote = CCY[sym]
    if base is None:
        return None
    r = load_rates()
    if base not in r.columns or quote not in r.columns:
        return None
    diff = (r[base] - r[quote]).reindex(
        r.index.union(idx)).sort_index().ffill().reindex(idx)
    return diff / 100.0          # percent → fraction (annualized rate differential)


# ──────────────────────────── metrics ───────────────────────────
def metrics(r: pd.Series, bpy: float, pos: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {}
    ann = np.sqrt(bpy)
    f = lambda x: float(x.mean() / x.std(ddof=1) * ann) if len(x) > 30 and x.std(ddof=1) > 0 else float("nan")
    rd, rc = r[r.index < SPLIT], r[r.index >= SPLIT]
    sh, shd, shc = f(r), f(rd), f(rc)
    lo, hi = block_bootstrap_sharpe(r.values, block=10, ppy=int(bpy))
    loc, hic = block_bootstrap_sharpe(rc.values, block=10, ppy=int(bpy)) if len(rc) >= 60 else (np.nan, np.nan)
    eq = (1 + r).cumprod(); dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    hit = (r > 0).mean() * 100
    cagr = float(eq.iloc[-1] ** (bpy / len(r)) - 1) * 100
    turn = float((pos - pos.shift(1)).abs().sum() / ((r.index[-1] - r.index[0]).days / 365.25)) if pos is not None else float("nan")
    go = (not np.isnan(shc) and shc >= 0.5 and not np.isnan(loc) and loc > 0 and not np.isnan(shd) and shd > 0)
    return dict(sh=sh, shd=shd, shc=shc, lo=lo, hi=hi, loc=loc, hic=hic, dd=dd,
                hit=hit, cagr=cagr, turn=turn, n=len(r), go=go)


def show(name: str, m: dict):
    if not m:
        print(f"    {name:22s}  insufficient data"); return
    g = "  ✅GO" if m["go"] else ""
    print(f"    {name:22s} Sh full={m['sh']:+.2f} disc={m['shd']:+.2f} "
          f"conf={m['shc']:+.2f}[{m['loc']:+.2f},{m['hic']:+.2f}]  "
          f"hit={m['hit']:.0f}% DD={m['dd']:.0f}% CAGR={m['cagr']:+.0f}% "
          f"turn={m['turn']:.0f}/yr n={m['n']}{g}", flush=True)


# ──────────────────────────── runner ────────────────────────────
def run(sym: str) -> dict:
    df = resample_d1(sym)
    ret = df["close"].pct_change()
    bpy = bars_per_year(df.index)
    cost = (df["spread"] * PIP[sym] + COMM_PIPS * PIP[sym]) / df["close"]   # one-way, fraction
    print(f"\n{'='*94}\n  {sym}  ({len(df):,} D1 bars {df.index[0].date()}→{df.index[-1].date()}, "
          f"~{bpy:.0f} bars/yr, real spread)\n{'='*94}")
    out = {}

    # 1. TREND (EWMAC)
    p = vol_scale(sig_trend(df, sym), ret, bpy)
    m = metrics(net_returns(p, ret, cost), bpy, p); out["TREND"] = m
    show("1.TREND ewmac", m)

    # 2. VOLMAN — predictability + does vol-targeting help vs fixed-size trend?
    rv = ret.abs()
    pred = rv.ewm(span=20, min_periods=10).std().shift(1)          # past-only vol forecast
    valid = pred.notna() & rv.shift(-1).notna()
    ic = float(pd.Series(pred[valid]).corr(rv.shift(-1)[valid], method="spearman"))
    raw_trend = sig_trend(df, sym)                                # fixed-size (no vol scaling)
    m_fixed = metrics(net_returns(raw_trend, ret, cost), bpy, raw_trend)
    print(f"    2.VOLMAN  vol-forecast IC(next |ret|)={ic:+.2f}  "
          f"|  trend vol-targeted Sh={m['sh']:+.2f}  vs fixed-size Sh={m_fixed.get('sh', float('nan')):+.2f}", flush=True)
    out["VOLMAN_ic"] = ic; out["VOLMAN_fixed"] = m_fixed

    # 3. BREAKOUT (Donchian 20/10)
    p = vol_scale(sig_breakout(df), ret, bpy)
    m = metrics(net_returns(p, ret, cost), bpy, p); out["BREAK"] = m
    show("3.BREAK donchian20/10", m)

    # 4. CARRY (rate differential) + trend-filtered
    cd = carry_diff(sym, df.index)
    if cd is not None:
        carry_acc = cd / bpy                                       # daily accrual
        sgn = np.sign(cd).fillna(0.0)
        p = vol_scale(sgn, ret, bpy)
        m = metrics(net_returns(p, ret, cost, extra_ret=carry_acc), bpy, p); out["CARRY"] = m
        show("4.CARRY rate-diff", m)
        # trend-filtered: only hold carry direction when EWMAC agrees
        tf = sgn.where(np.sign(sig_trend(df, sym)) == sgn, 0.0)
        p2 = vol_scale(tf, ret, bpy)
        m2 = metrics(net_returns(p2, ret, cost, extra_ret=carry_acc), bpy, p2); out["CARRY_tf"] = m2
        show("4b.CARRY+trendfilt", m2)
    else:
        print("    4.CARRY                no yield (gold) — carry sleeve N/A")

    # 5. REVERT (z-score mean reversion)
    p = vol_scale(sig_revert(df), ret, bpy)
    m = metrics(net_returns(p, ret, cost), bpy, p); out["REVERT"] = m
    show("5.REVERT zscore20", m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(PIP))
    args = ap.parse_args()
    syms = [args.symbol] if args.symbol else list(PIP)
    for s in syms:
        run(s)
    print("\nDone.")


if __name__ == "__main__":
    main()
