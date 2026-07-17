"""Discrete-engine confirmation of the FAST-CHAMPION short-term trend runner.

Runs the real lot/stop/%-risk trade engine (src.v5.xau_trend.run_trades) on
intraday bars, driving it with a FAST champion-recipe signal instead of the
slow H4 one (monkeypatched in — the engine reads module-level xau_signal).
Spread forced to the honest cent-account $0.34 round-trip. This is the
number a live fast bot would actually earn, versus the optimistic
continuous vectorized lab.

    python scripts/v5_xau_fast_trend_discrete.py --tf M30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.v5.xau_trend as xt
from src.cta.bootstrap import block_bootstrap_sharpe
from src.v5.artifacts import V5ArtifactWriter

EVAL_START = "2017-01-01"
CONF_RISK = {"low": 0.5, "med": 1.0, "high": 1.5}
DATA = {"M15": "data/XAUUSD_M15_long.csv",
        "M30": "data/XAUUSD_M30_long.csv",
        "H1": "data/XAUUSD_H1_long.csv"}
BPD = {"M15": 96, "M30": 48, "H1": 24}


def load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA[tf], parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def ewmac_fc(close, pairs, cap=2.0):
    ret = close.pct_change()
    pv = close * ret.ewm(span=36, min_periods=20).std()
    comb = None
    for f, s in pairs:
        raw = (close.ewm(span=f, min_periods=f).mean()
               - close.ewm(span=s, min_periods=s).mean()) / pv
        sc = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * sc).clip(-cap * 2, cap * 2)
        comb = fc if comb is None else comb + fc
    return (comb / len(pairs)).clip(-cap, cap)


def breakout_fc(close, windows, cap=2.0):
    comb = None
    for n in windows:
        hi = close.rolling(n, min_periods=n // 2).max()
        lo = close.rolling(n, min_periods=n // 2).min()
        mid = (hi + lo) / 2.0
        rng = (hi - lo).replace(0.0, np.nan)
        raw = ((close - mid) / rng * 4.0).ewm(span=max(2, n // 4)).mean()
        sc = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * sc).clip(-cap * 2, cap * 2)
        comb = fc if comb is None else comb + fc
    return (comb / len(windows)).clip(-cap, cap)


def make_fast_champion(tf: str, speed: str = "fast"):
    """Return a signal_fn(close)->forecast, long-only champion recipe at
    intraday speed. |forecast| ~ average strength, in [0, 2]."""
    bpd = BPD[tf]
    ep = {"vfast": [(8, 32), (16, 64)], "fast": [(16, 64), (32, 128)]}[speed]
    bwin_mult = {"vfast": 0.5, "fast": 1.0}[speed]
    bn = max(6, int(bpd * bwin_mult))

    def _norm(s):
        return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))

    def signal_fn(close: pd.Series) -> pd.Series:
        ew = ewmac_fc(close, ep)
        bk = breakout_fc(close, [bn, bn * 2])
        mx = np.maximum(ew.clip(lower=0.0), bk.clip(lower=0.0))
        return (0.5 * (_norm(mx.clip(lower=0) ** 1.5) * 0.8 + 0.15)
                + 0.5 * (_norm(bk.clip(lower=0) ** 1.5) * 0.8 + 0.15)).clip(0, 2)
    return signal_fn


def metrics(res: dict, equity0: float, label: str) -> dict:
    eq = res["equity"].loc[EVAL_START:].dropna()
    tr = res["trades"]
    if len(tr) == 0 or "pnl" not in tr.columns:
        return dict(label=label, sharpe=0.0, ci95=[0, 0], cagr_pct=0.0,
                    max_dd_pct=0.0, n_trades=0, trades_per_mo=0.0,
                    win_pct=0.0, pf=0.0, per_year={}, note="NO TRADES")
    tr = tr[tr["close_time"] >= EVAL_START]
    daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    sr = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    wins = tr[tr["pnl"] > 0]
    losses = tr[tr["pnl"] <= 0]
    dd = float((eq / eq.cummax() - 1).min() * 100)
    # per-year sharpe
    peryear = {}
    for y, g in daily.groupby(daily.index.year):
        peryear[int(y)] = round(float(g.mean() / g.std() * np.sqrt(252)), 2) \
            if g.std() > 0 else 0.0
    return dict(label=label, sharpe=round(sr, 3),
                ci95=[round(ci[0], 2), round(ci[1], 2)],
                cagr_pct=round(((eq.iloc[-1] / equity0) ** (1 / years) - 1) * 100, 2),
                max_dd_pct=round(dd, 1), n_trades=int(len(tr)),
                trades_per_mo=round(len(tr) / (years * 12), 1),
                win_pct=round(len(wins) / len(tr) * 100, 1) if len(tr) else 0.0,
                pf=round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 2)
                if len(losses) and losses["pnl"].sum() else float("inf"),
                per_year=peryear)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M30", choices=["M15", "M30", "H1"])
    ap.add_argument("--speed", default="fast", choices=["vfast", "fast"])
    ap.add_argument("--equity", type=float, default=3000.0)  # research notional
    ap.add_argument("--spread-usd", type=float, default=0.34)
    ap.add_argument("--slippage-usd", type=float, default=0.05)
    ap.add_argument("--risk-frac", type=float, default=0.01)
    ap.add_argument("--exit", default="trail", choices=["trail", "flip", "sltp"])
    ap.add_argument("--enter", type=float, default=0.5)
    ap.add_argument("--flip", type=float, default=1.0)
    args = ap.parse_args()

    df = load(args.tf)[["open", "high", "low", "close"]].copy()
    # force honest cent spread: engine does spread_px = spread(pips)*0.1;
    # $0.34 => 3.4 pips. slippage in pips = slip_usd/0.1.
    df["spread"] = args.spread_usd / 0.1

    orig = xt.xau_signal
    xt.xau_signal = make_fast_champion(args.tf, args.speed)
    try:
        res = xt.run_trades(
            df, equity0=args.equity, exit_mode=args.exit, flip_mode="confidence",
            params=dict(conf_risk_scale=CONF_RISK, risk_frac=args.risk_frac,
                        slippage_pips=args.slippage_usd / 0.1,
                        spread_cost_mult=1.0, entry_delay_bars=1,
                        enter_thresh=args.enter, flip_thresh=args.flip,
                        sl_atr=3.0, trail_atr=3.0))
    finally:
        xt.xau_signal = orig

    st = metrics(res, args.equity, f"fastchamp_{args.speed}_{args.tf}")
    print(f"\n=== DISCRETE fast-champion  {args.tf}/{args.speed}  "
          f"(spread ${args.spread_usd} + ${args.slippage_usd} slip, "
          f"risk {args.risk_frac:.1%}) ===")
    for k in ("sharpe", "ci95", "cagr_pct", "max_dd_pct", "n_trades",
              "trades_per_mo", "win_pct", "pf"):
        print(f"  {k:14s}: {st[k]}")
    print(f"  per-year SR   : {st['per_year']}")

    writer = V5ArtifactWriter()
    writer.write_run(
        run_id=f"fast-trend-discrete-{args.tf.lower()}-{args.speed}",
        settings=dict(strategy="fastchamp_discrete", tf=args.tf,
                      speed=args.speed, spread_usd=args.spread_usd,
                      slippage_usd=args.slippage_usd, risk_frac=args.risk_frac,
                      equity0=args.equity),
        trades=res["trades"].to_dict("records"),
        equity=res["equity"].loc[EVAL_START:].dropna(), stats=st,
        reconciliation={"status": "research_replay"})


if __name__ == "__main__":
    main()
