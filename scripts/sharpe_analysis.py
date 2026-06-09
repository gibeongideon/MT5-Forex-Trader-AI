"""
sharpe_analysis.py — Actual annualized Sharpe breakdown: overall + per year

Modes analysed: always | hedge_loss | partial_close
Symbols: EURUSD, USDJPY

Sharpe formula:
  S = mean(r) / std(r) * sqrt(bars_per_year)
  r = per-bar equity percentage returns
  bars_per_year = empirical count from raw data (keeps yearly Sharpes comparable)
  Risk-free rate = 0  (standard for Forex)

Usage:
    conda run -n envmt5 python scripts/sharpe_analysis.py
    conda run -n envmt5 python scripts/sharpe_analysis.py --symbol EURUSD
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_flip_modes import (
    simulate_mode, SYMBOL_PARAMS, TRAIL_PIPS, HEDGE_RATIO, ZONE_PIPS,
    INITIAL_BALANCE,
)
from src.pipeline import PredictorPipeline

TARGET_MODES = ["always", "hedge_loss", "partial_close"]


# ── Sharpe helpers ─────────────────────────────────────────────────────────────

def annualized_sharpe(equity: pd.Series, bars_per_year: float) -> float:
    """Per-bar equity returns → annualized Sharpe (rf=0)."""
    r = equity.pct_change().dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def year_stats(equity: pd.Series, trades: list, year: int,
               bars_per_year: float) -> dict:
    """
    Slice equity to a single calendar year and compute:
      sharpe, max_dd, win_rate, n_trades
    """
    mask = equity.index.year == year
    yr_eq = equity[mask]

    if len(yr_eq) < 10:
        return dict(sharpe=float("nan"), max_dd=float("nan"),
                    win_rate=float("nan"), n_trades=0)

    s = annualized_sharpe(yr_eq, bars_per_year)

    # drawdown within the year
    peak   = yr_eq.cummax()
    dd_pct = ((peak - yr_eq) / peak * 100)
    max_dd = float(dd_pct.max())

    # trades whose entry bar falls in this year
    yr_trades = [t for t in trades if yr_eq.index[0] <= equity.index[t.entry_bar] <= yr_eq.index[-1]]
    wins     = sum(1 for t in yr_trades if t.pnl_pips > 0)
    n        = len(yr_trades)

    return dict(
        sharpe   = s,
        max_dd   = max_dd,
        win_rate = wins / n if n else float("nan"),
        n_trades = n,
    )


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> None:
    params   = SYMBOL_PARAMS[symbol]
    pip_size = params["pip_size"]
    sl_pips  = params["sl_pips"]
    tp_pips  = params["tp_pips"]

    # Load model
    pipe = PredictorPipeline.from_config()
    pipe.load(params["model_dir"])

    # Raw OHLCV
    df_raw = pd.read_csv(params["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]

    # Empirical bars-per-year from this dataset
    span_years   = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bars_per_year = len(df_raw) / span_years
    years        = sorted(df_raw.index.year.unique().tolist())

    print(f"\n{'=' * 70}")
    print(f"  {symbol}  |  {len(df_raw):,} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  bars/year ≈ {bars_per_year:,.0f}  |  years: {years}")
    print(f"{'=' * 70}")

    # Build features once (shared across all modes)
    X_base, _ = pipe._fp.build(df_raw, fit=False)
    if pipe._enc is not None:
        latent = pipe._enc.transform(df_raw)
        shared = X_base.index.intersection(latent.index)
        X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
    else:
        X = X_base
    for c in pipe._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X       = X[pipe._feature_cols]
    signals = pipe.predict_batch(X)
    prices  = df_raw.reindex(signals.index)

    # Run 3 modes
    results = {}
    for mode in TARGET_MODES:
        r = simulate_mode(
            signals, prices, mode, sl_pips, tp_pips, pip_size,
            trail_pips=TRAIL_PIPS, hedge_ratio=HEDGE_RATIO, zone_pips=ZONE_PIPS,
        )
        results[mode] = r

    # ── Overall Sharpe ────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  OVERALL SHARPE  (full dataset, annualized, rf=0)")
    print(f"{'─'*70}")
    hdr = f"  {'Mode':<20} {'Sharpe':>8}  {'Win%':>7}  {'Trades':>8}  {'MaxDD':>8}  {'NetPnL':>9}"
    print(hdr)
    print(f"  {'-'*68}")
    for mode, r in results.items():
        eq   = r["equity"]
        s    = annualized_sharpe(eq, bars_per_year)
        win  = r["win_rate"]
        nt   = r["n_trades"]
        dd   = r["max_dd_pct"]
        pnl  = r["net_pnl_pct"]
        print(f"  {mode:<20} {s:>8.3f}  {win:>6.1%}  {nt:>8,d}  {dd:>7.1f}%  {pnl:>+8.1f}%")

    # ── Per-year Sharpe ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  PER-YEAR BREAKDOWN  (annualized Sharpe | win% | n_trades | maxDD%)")
    print(f"{'─'*70}")

    for mode, r in results.items():
        eq     = r["equity"]
        trades = r["trades"]
        print(f"\n  {mode}")
        col_w = 16
        yr_hdr = f"  {'Year':<8}" + "".join(f"{'Sharpe':>{col_w}}" for _ in years)
        print(yr_hdr)
        yr_hdr2 = f"  {'':8}" + "".join(
            f"{'win%/n/dd':>{col_w}}" for _ in years
        )
        print(yr_hdr2)
        print(f"  {'-'*68}")

        sharpe_row  = f"  {'Sharpe':<8}"
        detail_rows = {y: "" for y in years}

        for year in years:
            ys = year_stats(eq, trades, year, bars_per_year)
            s  = ys["sharpe"]
            s_str = f"{s:.3f}" if not np.isnan(s) else "  n/a"
            sharpe_row += f"{s_str:>{col_w}}"
            if not np.isnan(ys["win_rate"]):
                d = (f"{ys['win_rate']:.0%} / "
                     f"{ys['n_trades']:,d} / "
                     f"{ys['max_dd']:.1f}%")
            else:
                d = "n/a"
            detail_rows[year] = d

        print(sharpe_row)
        det_line = f"  {'detail':<8}"
        for year in years:
            det_line += f"{detail_rows[year]:>{col_w}}"
        print(det_line)

    # ── Side-by-side year table ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  SHARPE SUMMARY TABLE — {symbol}")
    print(f"{'─'*70}")
    col = 10
    header = f"  {'Mode':<22}" + f"{'Overall':>{col}}" + "".join(f"{y:>{col}}" for y in years)
    print(header)
    print(f"  {'-'*68}")
    for mode, r in results.items():
        eq     = r["equity"]
        trades = r["trades"]
        overall = annualized_sharpe(eq, bars_per_year)
        row = f"  {mode:<22}{overall:>{col}.3f}"
        for year in years:
            ys = year_stats(eq, trades, year, bars_per_year)
            s  = ys["sharpe"]
            row += f"{s:>{col}.3f}" if not np.isnan(s) else f"{'n/a':>{col}}"
        print(row)

    # ── Yearly win rate table ─────────────────────────────────────────────────
    print(f"\n  Win% by year")
    print(f"  {'-'*68}")
    header2 = f"  {'Mode':<22}" + f"{'Overall':>{col}}" + "".join(f"{y:>{col}}" for y in years)
    print(header2)
    for mode, r in results.items():
        eq     = r["equity"]
        trades = r["trades"]
        overall_win = r["win_rate"]
        row = f"  {mode:<22}{overall_win:>{col}.1%}"
        for year in years:
            ys = year_stats(eq, trades, year, bars_per_year)
            w  = ys["win_rate"]
            row += f"{w:>{col}.1%}" if not np.isnan(w) else f"{'n/a':>{col}}"
        print(row)

    # ── Yearly MaxDD table ────────────────────────────────────────────────────
    print(f"\n  MaxDD% by year")
    print(f"  {'-'*68}")
    print(header2)
    for mode, r in results.items():
        eq     = r["equity"]
        trades = r["trades"]
        overall_dd = r["max_dd_pct"]
        row = f"  {mode:<22}{overall_dd:>{col}.1f}%"
        for year in years:
            ys = year_stats(eq, trades, year, bars_per_year)
            d  = ys["max_dd"]
            row += f"{d:>{col}.1f}%" if not np.isnan(d) else f"{'n/a':>{col}}"
        print(row)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Sharpe ratio analysis by mode and year")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_PARAMS.keys()),
                   help="Single symbol (default: all)")
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_PARAMS.keys())
    for sym in symbols:
        run_symbol(sym)
    print()


if __name__ == "__main__":
    main()
