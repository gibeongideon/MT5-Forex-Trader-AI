"""
oos_sharpe.py — Out-of-sample Sharpe ratio for always | hedge_loss | partial_close

Two walk-forward splits on each symbol:
  Split A: train 2024       → test 2025-2026
  Split B: train 2024-2025  → test 2026

Steps per split:
  1. Fit scaler + encoder on train bars only (via build_features train_frac)
  2. Train XGBoost model on train labels only
  3. Generate signals on OOS test bars (model never saw them)
  4. Simulate 3 modes on test bars only (fresh balance per split)
  5. Compute real annualised Sharpe on OOS equity curve

Usage:
    conda run -n envmt5 python scripts/oos_sharpe.py
    conda run -n envmt5 python scripts/oos_sharpe.py --symbol EURUSD
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline
from scripts.backtest_flip_modes import (
    simulate_mode, SYMBOL_PARAMS, TRAIL_PIPS, HEDGE_RATIO, ZONE_PIPS,
    INITIAL_BALANCE,
)

TARGET_MODES = ["always", "hedge_loss", "partial_close"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def annualized_sharpe(equity: pd.Series, bars_per_year: float) -> float:
    r = equity.pct_change().dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def max_dd(equity: pd.Series) -> float:
    pk = equity.cummax()
    return float(((pk - equity) / pk * 100).max())


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _build_oos_features(pipe: PredictorPipeline,
                        df_raw: pd.DataFrame,
                        train_mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fit scaler + encoder on train rows only; transform the full dataset;
    return (X_full, y_full) with all columns aligned.
    """
    train_frac = train_mask.sum() / len(df_raw)

    # build_features fits scaler+encoder on first train_frac rows,
    # then transforms the entire df_raw with those fitted artifacts.
    X, y = pipe.build_features(df_raw, train_frac=train_frac)
    return X, y


# ── One split ──────────────────────────────────────────────────────────────────

def run_split(symbol: str, split_label: str,
              train_years: list[int], test_years: list[int],
              df_raw: pd.DataFrame, bars_per_year: float,
              cfg_overrides: dict) -> None:

    params   = SYMBOL_PARAMS[symbol]
    pip_size = params["pip_size"]
    sl_pips  = params["sl_pips"]
    tp_pips  = params["tp_pips"]

    train_mask = df_raw.index.year.isin(train_years)
    test_mask  = df_raw.index.year.isin(test_years)

    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    train_start = df_raw.index[train_mask][0].date()
    train_end   = df_raw.index[train_mask][-1].date()
    test_start  = df_raw.index[test_mask][0].date()
    test_end    = df_raw.index[test_mask][-1].date()

    print(f"\n  [{split_label}]  train {train_start}→{train_end} ({n_train:,} bars)"
          f"   test {test_start}→{test_end} ({n_test:,} bars)")

    # ── 1. Build features (scaler+encoder fit on train only) ──────────────────
    pipe = PredictorPipeline.from_config()
    for k, v in cfg_overrides.items():
        setattr(pipe.cfg, k, v)

    print("     fitting scaler+encoder on train...", end=" ", flush=True)
    X, y = _build_oos_features(pipe, df_raw, df_raw.index.year.isin(train_years))
    print("done")

    # ── 2. Split train / test features ────────────────────────────────────────
    train_idx = X.index[X.index.year.isin(train_years)]
    test_idx  = X.index[X.index.year.isin(test_years)]

    X_train = X.loc[X.index.isin(train_idx)]
    y_train = y.loc[y.index.isin(train_idx)]
    X_test  = X.loc[X.index.isin(test_idx)]

    # ── 3. Train model on train set only ──────────────────────────────────────
    print(f"     training XGBoost on {len(X_train):,} train rows...", end=" ", flush=True)
    pipe.fit_full(X_train, y_train)
    print("done")

    # ── 4. Generate OOS signals ───────────────────────────────────────────────
    print(f"     predicting on {len(X_test):,} test rows...", end=" ", flush=True)
    signals = pipe.predict_batch(X_test)
    prices  = df_raw.reindex(signals.index)
    print("done")

    n_buy  = (signals["signal"] == "buy").sum()
    n_sell = (signals["signal"] == "sell").sum()
    n_hold = (signals["signal"] == "hold").sum()
    print(f"     OOS signals: buy={n_buy:,}  sell={n_sell:,}  hold={n_hold:,}")

    # ── 5. Simulate + Sharpe ──────────────────────────────────────────────────
    print()
    col = 12
    hdr = (f"     {'Mode':<18} {'Sharpe':>{col}} {'Win%':>{col}} "
           f"{'Trades':>{col}} {'MaxDD':>{col}} {'NetPnL':>{col}}")
    print(hdr)
    print(f"     {'-'*74}")

    for mode in TARGET_MODES:
        r  = simulate_mode(signals, prices, mode, sl_pips, tp_pips, pip_size,
                           trail_pips=TRAIL_PIPS, hedge_ratio=HEDGE_RATIO,
                           zone_pips=ZONE_PIPS)
        eq = r["equity"]
        s  = annualized_sharpe(eq, bars_per_year)
        dd = max_dd(eq)
        s_str = f"{s:.3f}" if not np.isnan(s) else "n/a"
        print(
            f"     {mode:<18} {s_str:>{col}} "
            f"{r['win_rate']:>{col}.1%} "
            f"{r['n_trades']:>{col},d} "
            f"{dd:>{col}.1f}% "
            f"{r['net_pnl_pct']:>+{col}.1f}%"
        )


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> None:
    params = SYMBOL_PARAMS[symbol]

    df_raw = _load_raw(params["data_path"])
    years  = sorted(df_raw.index.year.unique().tolist())

    span_years    = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bars_per_year = len(df_raw) / span_years

    print(f"\n{'='*72}")
    print(f"  {symbol}  —  OUT-OF-SAMPLE SHARPE")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  bars/year ≈ {bars_per_year:,.0f}  |  available years: {years}")
    print(f"{'='*72}")
    print(f"  NOTE: Sharpe computed on OOS test bars only — model never saw them")

    # Override config to match deployed champion settings for this symbol
    # (from_config already has xgboost + fractal_enabled=true from config.yaml)
    cfg_overrides = {
        "artifacts_dir": params["model_dir"],
        "data_path":     params["data_path"],
    }

    # Split A: train 2024 → test 2025-2026
    train_a = [y for y in years if y == 2024]
    test_a  = [y for y in years if y > 2024]
    if train_a and test_a:
        run_split(symbol, "Split A", train_a, test_a, df_raw,
                  bars_per_year, cfg_overrides)
    else:
        print("  [Split A] Not enough years — skipping")

    # Split B: train 2024-2025 → test 2026
    train_b = [y for y in years if y < 2026]
    test_b  = [y for y in years if y == 2026]
    if train_b and test_b:
        run_split(symbol, "Split B", train_b, test_b, df_raw,
                  bars_per_year, cfg_overrides)
    else:
        print("  [Split B] Not enough years — skipping")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  INTERPRETATION GUIDE")
    print(f"  {'─'*70}")
    print(f"  Sharpe > 1.5  → strong out-of-sample edge")
    print(f"  Sharpe 0.5-1.5 → real but modest edge")
    print(f"  Sharpe < 0.5  → weak; watch live closely")
    print(f"  Sharpe < 0   → strategy is losing OOS")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Out-of-sample Sharpe: train/test date splits")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_PARAMS.keys()),
                   help="Single symbol (default: all)")
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_PARAMS.keys())
    for sym in symbols:
        run_symbol(sym)
    print()


if __name__ == "__main__":
    main()
