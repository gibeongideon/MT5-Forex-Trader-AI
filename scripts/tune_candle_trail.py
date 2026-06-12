"""
tune_candle_trail.py — Grid search for best candle_trail parameters.

Loads the model and generates signals ONCE, then sweeps all parameter
combinations using the fast simulate_trail() engine.  Reports top results
ranked by Sharpe, with MaxDD and Win rate shown for risk assessment.

Usage:
    conda run -n envmt5 python scripts/tune_candle_trail.py
    conda run -n envmt5 python scripts/tune_candle_trail.py --symbol EURUSD
    conda run -n envmt5 python scripts/tune_candle_trail.py --top 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.backtest_candle_trail import (
    SYMBOL_CFG, INITIAL_BALANCE, RISK_PCT, SPREAD_PIPS, COMMISSION_PIPS,
    THRESHOLD, _add_extra_features, simulate_base, simulate_trail,
)
from src.pipeline import PredictorPipeline

# ── Parameter grid ─────────────────────────────────────────────────────────────
GRID = {
    "trail_activation_pips": [8, 10, 12, 15, 20],
    "trail_pips_behind":     [5, 7, 10, 12],
    "max_bars_low":          [1],            # keep locked at 1 (same as base)
    "max_bars_med":          [2, 3],
    "max_bars_high":         [3, 4, 5, 6],
}


def _sharpe(eq: pd.Series, bars_per_year: float) -> float:
    ret = eq.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 2:
        return float("nan")
    return float(ret.mean() / ret.std() * np.sqrt(bars_per_year))


def tune_symbol(symbol: str, top_n: int) -> None:
    cfg = dict(SYMBOL_CFG[symbol])

    model_dir = Path(cfg["model_dir"])
    if not model_dir.exists():
        print(f"\n[{symbol}] Model not found at {model_dir}")
        return

    pipe = PredictorPipeline.from_config()
    pipe.load(str(model_dir))

    meta_path = model_dir / "pair_meta.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        cfg["sl_pips"] = float(saved.get("sl_pips", cfg["sl_pips"]))
        cfg["tp_pips"] = float(saved.get("tp_pips", cfg["tp_pips"]))

    df_raw = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.sort_index()

    span_yrs      = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bars_per_year = len(df_raw) / span_yrs

    print(f"\n{'═'*72}")
    print(f"  CANDLE TRAIL PARAMETER SEARCH — {symbol}")
    print(f"  {len(df_raw):,} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  SL={cfg['sl_pips']:.0f}p  TP={cfg['tp_pips']:.0f}p  "
          f"threshold={THRESHOLD:.0%}")
    print(f"{'═'*72}")

    print("  Building features + generating signals...", end=" ", flush=True)
    try:
        X_base, _ = pipe._fp.build(df_raw, fit=False)
        if pipe._enc is not None:
            latent = pipe._enc.transform(df_raw)
            shared = X_base.index.intersection(latent.index)
            X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
        else:
            X = X_base
        X = _add_extra_features(df_raw, X)
        for c in pipe._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[pipe._feature_cols]
        signals = pipe.predict_batch(X)
        prices  = df_raw.reindex(signals.index)
    except Exception as e:
        print(f"FAILED: {e}")
        return
    print(f"done  ({len(signals):,} bars)")

    # Base result for comparison
    r_base   = simulate_base(signals, prices, cfg["sl_pips"], cfg["tp_pips"], cfg["pip_size"])
    s_base   = _sharpe(r_base["equity"], bars_per_year)

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    print(f"  Sweeping {total} parameter combinations...")

    results = []
    for idx, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        # Skip if trail_pips_behind >= trail_activation_pips (can't trail)
        if params["trail_pips_behind"] >= params["trail_activation_pips"]:
            continue

        r = simulate_trail(
            signals, prices,
            cfg["sl_pips"], cfg["tp_pips"], cfg["pip_size"],
            trail_activation_pips = params["trail_activation_pips"],
            trail_pips_behind     = params["trail_pips_behind"],
            max_bars_low          = params["max_bars_low"],
            max_bars_med          = params["max_bars_med"],
            max_bars_high         = params["max_bars_high"],
        )
        s = _sharpe(r["equity"], bars_per_year)
        if not np.isnan(s):
            results.append({
                **params,
                "sharpe":    s,
                "win_rate":  r["win_rate"] * 100,
                "max_dd":    r["max_dd_pct"],
                "net_pnl":   r["net_pnl_pct"],
                "n_trades":  r["n_trades"],
                "avg_win":   r["avg_win_pips"],
            })

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{total}...", end="\r", flush=True)

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    W = 100
    print(f"\n  ── TOP {top_n} BY SHARPE (base candle_predictor: {s_base:+.3f}) {'─'*20}")
    print(f"  {'act':>4}  {'beh':>4}  {'lo':>3}  {'me':>3}  {'hi':>3}  "
          f"{'Sharpe':>8}  {'Win%':>6}  {'MaxDD%':>7}  {'Trades':>7}  "
          f"{'AvgWin':>7}")
    print(f"  {'─'*W}")

    for r in results[:top_n]:
        marker = " ★" if r["sharpe"] > s_base else ""
        print(
            f"  {r['trail_activation_pips']:>4.0f}  "
            f"{r['trail_pips_behind']:>4.0f}  "
            f"{r['max_bars_low']:>3d}  "
            f"{r['max_bars_med']:>3d}  "
            f"{r['max_bars_high']:>3d}  "
            f"{r['sharpe']:>+8.3f}  "
            f"{r['win_rate']:>5.1f}%  "
            f"{r['max_dd']:>6.1f}%  "
            f"{r['n_trades']:>7,}  "
            f"{r['avg_win']:>6.1f}p"
            f"{marker}"
        )

    print(f"\n  BASE candle_predictor:  Sharpe={s_base:+.3f}  "
          f"Win={r_base['win_rate']*100:.1f}%  "
          f"MaxDD={r_base['max_dd_pct']:.1f}%  "
          f"Trades={r_base['n_trades']:,}")

    # Best by different criteria
    best_sharpe = results[0]
    best_dd     = min(results, key=lambda x: x["max_dd"])
    # Best Sharpe among those with DD <= base*0.6
    low_dd_candidates = [r for r in results if r["max_dd"] <= r_base["max_dd_pct"] * 0.6]
    best_balanced = low_dd_candidates[0] if low_dd_candidates else best_sharpe

    print(f"\n  ── RECOMMENDATIONS ───────────────────────────────────────────────")
    for label, r in [
        ("Best Sharpe",     best_sharpe),
        ("Lowest MaxDD",    best_dd),
        ("Best balanced (Sharpe + DD≤60% of base)", best_balanced),
    ]:
        cmd = (f"--trail-activation-pips {r['trail_activation_pips']:.0f} "
               f"--trail-pips-behind {r['trail_pips_behind']:.0f} "
               f"--max-bars-low {r['max_bars_low']} "
               f"--max-bars-med {r['max_bars_med']} "
               f"--max-bars-high {r['max_bars_high']}")
        print(f"\n  {label}:")
        print(f"    Sharpe={r['sharpe']:+.3f}  Win={r['win_rate']:.1f}%  "
              f"MaxDD={r['max_dd']:.1f}%  Trades={r['n_trades']:,}")
        print(f"    {cmd}")


def main() -> None:
    p = argparse.ArgumentParser(description="Grid search candle_trail parameters")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    p.add_argument("--top",    type=int, default=15, help="How many results to show (default 15)")
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    for sym in symbols:
        tune_symbol(sym, args.top)
    print()


if __name__ == "__main__":
    main()
