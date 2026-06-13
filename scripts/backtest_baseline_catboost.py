"""
backtest_baseline_catboost.py — Zero-leakage CatBoost baseline (no encoder).

PURPOSE
───────
Benchmark a simple CatBoost model on pure engineered features (technical
indicators + session flags + MTF EMAs) with NO encoder and NO latent dims.

Because there is no ML component in feature construction, this walk-forward is
100% leakage-free by construction. If this baseline shows positive Sharpe, the
signal exists in the features alone. If it is also negative, the issue is the
label definition or the assumption that M15 directional signal exists at all.

COMPARISON TABLE (filled in after each run):
  Original WF (leaky encoder)      : EURUSD +7.118  / USDJPY +14.414
  Per-fold fresh encoder           : EURUSD −10.580 / USDJPY −15.479
  Pre-train + fine-tune encoder    : EURUSD −11.314 / USDJPY  −8.719
  Baseline CatBoost (no encoder)   : EURUSD ???     / USDJPY  ???

Usage:
    conda run -n envmt5 python scripts/backtest_baseline_catboost.py
    conda run -n envmt5 python scripts/backtest_baseline_catboost.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_baseline_catboost.py --folds 3
    conda run -n envmt5 python scripts/backtest_baseline_catboost.py --threshold 0.50
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline, PipelineConfig
from src.models.catboost_model import CatBoostModel
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG

# ── Config (same as backtest_clean_wf_finetune.py for fair comparison) ─────────
TRAIN_DAYS      = 120
TEST_DAYS       = 15
THRESHOLD       = 0.60
SL_PIPS         = 10.0
TP_PIPS         = 30.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_sliding_folds(index, train_days, test_days, max_folds=None):
    start     = index[0]
    end       = index[-1]
    td        = pd.Timedelta(days=train_days)
    te        = pd.Timedelta(days=test_days)
    folds     = []
    fold_idx  = 0
    train_end = start + td
    while train_end + te <= end + pd.Timedelta(days=1):
        test_end    = min(train_end + te, end)
        train_start = train_end - td
        folds.append((fold_idx, train_start, train_end, test_end))
        fold_idx  += 1
        train_end += te
        if max_folds and fold_idx >= max_folds:
            break
    return folds


def _simulate_trades(proba, classes, index, prices, pip_size, threshold):
    cm = {c: i for i, c in enumerate(classes)}
    pb = proba[:, cm.get(1,  cm.get("buy",  0))]
    ps = proba[:, cm.get(-1, cm.get("sell", 2))]
    trades = []
    for i, ts in enumerate(index):
        if pb[i] >= threshold and pb[i] > ps[i]:
            direction = "buy";  conf = float(pb[i])
        elif ps[i] >= threshold and ps[i] > pb[i]:
            direction = "sell"; conf = float(ps[i])
        else:
            continue
        nxt = prices.index[prices.index > ts]
        if not len(nxt):
            continue
        nt    = nxt[0]
        entry = prices.loc[ts,  "close"]
        h_nx  = prices.loc[nt,  "high"]
        l_nx  = prices.loc[nt,  "low"]
        c_nx  = prices.loc[nt,  "close"]
        sl    = SL_PIPS * pip_size
        tp    = TP_PIPS * pip_size
        if direction == "buy":
            if l_nx <= entry - sl:   pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif h_nx >= entry + tp: pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else: pips = (c_nx - entry) / pip_size - SPREAD_PIPS - COMM_PIPS
        else:
            if h_nx >= entry + sl:   pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif l_nx <= entry - tp: pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else: pips = (entry - c_nx) / pip_size - SPREAD_PIPS - COMM_PIPS
        trades.append({"ts": ts, "dir": direction, "conf": conf, "pips": pips})
    return trades


def _annualized_sharpe(trades):
    if len(trades) < 5:
        return float("nan")
    pnl     = [t["pips"] for t in trades]
    returns = pd.Series(pnl) / SL_PIPS * RISK_PCT
    span_days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
    tpy = len(trades) / span_days * 365.25 if span_days > 0 else len(trades)
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r < 1e-9:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(tpy))


def _equity_stats(trades):
    balance = INITIAL_BAL
    eq = [balance]
    for t in trades:
        balance += balance * RISK_PCT * (t["pips"] / SL_PIPS)
        eq.append(balance)
    eq_s = pd.Series(eq)
    dd   = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret  = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100
    return dd, ret


# ── Per-symbol WF ──────────────────────────────────────────────────────────────

def run_symbol(symbol: str, threshold: float, max_folds: int = None) -> dict:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])

    folds = _get_sliding_folds(df_raw.index, TRAIN_DAYS, TEST_DAYS,
                                max_folds=max_folds)

    print(f"\n{'='*72}")
    print(f"  BASELINE CatBoost (no encoder) — {symbol}")
    print(f"  {len(df_raw):,} bars  "
          f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  WF: sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  {len(folds)} folds  |  "
          f"threshold={threshold}  |  SL={SL_PIPS}p TP={TP_PIPS}p")
    print(f"  Features: base indicators + session flags + MTF EMAs (NO encoder)")
    print(f"{'='*72}\n")

    header = (f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>21}  "
              f"{'Trd':>4}  {'Win%':>5}  {'Sharpe':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_trades = []
    t0 = time.time()

    for fold_idx, train_start, train_end, test_end in folds:
        df_train = df_raw[(df_raw.index >= train_start) & (df_raw.index < train_end)].copy()
        df_fold  = df_raw[(df_raw.index >= train_start) & (df_raw.index < test_end)].copy()
        df_test  = df_raw[(df_raw.index >= train_end)   & (df_raw.index < test_end)].copy()

        if len(df_train) < 500 or len(df_test) < 30:
            continue

        # Build features with encoder disabled — pure technical indicators, zero leakage
        pipe = PredictorPipeline(PipelineConfig(
            label_horizon    = LABEL_HORIZON,
            label_threshold  = LABEL_THRESHOLD,
            encoder_enabled  = False,
        ))
        X_train, y_train = pipe.build_features(df_train, train_frac=1.0)
        X_train = _add_extra_features(df_train, X_train)
        feature_cols = list(X_train.columns)

        if len(X_train) < 50:
            continue

        # Transform full fold window (train+test) for OOS prediction
        X_base_full, _ = pipe._fp.build(df_fold, fit=False)
        X_full = _add_extra_features(df_fold, X_base_full)
        for c in feature_cols:
            if c not in X_full.columns:
                X_full[c] = 0.0
        X_full = X_full[feature_cols]
        X_test = X_full[(X_full.index >= train_end) & (X_full.index < test_end)]

        if len(X_test) < 10:
            continue

        model = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        model.train(X_train, y_train)
        proba   = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes     = list(model._classes)
        prices_test = df_raw.reindex(X_test.index)
        fold_trades = _simulate_trades(proba, classes, X_test.index,
                                       prices_test, pip_size, threshold)

        n_t   = len(fold_trades)
        win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
        f_sh  = _annualized_sharpe(fold_trades) if n_t >= 5 else float("nan")
        sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"

        print(f"  {fold_idx:>4}  "
              f"{str(train_start.date()):>12} → {str(train_end.date()):<12}  "
              f"{str(train_end.date()):>10} → {str(test_end.date()):<10}  "
              f"{n_t:>4}  {win_r:>4.0%}  {sh_s:>7}",
              flush=True)

        all_trades.extend(fold_trades)

    elapsed = (time.time() - t0) / 60

    print(f"\n  Elapsed: {elapsed:.1f} min\n")
    print(f"  {'─'*68}")
    print(f"  FINAL RESULT  [{symbol}]  Baseline CatBoost — no encoder")
    print(f"  {'─'*68}")

    if not all_trades:
        print("  No trades generated.")
        return {"symbol": symbol, "sharpe": float("nan"), "n_trades": 0,
                "win_rate": 0.0, "max_dd": float("nan")}

    n_total = len(all_trades)
    wins    = sum(1 for t in all_trades if t["pips"] > 0)
    wr      = wins / n_total
    dd, ret = _equity_stats(all_trades)
    sh      = _annualized_sharpe(all_trades)
    sh_s    = f"{sh:+.3f}" if not np.isnan(sh) else "n/a"

    print(f"  Sharpe (annualized) : {sh_s}")
    print(f"  Win rate            : {wr:.1%}  ({wins}W / {n_total-wins}L)")
    print(f"  Max drawdown        : {dd:.1f}%")
    print(f"  Net return          : {ret:+.1f}%")
    print(f"  Total trades        : {n_total}")
    print(f"\n  Comparison (all leakage-free methods, {symbol}):")
    print(f"    Original WF (leaky encoder)      : "
          f"EURUSD +7.118 / USDJPY +14.414")
    print(f"    Per-fold fresh encoder           : "
          f"EURUSD −10.580 / USDJPY −15.479")
    print(f"    Pre-train + fine-tune encoder    : "
          f"EURUSD −11.314 / USDJPY  −8.719")
    print(f"    Baseline CatBoost (no encoder)   : {sh_s}  ← this run")
    print(f"  {'─'*68}\n")

    return {"symbol": symbol, "sharpe": sh, "n_trades": n_total,
            "win_rate": wr, "max_dd": dd, "return_pct": ret}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-leakage CatBoost baseline — no encoder"
    )
    parser.add_argument("--symbol",    default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Confidence threshold (default 0.60, try 0.50 for more trades)")
    parser.add_argument("--folds",     type=int,   default=None,
                        help="Limit to first N folds (smoke test)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  ZERO-LEAKAGE BASELINE — CatBoost, no encoder")
    print(f"  Sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  threshold={args.threshold}")
    print(f"  Features: {31} base indicators + session flags + MTF EMAs (52 total)")
    if args.folds:
        print(f"  NOTE: limited to first {args.folds} folds (smoke test)")
    print(f"{'='*72}")

    results = []
    for sym in symbols:
        r = run_symbol(sym, threshold=args.threshold, max_folds=args.folds)
        results.append(r)

    if len(results) == 2:
        print(f"\n{'='*72}")
        print(f"  SUMMARY")
        print(f"  {'Symbol':>8}  {'Sharpe':>8}  {'Win%':>6}  {'MaxDD':>7}  "
              f"{'Trades':>7}  {'Return':>8}")
        print(f"  {'-'*58}")
        for r in results:
            sh_s = f"{r['sharpe']:+.3f}" if not np.isnan(r["sharpe"]) else "   n/a"
            print(f"  {r['symbol']:>8}  {sh_s:>8}  {r['win_rate']:>5.1%}  "
                  f"{r['max_dd']:>6.1f}%  {r['n_trades']:>7}  "
                  f"{r.get('return_pct', float('nan')):>+7.1f}%")
        print(f"{'='*72}\n")

    print("Done.")


if __name__ == "__main__":
    main()
