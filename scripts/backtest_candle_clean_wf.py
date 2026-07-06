"""
backtest_candle_clean_wf.py — Fully leakage-free candle predictor walk-forward backtest.

WHAT THIS FIXES vs the original train_candle_model.py WF:
  - Original: encoder trained once on first 80% of data, shared across ALL folds.
    Early fold OOS windows fall inside the encoder's training set → encoder leak.
  - This script: for each fold, fits a FRESH encoder ONLY on that fold's training
    window. The encoder has never seen that fold's OOS bars.

Result: the Sharpe reported here is the honest, clean number.

Walk-forward setup (mirrors original):
  - Sliding window: 120d train / 60d test
  - CatBoost: 300 iterations, depth=6, lr=0.05, l2=3.0, subsample=0.8
  - Threshold: 0.60 confidence to trade
  - SL=10p, TP=30p, spread=1.0p, commission=0.5p
  - Risk: 1% per trade

Usage:
    conda run -n envmt5 python scripts/backtest_candle_clean_wf.py
    conda run -n envmt5 python scripts/backtest_candle_clean_wf.py --symbol EURUSD
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
from scripts.train_candle_model import SYMBOL_CFG
from scripts.audit_live_champions import add_extra_features as _add_extra_fixed
from src.models.catboost_model import CatBoostModel


def _add_extra_features(df_raw, X):
    """Leak-free extras: fixed MTF EMAs (no resample lookahead)."""
    return _add_extra_fixed(df_raw, X, fix_lookahead=True)

# ── Config (mirrors original train_candle_model.py) ───────────────────────────

TRAIN_DAYS  = 120
TEST_DAYS   = 60
THRESHOLD   = 0.60
SL_PIPS     = 10.0
TP_PIPS     = 30.0
SPREAD_PIPS = 1.0
COMM_PIPS   = 0.5
RISK_PCT    = 0.01
INITIAL_BAL = 10_000.0
LABEL_HORIZON    = 1
LABEL_THRESHOLD  = 0.0005


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_sliding_folds(index: pd.DatetimeIndex, train_days: int, test_days: int):
    """
    Generate (fold_idx, train_start, train_end, test_end) for sliding window WF.
    First fold starts as soon as there is enough train data.
    """
    start = index[0]
    end   = index[-1]
    td    = pd.Timedelta(days=train_days)
    te    = pd.Timedelta(days=test_days)

    folds = []
    fold_idx   = 0
    train_end  = start + td
    while train_end + te <= end + pd.Timedelta(days=1):
        test_end    = min(train_end + te, end)
        train_start = train_end - td
        folds.append((fold_idx, train_start, train_end, test_end))
        fold_idx  += 1
        train_end += te

    return folds


def _simulate_trades(
    proba:    np.ndarray,
    classes:  list,
    index:    pd.DatetimeIndex,
    prices:   pd.DataFrame,
    pip_size: float,
) -> list[dict]:
    """Simulate 1-bar force-close trades from probabilities."""
    cm = {c: i for i, c in enumerate(classes)}
    pb = proba[:, cm.get(1,  cm.get("buy",  0))]
    ps = proba[:, cm.get(-1, cm.get("sell", 2))]

    trades = []
    for i, ts in enumerate(index):
        if pb[i] >= THRESHOLD and pb[i] > ps[i]:
            direction = "buy";  conf = float(pb[i])
        elif ps[i] >= THRESHOLD and ps[i] > pb[i]:
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

        sl = SL_PIPS * pip_size
        tp = TP_PIPS * pip_size

        if direction == "buy":
            if l_nx <= entry - sl:
                pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif h_nx >= entry + tp:
                pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else:
                pips = (c_nx - entry) / pip_size - SPREAD_PIPS - COMM_PIPS
        else:
            if h_nx >= entry + sl:
                pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif l_nx <= entry - tp:
                pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else:
                pips = (entry - c_nx) / pip_size - SPREAD_PIPS - COMM_PIPS

        trades.append({"ts": ts, "dir": direction, "conf": conf, "pips": pips})

    return trades


def _annualized_sharpe(trades: list[dict], bars_per_year: float) -> float:
    """
    Compute annualized Sharpe from a trade list using a per-trade equity series.
    Annualizes by sqrt(trades_per_year) — correct for trade-frequency sampling.
    """
    if len(trades) < 5:
        return float("nan")
    pnl = [t["pips"] for t in trades]
    returns = pd.Series(pnl) / SL_PIPS * RISK_PCT  # per-trade % return

    # Trades per year: interpolate from total trades and date span
    if len(trades) > 1:
        span_days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
        tpy       = len(trades) / span_days * 365.25 if span_days > 0 else len(trades)
    else:
        tpy = len(trades)

    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r < 1e-9:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(tpy))


# ── Per-symbol clean WF ────────────────────────────────────────────────────────

def run_clean_wf(symbol: str) -> None:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]

    print(f"\n{'='*70}")
    print(f"  CLEAN WF BACKTEST — {symbol}")
    print(f"  Per-fold encoder: NO encoder leakage")
    print(f"  Sliding {TRAIN_DAYS}d train / {TEST_DAYS}d test  |  CatBoost  |  threshold={THRESHOLD}")
    print(f"  SL={SL_PIPS}p  TP={TP_PIPS}p  spread={SPREAD_PIPS}p  comm={COMM_PIPS}p  risk={RISK_PCT:.0%}/trade")
    print(f"{'='*70}\n")

    t0     = time.time()
    df_raw = _load_raw(cfg_s["data_path"])
    span   = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bpy    = len(df_raw) / span

    print(f"  Loaded {len(df_raw):,} bars  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    folds = _get_sliding_folds(df_raw.index, TRAIN_DAYS, TEST_DAYS)
    print(f"  Folds: {len(folds)}  (sliding {TRAIN_DAYS}d/{TEST_DAYS}d)\n")

    header = f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>25}  " \
             f"{'Bars':>5}  {'Trd':>4}  {'Win%':>6}  {'Sharpe':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_trades: list[dict] = []

    for fold_idx, train_start, train_end, test_end in folds:
        # ── 1. Slice fold data ──────────────────────────────────────────────
        df_train = df_raw[(df_raw.index >= train_start) & (df_raw.index < train_end)].copy()
        # Use train+test for indicator/encoder transform (correct lookback for test)
        df_fold  = df_raw[(df_raw.index >= train_start) & (df_raw.index < test_end)].copy()
        df_test  = df_raw[(df_raw.index >= train_end)   & (df_raw.index < test_end)].copy()

        if len(df_train) < 500 or len(df_test) < 50:
            continue

        # ── 2. Build features — fit ONLY on fold train data ─────────────────
        pipe_fold = PredictorPipeline(PipelineConfig(
            label_horizon    = LABEL_HORIZON,
            label_threshold  = LABEL_THRESHOLD,
            encoder_mode     = "supervised",
            encoder_latent_dim = 8,
            encoder_epochs   = 30,
        ))
        # Fits scaler + encoder on df_train only
        X_train, y_train = pipe_fold.build_features(df_train, train_frac=1.0)
        X_train = _add_extra_features(df_train, X_train)
        feature_cols = list(X_train.columns)

        if len(X_train) < 100:
            continue

        # ── 3. Transform fold data using fold-specific encoder ──────────────
        # Use full fold (train+test) so test bars have correct indicator lookback
        X_base_fold, _ = pipe_fold._fp.build(df_fold, fit=False)
        if pipe_fold._enc is not None:
            lat_fold   = pipe_fold._enc.transform(df_fold)
            shared     = X_base_fold.index.intersection(lat_fold.index)
            X_fold_all = pd.concat([X_base_fold.loc[shared], lat_fold.loc[shared]], axis=1)
        else:
            X_fold_all = X_base_fold
        X_fold_all = _add_extra_features(df_fold, X_fold_all)

        # Ensure same columns as training
        for c in feature_cols:
            if c not in X_fold_all.columns:
                X_fold_all[c] = 0.0
        X_fold_all = X_fold_all[feature_cols]

        # Slice OOS portion
        X_test  = X_fold_all[(X_fold_all.index >= train_end) & (X_fold_all.index < test_end)]

        if len(X_test) < 10:
            continue

        # ── 4. Train fresh CatBoost on fold train features ──────────────────
        model = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        model.train(X_train, y_train)

        # ── 5. Predict on fold OOS (fold-encoded, never seen by encoder) ────
        proba   = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes = list(model._classes)

        # ── 6. Simulate 1-bar force-close trades ────────────────────────────
        prices_test = df_raw.reindex(X_test.index)
        fold_trades = _simulate_trades(proba, classes, X_test.index, prices_test, pip_size)

        # ── 7. Fold summary ──────────────────────────────────────────────────
        n_t   = len(fold_trades)
        win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
        f_sh  = _annualized_sharpe(fold_trades, bpy) if n_t >= 5 else float("nan")
        sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"

        print(f"  {fold_idx:>4}  "
              f"{str(train_start.date()):>12} → {str(train_end.date()):<12}  "
              f"{str(train_end.date()):>12} → {str(test_end.date()):<12}  "
              f"{len(X_test):>5}  {n_t:>4}  {win_r:>5.0%}  {sh_s:>7}")

        all_trades.extend(fold_trades)

    # ── Overall results ────────────────────────────────────────────────────────
    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"\n{'='*70}")
    print(f"  FINAL RESULT — {symbol} (CLEAN, per-fold encoder)")
    print(f"{'='*70}")

    if not all_trades:
        print("  No trades generated.")
        return

    n_total = len(all_trades)
    wins    = sum(1 for t in all_trades if t["pips"] > 0)
    losses  = sum(1 for t in all_trades if t["pips"] < 0)
    wr      = wins / n_total
    avg_pip = np.mean([t["pips"] for t in all_trades])
    pf_num  = sum(t["pips"] for t in all_trades if t["pips"] > 0)
    pf_den  = abs(sum(t["pips"] for t in all_trades if t["pips"] < 0))
    pf      = pf_num / pf_den if pf_den > 0 else float("inf")

    # Equity curve
    balance = INITIAL_BAL
    eq = [balance]
    for t in all_trades:
        balance += balance * RISK_PCT * (t["pips"] / SL_PIPS)
        eq.append(balance)
    eq_s = pd.Series(eq)
    dd   = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret  = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100

    overall_sh = _annualized_sharpe(all_trades, bpy)

    # Days span
    span_days = (all_trades[-1]["ts"] - all_trades[0]["ts"]).total_seconds() / 86400
    tpy       = n_total / span_days * 365.25 if span_days > 0 else n_total

    print(f"  Sharpe (annualized) : {overall_sh:+.3f}")
    print(f"  Win rate            : {wr:.1%}  ({wins}W / {losses}L)")
    print(f"  Profit factor       : {pf:.2f}")
    print(f"  Max drawdown        : {dd:.1f}%")
    print(f"  Net return          : {ret:+.1f}%")
    print(f"  Total trades        : {n_total}  (~{tpy:.0f}/yr)")
    print(f"  Avg pips/trade      : {avg_pip:+.2f}")
    print()

    # Comparison to original WF numbers
    orig = {"EURUSD": 7.118, "USDJPY": 14.414}
    if symbol in orig:
        print(f"  Comparison:")
        print(f"    Original WF (partial encoder leak) : {orig[symbol]:+.3f}")
        print(f"    Clean WF (this script)              : {overall_sh:+.3f}")
        delta = overall_sh - orig[symbol]
        print(f"    Delta                               : {delta:+.3f}  "
              f"({'leak inflated by {:+.0%}'.format(-delta/orig[symbol]) if delta < 0 else 'clean is higher'})")


def main():
    parser = argparse.ArgumentParser(description="Clean WF backtest (per-fold encoder)")
    parser.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*70}")
    print(f"  LEAKAGE-FREE CANDLE PREDICTOR BACKTEST")
    print(f"  Per-fold encoder: each fold fits fresh scaler+enc8 on train window only")
    print(f"  This is the honest walk-forward Sharpe")
    print(f"{'='*70}")

    for sym in symbols:
        run_clean_wf(sym)

    print("\nDone.")


if __name__ == "__main__":
    main()
