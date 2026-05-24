"""
Walk-Forward Validation — Phase 4/5.

Simulates realistic model training and evaluation over time:
  1. Train on an initial window of data
  2. Evaluate on the next N days (out-of-sample)
  3. Expand the training window forward by N days
  4. Repeat until end of data
  5. Aggregate all out-of-sample predictions into one equity curve

This prevents overfitting by ensuring the model never sees future data
during training. The final Sharpe ratio is from purely out-of-sample trades.

Usage:
    conda activate envmt5
    python scripts/walk_forward.py
    python scripts/walk_forward.py --model lightgbm --threshold 0.40
    python scripts/walk_forward.py --model random_forest --train-days 180

Arguments:
    --model       Model to use: xgboost | lightgbm | random_forest (default: from config.yaml)
    --features    Parquet feature matrix (from build_features.py)
    --labels      Parquet label file
    --train-days  Initial training window in calendar days (default 180)
    --test-days   Out-of-sample test window per fold in calendar days (default 30)
    --threshold   Min probability to open a trade (default 0.55)
    --sl          Stop loss pips (default 30)
    --tp          Take profit pips (default 60)
    --balance     Starting balance USD (default 10000)
    --risk        Risk per trade as fraction (default 0.01 = 1%)
    --pip-size    Pip size (default 0.0001 for EURUSD)
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.metrics import performance_report, sharpe_ratio, max_drawdown
from src.model_registry import _build_model


def _resolve_model_type(model_arg: Optional[str], config_path: str = "config.yaml") -> str:
    """Return the model type string: CLI arg takes priority over config.yaml."""
    if model_arg:
        return model_arg
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("active_model", "xgboost")
    except FileNotFoundError:
        return "xgboost"


@dataclass
class FoldResult:
    fold:         int
    train_start:  object
    train_end:    object
    test_start:   object
    test_end:     object
    n_trades:     int
    win_rate:     float
    sharpe:       float
    total_return: float


def run_walk_forward(
    X:              pd.DataFrame,
    y:              pd.Series,
    prices:         pd.DataFrame,      # original OHLCV for SL/TP simulation
    train_days:     int   = 180,
    test_days:      int   = 30,
    threshold:      float = 0.55,
    sl_pips:        float = 30.0,
    tp_pips:        float = 60.0,
    pip_size:       float = 0.0001,
    initial_balance: float = 10_000.0,
    risk_pct:       float = 0.01,
    model_type:     str   = "xgboost",
) -> tuple[list[dict], pd.Series, list[FoldResult]]:

    sl_pts = sl_pips * pip_size
    tp_pts = tp_pips * pip_size

    all_trades: list[dict] = []
    equity_pieces: list[pd.Series] = []
    fold_results: list[FoldResult] = []
    balance = initial_balance

    # Build time-based folds
    dates     = X.index
    start_dt  = dates[0]
    train_end = start_dt + pd.Timedelta(days=train_days)

    fold = 0
    while train_end < dates[-1]:
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > dates[-1]:
            test_end = dates[-1]

        # Slice train / test
        X_train = X[X.index < train_end]
        y_train = y[y.index < train_end]
        X_test  = X[(X.index >= train_end) & (X.index < test_end)]

        if len(X_train) < 500 or len(X_test) < 10:
            train_end = test_end
            continue

        # Train a fresh model on expanding window (type from --model / config.yaml).
        # Ensemble is trained once via train_ensemble.py; here we retrain per fold.
        model = _build_model(model_type)
        model.train(X_train, y_train)

        # Predict on test window
        proba = model.predict_proba(X_test)  # (n, 3): [P_buy, P_hold, P_sell]

        # Simulate trades on test bars
        fold_trades, fold_equity, balance = _simulate_trades(
            X_test.index, proba, prices,
            threshold, sl_pts, tp_pts, pip_size, sl_pips, tp_pips,
            balance, risk_pct, fold,
        )

        all_trades.extend(fold_trades)
        if len(fold_equity) > 0:
            equity_pieces.append(fold_equity)

        # Compute fold stats
        pnl = [t["pnl_pips"] for t in fold_trades]
        wins = [p for p in pnl if p > 0]
        wr   = len(wins) / len(pnl) if pnl else 0.0
        sharpe = sharpe_ratio(fold_equity) if len(fold_equity) > 5 else 0.0
        ret    = (fold_equity.iloc[-1] / fold_equity.iloc[0] - 1) * 100 if len(fold_equity) > 0 else 0.0

        fold_results.append(FoldResult(
            fold=fold,
            train_start=start_dt.date(),
            train_end=train_end.date(),
            test_start=train_end.date(),
            test_end=test_end.date(),
            n_trades=len(fold_trades),
            win_rate=wr,
            sharpe=sharpe,
            total_return=ret,
        ))

        fold += 1
        train_end = test_end  # expand window

    equity = pd.concat(equity_pieces) if equity_pieces else pd.Series(dtype=float)
    return all_trades, equity, fold_results


def _simulate_trades(
    test_index, proba, prices,
    threshold, sl_pts, tp_pts, pip_size, sl_pips, tp_pips,
    balance, risk_pct, fold,
) -> tuple[list[dict], pd.Series, float]:
    """Bar-by-bar trade simulation on a test window using model probabilities."""
    trades: list[dict] = []
    equity_curve: list[float] = []
    open_trade: Optional[dict] = None

    for i, ts in enumerate(test_index):
        if ts not in prices.index:
            equity_curve.append(balance)
            continue

        row = prices.loc[ts]
        high  = row["high"]
        low   = row["low"]
        close = row["close"]

        # Check SL/TP on open trade
        if open_trade is not None:
            if open_trade["direction"] == "buy":
                if low <= open_trade["sl"]:
                    _close_trade(open_trade, open_trade["sl"], ts, -sl_pips, "sl")
                elif high >= open_trade["tp"]:
                    _close_trade(open_trade, open_trade["tp"], ts, tp_pips, "tp")
            else:
                if high >= open_trade["sl"]:
                    _close_trade(open_trade, open_trade["sl"], ts, -sl_pips, "sl")
                elif low <= open_trade["tp"]:
                    _close_trade(open_trade, open_trade["tp"], ts, tp_pips, "tp")

            if open_trade.get("exit_time") is not None:
                pnl_dollars = (open_trade["pnl_pips"] / sl_pips) * (balance * risk_pct)
                open_trade["pnl_dollars"] = pnl_dollars
                balance += pnl_dollars
                trades.append(open_trade)
                open_trade = None

        # New signal
        if open_trade is None and i < len(proba):
            p = proba[i] if proba.ndim == 2 else proba
            p_buy, p_hold, p_sell = p

            direction = None
            confidence = 0.0
            if p_buy >= threshold:
                direction, confidence = "buy", float(p_buy)
            elif p_sell >= threshold:
                direction, confidence = "sell", float(p_sell)

            if direction is not None:
                price = close
                if direction == "buy":
                    sl = price - sl_pts
                    tp = price + tp_pts
                else:
                    sl = price + sl_pts
                    tp = price - tp_pts

                open_trade = {
                    "fold": fold, "direction": direction,
                    "entry_time": ts, "entry_price": price,
                    "sl": sl, "tp": tp,
                    "confidence": confidence,
                    "exit_time": None, "pnl_pips": 0.0, "pnl_dollars": 0.0,
                    "sl_pips": sl_pips, "tp_pips": tp_pips,
                }

        equity_curve.append(balance)

    # Force-close at end of window
    if open_trade is not None and len(test_index) > 0:
        last_ts = test_index[-1]
        if last_ts in prices.index:
            last_price = prices.loc[last_ts, "close"]
        else:
            last_price = open_trade["entry_price"]
        pips = (last_price - open_trade["entry_price"]) / pip_size
        if open_trade["direction"] == "sell":
            pips = -pips
        _close_trade(open_trade, last_price, last_ts, pips, "end")
        pnl_dollars = (open_trade["pnl_pips"] / sl_pips) * (balance * risk_pct)
        open_trade["pnl_dollars"] = pnl_dollars
        balance += pnl_dollars
        trades.append(open_trade)

    equity = pd.Series(equity_curve, index=test_index[:len(equity_curve)])
    return trades, equity, balance


def _close_trade(t, price, ts, pips, reason):
    t["exit_time"]   = ts
    t["exit_price"]  = price
    t["pnl_pips"]    = pips
    t["exit_reason"] = reason


def main():
    p = argparse.ArgumentParser(description="Walk-Forward Validation")
    p.add_argument("--model",       default=None,
                   help="Model type: xgboost | lightgbm | random_forest (default: from config.yaml)")
    p.add_argument("--features",    default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--labels",      default="data/features/EURUSD_M15_labels.parquet")
    p.add_argument("--prices",      default="data/EURUSD_M15.csv",
                   help="Original OHLCV CSV for SL/TP simulation")
    p.add_argument("--train-days",  type=int,   default=180)
    p.add_argument("--test-days",   type=int,   default=30)
    p.add_argument("--threshold",   type=float, default=0.55)
    p.add_argument("--sl",          type=float, default=30.0)
    p.add_argument("--tp",          type=float, default=60.0)
    p.add_argument("--balance",     type=float, default=10_000.0)
    p.add_argument("--risk",        type=float, default=0.01)
    p.add_argument("--pip-size",    type=float, default=0.0001)
    args = p.parse_args()

    model_type = _resolve_model_type(args.model)

    # Load features
    print(f"Model: {model_type}  (change via --model or config.yaml active_model)")
    print("Loading features and prices...")
    X      = pd.read_parquet(args.features)
    y      = pd.read_parquet(args.labels)["label"]
    prices = pd.read_csv(args.prices, index_col="time")
    prices.index = pd.to_datetime(prices.index)

    print(f"Features: {X.shape[1]} cols, {len(X):,} rows")
    print(f"Labels:   {len(y):,} rows  (buy={( y==1).sum():,}  hold={(y==0).sum():,}  sell={(y==-1).sum():,})")
    print(f"Walk-forward: train={args.train_days}d, test={args.test_days}d, threshold={args.threshold}")
    print()

    trades, equity, folds = run_walk_forward(
        X, y, prices,
        train_days=args.train_days,
        test_days=args.test_days,
        threshold=args.threshold,
        sl_pips=args.sl,
        tp_pips=args.tp,
        pip_size=args.pip_size,
        initial_balance=args.balance,
        risk_pct=args.risk,
        model_type=model_type,
    )

    # Per-fold summary
    w = 78
    print("─" * w)
    print(f"{'Fold':>4}  {'Train':>22}  {'Test':>22}  {'Trades':>6}  {'WinRate':>7}  {'Sharpe':>6}  {'Return':>7}")
    print("─" * w)
    for f in folds:
        print(f"  {f.fold:>2}  {str(f.train_start)+' → '+str(f.train_end):>22}  "
              f"{str(f.test_start)+' → '+str(f.test_end):>22}  "
              f"{f.n_trades:>6}  {f.win_rate*100:>6.1f}%  {f.sharpe:>6.2f}  {f.total_return:>+6.1f}%")
    print("─" * w)

    if trades and len(equity) > 0:
        performance_report(
            trades, equity, args.balance,
            title=f"WALK-FORWARD OUT-OF-SAMPLE RESULTS ({model_type.upper()})",
            extra_params={
                "Model":        model_type,
                "Folds":        len(folds),
                "Train window": f"{args.train_days} days (expanding)",
                "Test window":  f"{args.test_days} days per fold",
                "Threshold":    f"{args.threshold:.0%}",
                "SL / TP":      f"{args.sl}p / {args.tp}p",
            },
        )
    else:
        print("No trades generated — try lowering --threshold.")


if __name__ == "__main__":
    main()
