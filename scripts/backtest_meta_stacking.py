"""
backtest_meta_stacking.py — Zero-leakage 2-level stacked ensemble.

ARCHITECTURE
─────────────
Level 1 (base learners, trained on first half of training window):
  • CatBoost — on pure engineered features (no encoder)
  • XGBoost  — on pure engineered features (no encoder)

Level 2 (meta-model, trained on OOS predictions from Level 1):
  • CatBoost meta-model
  • Inputs: original engineered features + [cat_p_buy, cat_p_sell,
            xgb_p_buy, xgb_p_sell] (4 stacking columns)
  • Trains only on the second half of the training window where both
    base models produced out-of-sample predictions

ZERO LOOK-FORWARD GUARANTEE
─────────────────────────────
For each WF fold (120d train / 15d test):

  base-train  [t-120d → t-60d]   → trains CatBoost_L1 + XGBoost_L1
  meta-train  [t-60d  → t]       → L1 models predict OOS here → trains meta
  test        [t      → t+15d]   → L1 models RETRAINED on full 120d
                                   → meta predicts on combined features

No data from the test window is ever visible during training at any level.
The meta-model trains only on predictions that were out-of-sample for L1.

COMPARISON:
  Original WF (leaky encoder)      : EURUSD +7.118  / USDJPY +14.414
  Per-fold fresh encoder           : EURUSD −10.580 / USDJPY −15.479
  Pre-train + fine-tune encoder    : EURUSD −11.314 / USDJPY  −8.719
  Baseline CatBoost (no encoder)   : EURUSD ???     / USDJPY  ???
  Meta stacking CatBoost+XGB       : EURUSD ???     / USDJPY  ???

Usage:
    conda run -n envmt5 python scripts/backtest_meta_stacking.py
    conda run -n envmt5 python scripts/backtest_meta_stacking.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_meta_stacking.py --folds 3
    conda run -n envmt5 python scripts/backtest_meta_stacking.py --threshold 0.50
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
from src.models.xgboost_model import XGBoostModel
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_DAYS      = 120
TEST_DAYS       = 15
BASE_SPLIT_FRAC = 0.5    # first 50% of training window → base models
THRESHOLD       = 0.60
SL_PIPS         = 10.0
TP_PIPS         = 30.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005

META_COLS = ["cat_p_buy", "cat_p_sell", "xgb_p_buy", "xgb_p_sell"]


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


def _proba_cols(proba: np.ndarray, classes: list) -> dict:
    """Return buy/sell probability columns given sorted class array."""
    cm = {c: i for i, c in enumerate(classes)}
    p_buy  = proba[:, cm.get(1,  cm.get("buy",  len(cm)-1))]
    p_sell = proba[:, cm.get(-1, cm.get("sell", 0))]
    return p_buy, p_sell


def _append_meta_cols(X: pd.DataFrame, cat_model, xgb_model) -> pd.DataFrame:
    """Predict from both base models and append 4 meta-feature columns."""
    cat_proba = cat_model.predict_proba(X)
    xgb_proba = xgb_model.predict_proba(X)
    cat_buy, cat_sell = _proba_cols(cat_proba, list(cat_model._classes))
    xgb_buy, xgb_sell = _proba_cols(xgb_proba, list(xgb_model._classes))
    X = X.copy()
    X["cat_p_buy"]  = cat_buy
    X["cat_p_sell"] = cat_sell
    X["xgb_p_buy"]  = xgb_buy
    X["xgb_p_sell"] = xgb_sell
    return X


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


def _build_features(df_train, df_scope, pipe_template_cfg):
    """Fit scaler on df_train, transform df_scope. Returns (X_train, y_train, X_scope)."""
    pipe = PredictorPipeline(pipe_template_cfg)
    X_train, y_train = pipe.build_features(df_train, train_frac=1.0)
    X_train = _add_extra_features(df_train, X_train)

    X_base_scope, _ = pipe._fp.build(df_scope, fit=False)
    X_scope = _add_extra_features(df_scope, X_base_scope)

    feat_cols = list(X_train.columns)
    for c in feat_cols:
        if c not in X_scope.columns:
            X_scope[c] = 0.0
    X_scope = X_scope[feat_cols]
    return X_train, y_train, X_scope, feat_cols


# ── Per-symbol run ─────────────────────────────────────────────────────────────

def run_symbol(symbol: str, threshold: float, max_folds: int = None) -> dict:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])

    folds = _get_sliding_folds(df_raw.index, TRAIN_DAYS, TEST_DAYS,
                                max_folds=max_folds)

    cfg = PipelineConfig(
        label_horizon    = LABEL_HORIZON,
        label_threshold  = LABEL_THRESHOLD,
        encoder_enabled  = False,
    )

    print(f"\n{'='*72}")
    print(f"  META STACKING (CatBoost + XGBoost → CatBoost meta)  [{symbol}]")
    print(f"  {len(df_raw):,} bars  "
          f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  WF: sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  {len(folds)} folds  |  "
          f"threshold={threshold}")
    print(f"  L1 split: first {BASE_SPLIT_FRAC:.0%} of training → base models")
    print(f"            last  {1-BASE_SPLIT_FRAC:.0%} of training → meta-train (OOS from L1)")
    print(f"  NO encoder at any level — pure engineered features")
    print(f"{'='*72}\n")

    header = (f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>21}  "
              f"{'Trd':>4}  {'Win%':>5}  {'Sharpe':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_trades = []
    t0 = time.time()

    for fold_idx, train_start, train_end, test_end in folds:
        # ── Slice windows ──────────────────────────────────────────────────
        df_train_full = df_raw[
            (df_raw.index >= train_start) & (df_raw.index < train_end)
        ].copy()
        df_test = df_raw[
            (df_raw.index >= train_end) & (df_raw.index < test_end)
        ].copy()

        if len(df_train_full) < 500 or len(df_test) < 30:
            continue

        # Split training window into base-train / meta-train
        n_base = int(len(df_train_full) * BASE_SPLIT_FRAC)
        df_base_train = df_train_full.iloc[:n_base].copy()
        df_meta_train = df_train_full.iloc[n_base:].copy()

        if len(df_base_train) < 200 or len(df_meta_train) < 100:
            continue

        # ── L1: build base features and train base models ──────────────────
        # Fit scaler on base-train, transform both base-train and meta-train
        X_base_tr, y_base_tr, X_meta_scope, feat_cols = _build_features(
            df_base_train,
            pd.concat([df_base_train, df_meta_train]),
            cfg,
        )

        # Slice meta-train portion from the scope features
        X_meta_tr = X_meta_scope[X_meta_scope.index >= df_meta_train.index[0]]
        y_meta_tr_raw = pd.Series(
            np.where(
                df_meta_train["close"].shift(-LABEL_HORIZON) >
                df_meta_train["close"] * (1 + LABEL_THRESHOLD), 1,
                np.where(
                    df_meta_train["close"].shift(-LABEL_HORIZON) <
                    df_meta_train["close"] * (1 - LABEL_THRESHOLD), -1, 0
                )
            ),
            index=df_meta_train.index,
        ).dropna()
        y_meta_tr = y_meta_tr_raw.reindex(X_meta_tr.index).dropna()
        X_meta_tr = X_meta_tr.reindex(y_meta_tr.index)

        if len(X_base_tr) < 100 or len(X_meta_tr) < 50:
            continue

        # Train L1 base models on base-train only
        cat_l1 = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        xgb_l1 = XGBoostModel(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample=0.8, calibration_cv=0,
        )
        cat_l1.train(X_base_tr, y_base_tr)
        xgb_l1.train(X_base_tr, y_base_tr)

        # Generate OOS meta-features on meta-train window
        X_meta_with_stack = _append_meta_cols(X_meta_tr, cat_l1, xgb_l1)

        # ── L2: train meta-model on meta-train + stacking features ─────────
        meta_model = CatBoostModel(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            l2_leaf_reg=5.0, subsample=0.8, calibration_cv=0,
        )
        meta_model.train(X_meta_with_stack, y_meta_tr)

        # ── Test: retrain L1 on FULL training window for best quality ───────
        X_full_tr, y_full_tr, X_test_scope, _ = _build_features(
            df_train_full,
            pd.concat([df_train_full, df_test]),
            cfg,
        )
        X_test_only = X_test_scope[X_test_scope.index >= df_test.index[0]]

        if len(X_full_tr) < 200 or len(X_test_only) < 10:
            continue

        cat_final = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        xgb_final = XGBoostModel(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample=0.8, calibration_cv=0,
        )
        cat_final.train(X_full_tr, y_full_tr)
        xgb_final.train(X_full_tr, y_full_tr)

        # Append stacking columns to test features using full-trained L1
        X_test_stacked = _append_meta_cols(X_test_only, cat_final, xgb_final)

        # Align meta-feature columns in test set to what meta-model saw
        for c in list(X_meta_with_stack.columns):
            if c not in X_test_stacked.columns:
                X_test_stacked[c] = 0.0
        X_test_stacked = X_test_stacked[list(X_meta_with_stack.columns)]

        # Meta-model final prediction
        proba   = meta_model.predict_proba(X_test_stacked)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes = list(meta_model._classes)

        prices_test = df_raw.reindex(X_test_stacked.index)
        fold_trades = _simulate_trades(proba, classes, X_test_stacked.index,
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
    print(f"  FINAL RESULT  [{symbol}]  Meta stacking (CatBoost+XGB → CatBoost)")
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
    print(f"\n  Comparison (all leakage-free, {symbol}):")
    print(f"    Original WF (leaky encoder)      : "
          f"EURUSD +7.118 / USDJPY +14.414")
    print(f"    Per-fold fresh encoder           : "
          f"EURUSD −10.580 / USDJPY −15.479")
    print(f"    Pre-train + fine-tune encoder    : "
          f"EURUSD −11.314 / USDJPY  −8.719")
    print(f"    Baseline CatBoost no encoder     : see baseline log")
    print(f"    Meta stacking CatBoost+XGB       : {sh_s}  ← this run")
    print(f"  {'─'*68}\n")

    return {"symbol": symbol, "sharpe": sh, "n_trades": n_total,
            "win_rate": wr, "max_dd": dd, "return_pct": ret}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-leakage meta stacking: CatBoost + XGBoost → CatBoost meta"
    )
    parser.add_argument("--symbol",    default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--folds",     type=int,   default=None)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  ZERO-LEAKAGE META STACKING — CatBoost + XGBoost → CatBoost meta")
    print(f"  Sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  threshold={args.threshold}")
    print(f"  L1 base-train: first {BASE_SPLIT_FRAC:.0%} of window")
    print(f"  L2 meta-train: last  {1-BASE_SPLIT_FRAC:.0%} of window (OOS from L1)")
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
