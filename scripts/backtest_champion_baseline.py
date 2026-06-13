"""
backtest_champion_baseline.py — Zero-leakage champion-label baseline (no encoder).

Tests whether the CHAMPION's label definition (4-bar horizon, 0.03% threshold)
has a learnable directional signal using XGBoost on pure engineered features —
no encoder, no latent dims, no leakage of any kind.

This directly answers: "Is the champion's edge real, or was it all encoder leakage?"

CHAMPION CONFIG (mirrored exactly):
  label_horizon   = 4      (predict direction over next 4 bars = 1 hour on M15)
  label_threshold = 0.0003 (0.03% = ~3 pips on EURUSD)
  SL = 30p  TP = 60p  (1:2 R:R, breakeven at 33.3% win rate)
  threshold = 0.40   (lower than candle predictor — more trades, lower confidence)
  WF: expanding 180d min / 30d step (same as original champion WF)
  Trade exit: check SL/TP on each of the 4 bars, force-close at bar 4

WHY THE CHAMPION LABEL IS BETTER THAN THE 1-BAR CANDLE LABEL:
  • 4-bar horizon = 1 hour — smoother, less noise
  • 0.03% threshold → ~40% non-hold labels (vs 8% for candle predictor)
  • More training examples per fold for the model to learn from
  • 1:2 R:R needs only 33% win rate to break even (vs 25% for 1:3 candle)

COMPARISON:
  Original champion WF (leaky encoder) : EURUSD +1.35  / USDJPY +3.24
  Option B champion clean              : EURUSD −1.542 / USDJPY −3.919
  Champion baseline — no encoder       : EURUSD ???    / USDJPY ???

Usage:
    conda run -n envmt5 python scripts/backtest_champion_baseline.py
    conda run -n envmt5 python scripts/backtest_champion_baseline.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_champion_baseline.py --folds 5
    conda run -n envmt5 python scripts/backtest_champion_baseline.py --test-days 15
    conda run -n envmt5 python scripts/backtest_champion_baseline.py --thresholds 0.50 0.55 0.60
    conda run -n envmt5 python scripts/backtest_champion_baseline.py --sweep
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

from sklearn.isotonic import IsotonicRegression

from src.pipeline import PredictorPipeline, PipelineConfig
from src.models.xgboost_model import XGBoostModel
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG


# ── Temporal-holdout calibration wrapper ───────────────────────────────────────

class TemporalCalibratedXGBoost:
    """
    XGBoost + temporal-holdout isotonic calibration.

    Training split (zero leakage):
      base-train  : first CALIB_HOLDOUT_FRAC of training window
      calib-hold  : last  CALIB_HOLDOUT_FRAC of training window  ← OOS for XGBoost
      isotonic    : fitted on (XGBoost OOS probas → actual labels)

    Why better than sklearn calibration_cv=k for time series:
      - k-fold CV shuffles data → calibration set overlaps with training set in time
      - Temporal holdout keeps the calibration window strictly AFTER training
      - Single XGBoost fit instead of k+1 fits → 5× faster

    Usage:
      model = TemporalCalibratedXGBoost()
      model.train(X_train, y_train)           # fits XGB + isotonic
      proba = model.predict_proba(X_test)     # returns calibrated probas
      model._classes                          # sorted class array
    """

    CALIB_FRAC = 0.20     # last 20% of training window used for calibration

    def __init__(self):
        self._xgb      = XGBoostModel(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample=0.8, calibration_cv=0,
        )
        self._iso      = {}          # one IsotonicRegression per class
        self._classes  = None
        self._n_calib  = 0

    def train(self, X: pd.DataFrame, y: pd.Series) -> "TemporalCalibratedXGBoost":
        n          = len(X)
        n_calib    = max(50, int(n * self.CALIB_FRAC))
        n_base     = n - n_calib

        X_base = X.iloc[:n_base]
        y_base = y.iloc[:n_base]
        X_cal  = X.iloc[n_base:]
        y_cal  = y.iloc[n_base:]

        # Train XGBoost on base portion only
        self._xgb.train(X_base, y_base)
        self._classes = self._xgb._classes
        self._n_calib = n_calib

        # Generate OOS probas on calibration holdout
        raw_cal = self._xgb.predict_proba(X_cal)   # (n_calib, n_classes)

        # Fit one isotonic regressor per class (one-vs-rest)
        y_cal_arr = y_cal.values
        for i, cls in enumerate(self._classes):
            y_bin = (y_cal_arr == cls).astype(float)
            iso   = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_cal[:, i], y_bin)
            self._iso[cls] = iso

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw   = self._xgb.predict_proba(X)
        cal   = np.zeros_like(raw)
        for i, cls in enumerate(self._classes):
            cal[:, i] = self._iso[cls].predict(raw[:, i])
        # Re-normalise rows to sum to 1 (isotonic may break normalisation)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

# ── Champion config (mirrors validate_champion_option_b.py exactly) ────────────
MIN_TRAIN_DAYS  = 180
STEP_DAYS       = 30
TEST_DAYS       = 30     # override with --test-days 15 for more folds
LABEL_HORIZON   = 4
LABEL_THRESHOLD = 0.0003
THRESHOLD       = 0.40
SL_PIPS         = 30.0
TP_PIPS         = 60.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_expanding_folds(index, min_train_days, step_days, test_days,
                          max_folds=None):
    start     = index[0]
    end       = index[-1]
    folds     = []
    fold_idx  = 0
    train_end = start + pd.Timedelta(days=min_train_days)
    while train_end + pd.Timedelta(days=test_days) <= end + pd.Timedelta(days=1):
        test_end = min(train_end + pd.Timedelta(days=test_days), end)
        folds.append((fold_idx, start, train_end, test_end))
        fold_idx  += 1
        train_end += pd.Timedelta(days=step_days)
        if max_folds and fold_idx >= max_folds:
            break
    return folds


def _simulate_trades(proba, classes, index, prices, pip_size, threshold=None):
    """4-bar trade exit: check SL/TP on each intermediate bar, force-close at bar 4."""
    if threshold is None:
        threshold = THRESHOLD
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
        fut = prices.index[prices.index > ts]
        if len(fut) < LABEL_HORIZON:
            continue
        entry = prices.loc[ts, "close"]
        sl    = SL_PIPS * pip_size
        tp    = TP_PIPS * pip_size
        pips_result = None
        for bar_ts in fut[:LABEL_HORIZON]:
            row = prices.loc[bar_ts]
            if direction == "buy":
                if row["low"]  <= entry - sl:
                    pips_result = -SL_PIPS - SPREAD_PIPS - COMM_PIPS; break
                if row["high"] >= entry + tp:
                    pips_result =  TP_PIPS - SPREAD_PIPS - COMM_PIPS; break
            else:
                if row["high"] >= entry + sl:
                    pips_result = -SL_PIPS - SPREAD_PIPS - COMM_PIPS; break
                if row["low"]  <= entry - tp:
                    pips_result =  TP_PIPS - SPREAD_PIPS - COMM_PIPS; break
        if pips_result is None:
            c_nx = prices.loc[fut[LABEL_HORIZON - 1], "close"]
            if direction == "buy":
                pips_result = (c_nx - entry) / pip_size - SPREAD_PIPS - COMM_PIPS
            else:
                pips_result = (entry - c_nx) / pip_size - SPREAD_PIPS - COMM_PIPS
        trades.append({"ts": ts, "dir": direction, "conf": conf,
                        "pips": pips_result})
    return trades


def _annualized_sharpe(trades):
    if len(trades) < 10:
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


# ── Per-symbol run ─────────────────────────────────────────────────────────────

def run_symbol(symbol: str, test_days: int, thresholds: list,
               max_folds: int = None) -> list:
    """Train once per fold, evaluate at each threshold. Returns list of result dicts."""
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])

    folds = _get_expanding_folds(df_raw.index, MIN_TRAIN_DAYS, STEP_DAYS,
                                  test_days, max_folds=max_folds)

    print(f"\n{'='*72}")
    print(f"  CHAMPION BASELINE (XGBoost, no encoder)  [{symbol}]")
    print(f"  {len(df_raw):,} bars  "
          f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  WF: expanding {MIN_TRAIN_DAYS}d / step={STEP_DAYS}d / test={test_days}d  |  "
          f"{len(folds)} folds")
    print(f"  Label: horizon={LABEL_HORIZON} bars  threshold={LABEL_THRESHOLD}")
    print(f"  Trade SL={SL_PIPS}p  TP={TP_PIPS}p  (breakeven at 33.3% win rate)")
    print(f"  Thresholds to sweep: {thresholds}")
    print(f"  NO encoder — pure XGBoost on engineered features")
    print(f"{'='*72}\n")

    cfg = PipelineConfig(
        label_horizon    = LABEL_HORIZON,
        label_threshold  = LABEL_THRESHOLD,
        encoder_enabled  = False,
    )

    # Collect raw per-fold predictions: list of (proba, classes, test_index, prices_test)
    fold_preds = []
    t0 = time.time()

    print(f"  Training {len(folds)} folds...", flush=True)

    for fold_idx, train_start, train_end, test_end in folds:
        df_train = df_raw[(df_raw.index >= train_start) &
                          (df_raw.index <  train_end)].copy()
        df_fold  = df_raw[(df_raw.index >= train_start) &
                          (df_raw.index <  test_end)].copy()
        df_test  = df_raw[(df_raw.index >= train_end) &
                          (df_raw.index <  test_end)].copy()

        if len(df_train) < 500 or len(df_test) < 50:
            continue

        pipe = PredictorPipeline(cfg)
        X_train, y_train = pipe.build_features(df_train, train_frac=1.0)
        X_train = _add_extra_features(df_train, X_train)
        feature_cols = list(X_train.columns)

        if len(X_train) < 100:
            continue

        X_base_fold, _ = pipe._fp.build(df_fold, fit=False)
        X_fold = _add_extra_features(df_fold, X_base_fold)
        for c in feature_cols:
            if c not in X_fold.columns:
                X_fold[c] = 0.0
        X_fold = X_fold[feature_cols]
        X_test = X_fold[(X_fold.index >= train_end) & (X_fold.index < test_end)]

        if len(X_test) < 20:
            continue

        model = TemporalCalibratedXGBoost()
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        prices_test = df_raw.reindex(X_test.index)
        fold_preds.append({
            "fold_idx":    fold_idx,
            "train_end":   train_end,
            "test_end":    test_end,
            "proba":       proba,
            "classes":     list(model._classes),
            "test_index":  X_test.index,
            "prices_test": prices_test,
        })
        print(f"    fold {fold_idx:>2}  {str(train_end.date())} → {str(test_end.date())}  "
              f"({len(X_test)} test bars)", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Training done in {elapsed:.1f} min.  Sweeping thresholds...\n")

    # ── Evaluate each threshold against cached fold predictions ──────────────
    results = []
    for thr in thresholds:
        header = (f"  Threshold={thr:.2f}  "
                  f"{'Fold':>4}  {'Train end':>12}  {'Test window':>23}  "
                  f"{'Trd':>4}  {'Win%':>5}  {'Sharpe':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        all_trades = []
        for fp in fold_preds:
            fold_trades = _simulate_trades(
                fp["proba"], fp["classes"], fp["test_index"],
                fp["prices_test"], pip_size, threshold=thr,
            )
            n_t   = len(fold_trades)
            win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
            f_sh  = _annualized_sharpe(fold_trades) if n_t >= 10 else float("nan")
            sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"
            print(f"  {' ':14}{fp['fold_idx']:>4}  {str(fp['train_end'].date()):>12}  "
                  f"{str(fp['train_end'].date()):>10} → {str(fp['test_end'].date()):<12}  "
                  f"{n_t:>4}  {win_r:>4.0%}  {sh_s:>7}",
                  flush=True)
            all_trades.extend(fold_trades)

        if not all_trades:
            print(f"  threshold={thr:.2f}: no trades\n")
            results.append({"symbol": symbol, "threshold": thr, "sharpe": float("nan"),
                             "n_trades": 0, "win_rate": 0.0,
                             "max_dd": float("nan"), "return_pct": float("nan")})
            continue

        n_total = len(all_trades)
        wins    = sum(1 for t in all_trades if t["pips"] > 0)
        wr      = wins / n_total
        dd, ret = _equity_stats(all_trades)
        sh      = _annualized_sharpe(all_trades)
        sh_s    = f"{sh:+.3f}" if not np.isnan(sh) else "n/a"

        print(f"\n  ── threshold={thr:.2f}  [{symbol}] ──")
        print(f"     Sharpe: {sh_s}  |  Win rate: {wr:.1%}  ({wins}W/{n_total-wins}L)  "
              f"|  DD: {dd:.1f}%  |  Trades: {n_total}\n")

        results.append({"symbol": symbol, "threshold": thr, "sharpe": sh,
                         "n_trades": n_total, "win_rate": wr,
                         "max_dd": dd, "return_pct": ret})

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Champion-label XGBoost baseline — no encoder, zero leakage"
    )
    parser.add_argument("--symbol",     default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--test-days",  type=int, default=TEST_DAYS,
                        help="Test window length in days (default 30)")
    parser.add_argument("--folds",      type=int, default=None,
                        help="Limit to first N folds (smoke test)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="One or more trading thresholds to evaluate (e.g. 0.50 0.55 0.60)")
    parser.add_argument("--sweep",      action="store_true",
                        help="Sweep thresholds 0.40 0.45 0.50 0.55 0.60 0.65")
    args = parser.parse_args()

    if args.sweep:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    elif args.thresholds:
        thresholds = sorted(args.thresholds)
    else:
        thresholds = [THRESHOLD]

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  CHAMPION LABEL BASELINE — XGBoost, no encoder, zero leakage")
    print(f"  Expanding {MIN_TRAIN_DAYS}d/{args.test_days}d  |  "
          f"thresholds={thresholds}  |  horizon={LABEL_HORIZON}  |  "
          f"label_thresh={LABEL_THRESHOLD}")
    if args.folds:
        print(f"  NOTE: limited to first {args.folds} folds (smoke test)")
    print(f"{'='*72}")

    all_results = []
    for sym in symbols:
        sym_results = run_symbol(sym, test_days=args.test_days,
                                 thresholds=thresholds, max_folds=args.folds)
        all_results.extend(sym_results)

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  THRESHOLD SWEEP SUMMARY")
    print(f"  {'Symbol':>8}  {'Thresh':>7}  {'Sharpe':>8}  {'Win%':>6}  "
          f"{'MaxDD':>7}  {'Trades':>7}  {'Return':>8}")
    print(f"  {'-'*65}")
    for r in all_results:
        sh_s = f"{r['sharpe']:+.3f}" if not np.isnan(r.get("sharpe", float("nan"))) else "   n/a"
        dd_s = f"{r['max_dd']:.1f}%" if not np.isnan(r.get("max_dd", float("nan"))) else "  n/a"
        ret_s = f"{r.get('return_pct', float('nan')):+.1f}%" \
                if not np.isnan(r.get("return_pct", float("nan"))) else "  n/a"
        print(f"  {r['symbol']:>8}  {r['threshold']:>7.2f}  {sh_s:>8}  "
              f"{r['win_rate']:>5.1%}  {dd_s:>7}  {r['n_trades']:>7}  {ret_s:>8}")
    print(f"{'='*72}\n")

    print("Done.")


if __name__ == "__main__":
    main()
