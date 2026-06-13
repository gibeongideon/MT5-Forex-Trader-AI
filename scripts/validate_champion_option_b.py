"""
validate_champion_option_b.py — Option B leakage check for the champion pipeline.

Same methodology as backtest_option_b_fixed_encoder_wf.py but for the main XGBoost
champion (4-bar horizon, expanding WF) instead of the candle predictor.

Tests both configs:
  v1 (40 features) — XGBoost + enc8, no candle injection
  v2 (42 features) — XGBoost + enc8 + candle_p_buy/sell (current deployed champion)

Option B setup:
  - Encoder trained ONCE on first 60% of bars (36,000 bars, ~1.5 years)
  - WF runs only on the remaining 40% holdout (encoder never saw these bars)
  - XGBoost retrained per fold (expanding window: 180d min, 30d step)
  - No leakage possible: all test windows are strictly after encoder cutoff

Original WF Sharpe (full 2.4 years, expanding):
  v1: EURUSD +1.35 / USDJPY +3.24
  v2: EURUSD +3.01 / USDJPY +4.27

If Option B Sharpe ≈ original → no leakage inflation.
If Option B Sharpe << original → encoder was inflating early folds.

Usage:
    conda run -n envmt5 python scripts/validate_champion_option_b.py
    conda run -n envmt5 python scripts/validate_champion_option_b.py --symbol EURUSD
    conda run -n envmt5 python scripts/validate_champion_option_b.py --encoder-frac 0.7
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
from src.models.model_registry import _build_model
from src.evaluation.backtester import BacktestConfig, Backtester

# ── Config — mirrors champion setup ───────────────────────────────────────────
ENCODER_FRAC    = 0.60
MIN_TRAIN_DAYS  = 180
STEP_DAYS       = 30
TEST_DAYS       = 30
LABEL_HORIZON   = 4
LABEL_THRESHOLD = 0.0003
THRESHOLD       = 0.40
SL_PIPS         = 30.0
TP_PIPS         = 60.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0

SYMBOL_CFG = {
    "EURUSD": dict(
        data_path = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        orig_v1   = 1.35,
        orig_v2   = 3.01,
    ),
    "USDJPY": dict(
        data_path = "data/USDJPY_M15.csv",
        pip_size  = 0.01,
        orig_v1   = 3.24,
        orig_v2   = 4.27,
    ),
}


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_expanding_folds(index, min_train_days, step_days, test_days):
    start = index[0]
    end   = index[-1]
    folds = []
    fold_idx  = 0
    train_end = start + pd.Timedelta(days=min_train_days)
    while train_end + pd.Timedelta(days=test_days) <= end + pd.Timedelta(days=1):
        test_end = min(train_end + pd.Timedelta(days=test_days), end)
        folds.append((fold_idx, start, train_end, test_end))
        fold_idx  += 1
        train_end += pd.Timedelta(days=step_days)
    return folds


def _simulate_trades(proba, classes, index, prices, pip_size):
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
        # Force-close after 4 bars (label_horizon)
        fut = prices.index[prices.index > ts]
        if len(fut) < LABEL_HORIZON:
            continue
        nt    = fut[LABEL_HORIZON - 1]
        entry = prices.loc[ts,  "close"]
        c_nx  = prices.loc[nt,  "close"]
        # Check SL/TP on intermediate bars
        sl = SL_PIPS * pip_size
        tp = TP_PIPS * pip_size
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
            if direction == "buy":
                pips_result = (c_nx - entry) / pip_size - SPREAD_PIPS - COMM_PIPS
            else:
                pips_result = (entry - c_nx) / pip_size - SPREAD_PIPS - COMM_PIPS
        trades.append({"ts": ts, "dir": direction, "conf": conf, "pips": pips_result})
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
    dd  = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100
    return dd, ret


def run_symbol(symbol: str, encoder_frac: float) -> None:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])
    n_all    = len(df_raw)

    enc_cutoff_idx = int(n_all * encoder_frac)
    df_enc_train   = df_raw.iloc[:enc_cutoff_idx].copy()
    df_holdout     = df_raw.iloc[enc_cutoff_idx:].copy()
    enc_cutoff_dt  = df_enc_train.index[-1]

    print(f"\n{'='*72}")
    print(f"  CHAMPION OPTION B VALIDATION — {symbol}")
    print(f"  Encoder trained on first {encoder_frac:.0%} ({len(df_enc_train):,} bars, "
          f"up to {enc_cutoff_dt.date()})")
    print(f"  WF holdout: {len(df_holdout):,} bars  "
          f"({df_holdout.index[0].date()} → {df_holdout.index[-1].date()})")
    print(f"  WF: expanding {MIN_TRAIN_DAYS}d min, step={STEP_DAYS}d, test={TEST_DAYS}d")
    print(f"{'='*72}\n")

    # ── Build fixed-encoder feature matrix for ALL data ────────────────────────
    t0 = time.time()

    for config_tag, use_candle in [("v1 (40 feat, no candle)", False),
                                   ("v2 (42 feat, candle)", True)]:

        candle_path = ROOT / "data" / "features" / f"candle_signal_{symbol}.parquet"
        if use_candle and not candle_path.exists():
            print(f"  Skipping {config_tag}: candle parquet not found at {candle_path}")
            continue

        print(f"\n  --- Config: {config_tag} ---")
        print(f"  Training fixed encoder on first {encoder_frac:.0%} block...", flush=True)

        pipe_fixed = PredictorPipeline(PipelineConfig(
            label_horizon    = LABEL_HORIZON,
            label_threshold  = LABEL_THRESHOLD,
            encoder_mode     = "supervised",
            encoder_latent_dim = 8,
            encoder_epochs   = 30,
        ))
        # Fit encoder on first 60% only
        pipe_fixed.build_features(df_enc_train, train_frac=1.0)
        print(f"  Encoder frozen at {enc_cutoff_dt.date()}.\n")

        # Pre-compute feature matrix for ALL bars using the frozen encoder
        X_base_all, _ = pipe_fixed._fp.build(df_raw, fit=False)
        if pipe_fixed._enc is not None:
            lat_all = pipe_fixed._enc.transform(df_raw)
            shared  = X_base_all.index.intersection(lat_all.index)
            X_all   = pd.concat([X_base_all.loc[shared], lat_all.loc[shared]], axis=1)
        else:
            X_all = X_base_all

        # Inject candle features for v2
        if use_candle:
            cf = pd.read_parquet(candle_path)[["candle_p_buy", "candle_p_sell"]]
            sh = X_all.index.intersection(cf.index)
            if len(sh) > 0:
                X_all = pd.concat([X_all.loc[sh], cf.loc[sh]], axis=1)
            else:
                print("  WARNING: no candle feature overlap, skipping v2.")
                continue

        # Rebuild labels for full dataset
        _, y_all = pipe_fixed.build_features(df_raw, train_frac=1.0)
        y_all = y_all.reindex(X_all.index)
        feature_cols = list(X_all.columns)

        print(f"  Feature matrix: {X_all.shape}  columns={len(feature_cols)}")
        print(f"  Pre-computed for all {len(X_all):,} rows.\n")

        # ── WF on holdout only ─────────────────────────────────────────────────
        folds = _get_expanding_folds(df_holdout.index, MIN_TRAIN_DAYS, STEP_DAYS, TEST_DAYS)
        print(f"  WF folds on holdout: {len(folds)}\n")

        if not folds:
            print("  ERROR: not enough holdout data. Try smaller --encoder-frac.")
            continue

        header = (f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>21}  "
                  f"{'Trd':>4}  {'Win%':>5}  {'Sharpe':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        all_trades = []

        for fold_idx, train_start, train_end, test_end in folds:
            X_fold_train = X_all[(X_all.index >= train_start) & (X_all.index < train_end)]
            y_fold_train = y_all[(y_all.index >= train_start) & (y_all.index < train_end)]
            X_fold_test  = X_all[(X_all.index >= train_end)   & (X_all.index < test_end)]

            mask = y_fold_train.notna()
            X_fold_train = X_fold_train[mask]
            y_fold_train = y_fold_train[mask]

            if len(X_fold_train) < 200 or len(X_fold_test) < 20:
                continue

            model = _build_model("xgboost")
            model.train(X_fold_train, y_fold_train)

            proba   = model.predict_proba(X_fold_test)
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            classes = list(model._classes)

            prices_test = df_raw.reindex(X_fold_test.index)
            fold_trades = _simulate_trades(proba, classes, X_fold_test.index,
                                           prices_test, pip_size)
            n_t   = len(fold_trades)
            win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
            f_sh  = _annualized_sharpe(fold_trades) if n_t >= 10 else float("nan")
            sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"

            print(f"  {fold_idx:>4}  "
                  f"{str(train_start.date()):>12} → {str(train_end.date()):<12}  "
                  f"{str(train_end.date()):>10} → {str(test_end.date()):<10}  "
                  f"{n_t:>4}  {win_r:>4.0%}  {sh_s:>7}")

            all_trades.extend(fold_trades)

        # ── Final result ───────────────────────────────────────────────────────
        print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")
        print(f"\n  {'─'*60}")
        print(f"  RESULT — {symbol}  {config_tag}")
        print(f"  {'─'*60}")

        if not all_trades:
            print("  No trades generated.")
            continue

        n_total = len(all_trades)
        wins    = sum(1 for t in all_trades if t["pips"] > 0)
        wr      = wins / n_total
        dd, ret = _equity_stats(all_trades)
        sh      = _annualized_sharpe(all_trades)
        sh_s    = f"{sh:+.3f}" if not np.isnan(sh) else "n/a"

        orig = cfg_s["orig_v2"] if use_candle else cfg_s["orig_v1"]
        print(f"  Sharpe (annualized) : {sh_s}")
        print(f"  Win rate            : {wr:.1%}  ({wins}W / {n_total-wins}L)")
        print(f"  Max drawdown        : {dd:.1f}%")
        print(f"  Net return          : {ret:+.1f}%")
        print(f"  Total trades        : {n_total}")
        print(f"\n  Comparison:")
        print(f"    Original WF (possible encoder leak): {orig:+.3f}")
        if not np.isnan(sh):
            delta = sh - orig
            if abs(delta) < 0.5:
                verdict = "CONSISTENT — leakage not detected"
            elif delta < 0:
                verdict = f"DEGRADED by {abs(delta):.2f} — encoder may have inflated original WF"
            else:
                verdict = f"STRONGER — holdout is favourable"
            print(f"    Option B clean WF                 : {sh_s}")
            print(f"    Delta                             : {delta:+.3f}  → {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",       default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--encoder-frac", type=float, default=ENCODER_FRAC)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  CHAMPION PIPELINE — OPTION B LEAKAGE VALIDATION")
    print(f"  Encoder frozen on first {args.encoder_frac:.0%} of data")
    print(f"  XGBoost WF on remaining {1-args.encoder_frac:.0%} (truly unseen)")
    print(f"  Tests both v1 (40 feat) and v2 (42 feat + candle)")
    print(f"{'='*72}")

    for sym in symbols:
        run_symbol(sym, args.encoder_frac)

    print("\nDone.")


if __name__ == "__main__":
    main()
