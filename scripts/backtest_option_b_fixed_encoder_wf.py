"""
backtest_option_b_fixed_encoder_wf.py — Option B: Fixed encoder, WF CatBoost only.

This is the production-matching validation:
  - Encoder trained ONCE on the first ENCODER_FRAC (60%) of all bars
  - Walk-forward runs only on the REMAINING 40% (encoder never saw this data)
  - Each fold: fresh CatBoost, but encoder is shared and frozen
  - No leakage: all WF test windows are strictly after encoder training cutoff

This matches live production behaviour: the deployed model uses a full-data encoder
that won't be retrained. The question is: does the model generalise to unseen bars
when the encoder is well-trained but frozen?

Walk-forward (on holdout 40% only):
  - Sliding 120d train / 60d test within the holdout portion
  - CatBoost: 300 iters, depth=6, lr=0.05, l2=3.0, subsample=0.8
  - Same params as all other backtest scripts

Usage:
    conda run -n envmt5 python scripts/backtest_option_b_fixed_encoder_wf.py
    conda run -n envmt5 python scripts/backtest_option_b_fixed_encoder_wf.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_option_b_fixed_encoder_wf.py --encoder-frac 0.7
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
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG
from src.models.catboost_model import CatBoostModel

# ── Config ─────────────────────────────────────────────────────────────────────
ENCODER_FRAC    = 0.60    # fraction of ALL bars used to train the fixed encoder
TRAIN_DAYS      = 120     # CatBoost sliding-window train length (within holdout)
TEST_DAYS       = 60      # OOS test window per fold
THRESHOLD       = 0.60
SL_PIPS         = 10.0
TP_PIPS         = 30.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_sliding_folds(index, train_days, test_days):
    start = index[0]
    end   = index[-1]
    td    = pd.Timedelta(days=train_days)
    te    = pd.Timedelta(days=test_days)
    folds = []
    fold_idx  = 0
    train_end = start + td
    while train_end + te <= end + pd.Timedelta(days=1):
        test_end    = min(train_end + te, end)
        train_start = train_end - td
        folds.append((fold_idx, train_start, train_end, test_end))
        fold_idx  += 1
        train_end += te
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


def _annualized_sharpe(trades):
    if len(trades) < 5:
        return float("nan")
    pnl = [t["pips"] for t in trades]
    returns = pd.Series(pnl) / SL_PIPS * RISK_PCT
    if len(trades) > 1:
        span_days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
        tpy = len(trades) / span_days * 365.25 if span_days > 0 else len(trades)
    else:
        tpy = len(trades)
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r < 1e-9:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(tpy))


def run_option_b(symbol: str, encoder_frac: float) -> None:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]

    print(f"\n{'='*72}")
    print(f"  OPTION B — FIXED ENCODER ({encoder_frac:.0%} of data), WF CATBOOST ONLY — {symbol}")
    print(f"  Encoder trained on first {encoder_frac:.0%} of bars, FROZEN for all WF folds")
    print(f"  WF runs on remaining {1-encoder_frac:.0%} (encoder never saw these bars)")
    print(f"{'='*72}\n")

    t0     = time.time()
    df_raw = _load_raw(cfg_s["data_path"])
    n_all  = len(df_raw)

    enc_cutoff_idx = int(n_all * encoder_frac)
    df_enc_train   = df_raw.iloc[:enc_cutoff_idx].copy()
    df_holdout     = df_raw.iloc[enc_cutoff_idx:].copy()
    enc_cutoff_dt  = df_enc_train.index[-1]

    print(f"  Total bars      : {n_all:,}  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  Encoder training: {len(df_enc_train):,} bars  (up to {enc_cutoff_dt.date()})")
    print(f"  WF holdout      : {len(df_holdout):,} bars  ({df_holdout.index[0].date()} → {df_holdout.index[-1].date()})\n")

    # ── Train fixed encoder on the first encoder_frac of data ─────────────────
    print("  Training fixed encoder on first block...")
    pipe_fixed = PredictorPipeline(PipelineConfig(
        label_horizon    = LABEL_HORIZON,
        label_threshold  = LABEL_THRESHOLD,
        encoder_mode     = "supervised",
        encoder_latent_dim = 8,
        encoder_epochs   = 30,
    ))
    pipe_fixed.build_features(df_enc_train, train_frac=1.0)
    print(f"  Encoder trained on {len(df_enc_train):,} bars. Freezing.\n")

    # Pre-compute latent features for ALL data (encoder is fixed — this is valid)
    # The encoder has never seen df_holdout, so latent features on holdout are clean.
    X_base_all, _ = pipe_fixed._fp.build(df_raw, fit=False)
    if pipe_fixed._enc is not None:
        lat_all   = pipe_fixed._enc.transform(df_raw)
        shared    = X_base_all.index.intersection(lat_all.index)
        X_all     = pd.concat([X_base_all.loc[shared], lat_all.loc[shared]], axis=1)
    else:
        X_all = X_base_all
    X_all = _add_extra_features(df_raw, X_all)
    feature_cols = list(X_all.columns)

    # Build labels for all bars (needed to train CatBoost per fold)
    # Use the fixed pipeline's label builder
    _, y_all = pipe_fixed.build_features(df_raw, train_frac=1.0)
    # Align labels with X_all
    y_all = y_all.reindex(X_all.index)

    print(f"  Pre-computed {len(X_all):,} feature rows for full dataset.\n")

    # ── Walk-forward on holdout portion ───────────────────────────────────────
    folds = _get_sliding_folds(df_holdout.index, TRAIN_DAYS, TEST_DAYS)
    print(f"  WF folds on holdout: {len(folds)}  (sliding {TRAIN_DAYS}d/{TEST_DAYS}d)\n")

    if not folds:
        print("  ERROR: not enough holdout data for WF. Try a smaller --encoder-frac.")
        return

    header = (f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>25}  "
              f"{'Bars':>5}  {'Trd':>4}  {'Win%':>6}  {'Sharpe':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_trades = []

    for fold_idx, train_start, train_end, test_end in folds:
        # CatBoost train features: use pre-computed X_all (fixed encoder) for fold train window
        X_fold_train = X_all[(X_all.index >= train_start) & (X_all.index < train_end)]
        y_fold_train = y_all[(y_all.index >= train_start) & (y_all.index < train_end)]
        X_fold_test  = X_all[(X_all.index >= train_end)   & (X_all.index < test_end)]

        # Drop NaN rows (label build can introduce NaN at end)
        mask_train = y_fold_train.notna()
        X_fold_train = X_fold_train[mask_train]
        y_fold_train = y_fold_train[mask_train]

        if len(X_fold_train) < 100 or len(X_fold_test) < 10:
            continue

        # Train fresh CatBoost on this fold's window (fixed-encoder features)
        model = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        model.train(X_fold_train, y_fold_train)

        proba   = model.predict_proba(X_fold_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes = list(model._classes)

        prices_test = df_raw.reindex(X_fold_test.index)
        fold_trades = _simulate_trades(proba, classes, X_fold_test.index, prices_test, pip_size)

        n_t   = len(fold_trades)
        win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
        f_sh  = _annualized_sharpe(fold_trades) if n_t >= 5 else float("nan")
        sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"

        print(f"  {fold_idx:>4}  "
              f"{str(train_start.date()):>12} → {str(train_end.date()):<12}  "
              f"{str(train_end.date()):>12} → {str(test_end.date()):<12}  "
              f"{len(X_fold_test):>5}  {n_t:>4}  {win_r:>5.0%}  {sh_s:>7}")

        all_trades.extend(fold_trades)

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"\n{'='*72}")
    print(f"  FINAL RESULT — {symbol} (OPTION B: fixed encoder {encoder_frac:.0%}, WF CatBoost)")
    print(f"{'='*72}")

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

    balance = INITIAL_BAL
    eq = [balance]
    for t in all_trades:
        balance += balance * RISK_PCT * (t["pips"] / SL_PIPS)
        eq.append(balance)
    eq_s = pd.Series(eq)
    dd   = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret  = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100

    overall_sh = _annualized_sharpe(all_trades)
    span_days  = (all_trades[-1]["ts"] - all_trades[0]["ts"]).total_seconds() / 86400
    tpy        = n_total / span_days * 365.25 if span_days > 0 else n_total

    print(f"  Sharpe (annualized) : {overall_sh:+.3f}")
    print(f"  Win rate            : {wr:.1%}  ({wins}W / {losses}L)")
    print(f"  Profit factor       : {pf:.2f}")
    print(f"  Max drawdown        : {dd:.1f}%")
    print(f"  Net return          : {ret:+.1f}%")
    print(f"  Total trades        : {n_total}  (~{tpy:.0f}/yr)")
    print(f"  Avg pips/trade      : {avg_pip:+.2f}")

    orig  = {"EURUSD":  7.118, "USDJPY": 14.414}
    clean = {"EURUSD": -10.580, "USDJPY": -15.479}
    if symbol in orig:
        print(f"\n  Comparison:")
        print(f"    Original WF (partial encoder leak)  : {orig[symbol]:+.3f}")
        print(f"    Sliding clean WF (encoder starved)  : {clean[symbol]:+.3f}")
        print(f"    Option B fixed encoder (this script): {overall_sh:+.3f}")
        print(f"    WF data coverage                    : last {1-encoder_frac:.0%} of bars ({len(df_holdout):,} bars)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--encoder-frac", type=float, default=ENCODER_FRAC,
                        help="Fraction of all bars used to train the fixed encoder (default 0.60)")
    args = parser.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  OPTION B — FIXED ENCODER ({args.encoder_frac:.0%}), WF CATBOOST ONLY")
    print(f"  Matches production: encoder trained on large block, frozen at deploy time")
    print(f"{'='*72}")

    for sym in symbols:
        run_option_b(sym, args.encoder_frac)

    print("\nDone.")


if __name__ == "__main__":
    main()
