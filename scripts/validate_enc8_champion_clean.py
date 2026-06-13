"""
validate_enc8_champion_clean.py — Definitive clean test of the +3.14 champion.

The model in question: the NON-candle pipeline champion (data/models/pipeline_*),
XGBoost + enc8 supervised encoder, 40 features (31 base + fractal_corr + 8 latent),
4-bar label (horizon=4, threshold=0.0003), expanding 180d/30d WF, champion exit
(threshold=0.40, SL=30p, TP=60p). Reported WF Sharpe +3.01 / +4.27 (hybrid v2) and
+2.31–3.13 (enc8 baseline) — all built with the encoder fit on the first 80% of
data and reused across folds (encoder leak).

This script reproduces that EXACT config but fits a FRESH enc8 encoder on each
fold's training window only (no leak). Note: this model does NOT use the MTF-EMA
features, so leak #2 does not apply here — this isolates the encoder leak.

Usage:
    python scripts/validate_enc8_champion_clean.py --symbol EURUSD
    python scripts/validate_enc8_champion_clean.py                 # both
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
from src.models.xgboost_model import XGBoostModel
from scripts.train_candle_model import SYMBOL_CFG
from scripts.backtest_champion_baseline import (
    _get_expanding_folds, _load_raw, _simulate_trades, _annualized_sharpe,
    _equity_stats, THRESHOLD, SL_PIPS, TP_PIPS,
)

MIN_TRAIN_DAYS  = 180
STEP_DAYS       = 30
TEST_DAYS       = 30
LABEL_HORIZON   = 4
LABEL_THRESHOLD = 0.0003


def _build_cfg() -> PipelineConfig:
    return PipelineConfig(
        label_horizon      = LABEL_HORIZON,
        label_threshold    = LABEL_THRESHOLD,
        encoder_enabled    = True,
        encoder_mode       = "supervised",
        encoder_latent_dim = 8,
        encoder_epochs     = 30,
        fractal_enabled    = True,
    )


def run_symbol(symbol: str, max_folds=None) -> dict:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])
    folds    = _get_expanding_folds(df_raw.index, MIN_TRAIN_DAYS, STEP_DAYS,
                                    TEST_DAYS, max_folds=max_folds)

    print(f"\n{'='*72}")
    print(f"  +3.14 CHAMPION CLEAN TEST — {symbol}  (enc8 + XGBoost, 40 feat)")
    print(f"  PER-FOLD encoder (no leak)  |  expanding {MIN_TRAIN_DAYS}d/{STEP_DAYS}d  |  {len(folds)} folds")
    print(f"  label h={LABEL_HORIZON} thr={LABEL_THRESHOLD}  |  exit thr={THRESHOLD} SL={SL_PIPS} TP={TP_PIPS}")
    print(f"{'='*72}\n")
    print(f"  {'Fold':>4}  {'Test window':>25}  {'Feat':>4}  {'Trd':>5}  {'Win%':>5}  {'Sharpe':>7}")
    print("  " + "-"*62)

    cfg = _build_cfg()
    all_trades = []
    t0 = time.time()

    for fi, tr_start, tr_end, te_end in folds:
        df_train = df_raw[(df_raw.index >= tr_start) & (df_raw.index < tr_end)].copy()
        df_fold  = df_raw[(df_raw.index >= tr_start) & (df_raw.index < te_end)].copy()
        df_test  = df_raw[(df_raw.index >= tr_end)   & (df_raw.index < te_end)].copy()
        if len(df_train) < 500 or len(df_test) < 50:
            continue

        # per-fold fresh scaler + enc8 (fit on train window only)
        pipe = PredictorPipeline(cfg)
        X_train, y_train = pipe.build_features(df_train, train_frac=1.0)
        cols = list(X_train.columns)
        if len(X_train) < 100:
            continue

        # transform full fold (train+test) with fold-specific encoder
        X_base_fold, _ = pipe._fp.build(df_fold, fit=False)
        if pipe._enc is not None:
            lat = pipe._enc.transform(df_fold)
            sh  = X_base_fold.index.intersection(lat.index)
            X_fold = pd.concat([X_base_fold.loc[sh], lat.loc[sh]], axis=1)
        else:
            X_fold = X_base_fold
        for c in cols:
            if c not in X_fold.columns:
                X_fold[c] = 0.0
        X_fold = X_fold[cols]
        X_test = X_fold[(X_fold.index >= tr_end) & (X_fold.index < te_end)]
        if len(X_test) < 20:
            continue

        model = XGBoostModel(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample=0.8, calibration_cv=3)
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        prices_test = df_raw.reindex(X_test.index)
        ft = _simulate_trades(proba, list(model._classes), X_test.index,
                              prices_test, pip_size)
        n = len(ft)
        wr = sum(1 for t in ft if t["pips"] > 0) / n if n else 0.0
        fsh = _annualized_sharpe(ft) if n >= 10 else float("nan")
        sh_s = f"{fsh:+.2f}" if not np.isnan(fsh) else "  n/a"
        print(f"  {fi:>4}  {str(tr_end.date()):>10} → {str(te_end.date()):<12}  "
              f"{len(cols):>4}  {n:>5}  {wr:>4.0%}  {sh_s:>7}", flush=True)
        all_trades += ft

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")
    if not all_trades:
        print("  No trades."); return {"symbol": symbol, "sharpe": float("nan"), "n": 0}
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t["pips"] > 0)
    wr = wins / n
    dd, ret = _equity_stats(all_trades)
    sh = _annualized_sharpe(all_trades)
    claimed = {"EURUSD": "+3.01 (hybrid) / +2.31–3.13 (enc8)", "USDJPY": "+4.27 (hybrid)"}
    print(f"\n  {'─'*64}")
    print(f"  CLEAN RESULT — {symbol}  (enc8 + XGBoost, per-fold encoder)")
    print(f"  {'─'*64}")
    print(f"  Sharpe (annualized) : {sh:+.3f}")
    print(f"  Win rate            : {wr:.1%}  ({wins}W/{n-wins}L)")
    print(f"  Max drawdown        : {dd:.1f}%")
    print(f"  Net return          : {ret:+.1f}%")
    print(f"  Total trades        : {n}")
    print(f"  Claimed (leaky)     : {claimed.get(symbol,'?')}")
    print(f"  {'─'*64}\n")
    return {"symbol": symbol, "sharpe": sh, "n": n, "win": wr, "dd": dd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()
    syms = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    print(f"\n{'#'*72}\n  DEFINITIVE CLEAN TEST OF THE +3.14 ENC8 CHAMPION\n{'#'*72}")
    res = [run_symbol(s, args.folds) for s in syms]
    print(f"\n{'='*72}\n  SUMMARY")
    for r in res:
        sh = f"{r['sharpe']:+.3f}" if not np.isnan(r['sharpe']) else "n/a"
        print(f"    {r['symbol']:>8}  clean Sharpe={sh}  trades={r['n']}")
    print(f"{'='*72}\nDone.")


if __name__ == "__main__":
    main()
