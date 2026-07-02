"""
build_candle_features.py — Generate OOS candle signal parquet for feature injection.

Uses the 13 cached WF fold CatBoost models to generate truly OOS candle predictions
(each bar is predicted by a model that NEVER saw that bar during training).

Output: data/features/candle_signal_{SYMBOL}.parquet
  columns: candle_p_buy, candle_p_hold, candle_p_sell
  index: DatetimeIndex (same as the OHLCV CSV)

These parquets are then injected as features into the main champion pipeline
and validated through the V5 strict replay path.

Usage:
    conda run -n envmt5 python scripts/build_candle_features.py
    conda run -n envmt5 python scripts/build_candle_features.py --symbol EURUSD
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline
from scripts.verify_candle_oos import (
    SYMBOL_CFG, TRAIN_DAYS, TEST_DAYS,
    _add_extra_features, get_fold_boundaries,
)

OUT_DIR = ROOT / "data" / "features"


def build_symbol(symbol: str) -> None:
    cfg       = SYMBOL_CFG[symbol]
    cache_dir = Path(cfg["wf_cache_dir"])

    fold_files = sorted(
        cache_dir.glob("catboost_fold*.joblib"),
        key=lambda p: int(p.stem.split("fold")[1].split("_")[0]),
    )
    if not fold_files:
        print(f"  [{symbol}] No cached fold models in {cache_dir}")
        return
    print(f"  [{symbol}] Found {len(fold_files)} cached fold models")

    fold_models = []
    for fp in fold_files:
        cached = joblib.load(fp)
        m = cached["model"] if isinstance(cached, dict) else cached
        fold_models.append(m)

    # Build full feature matrix using the deployed model (for scaler/encoder)
    df_raw = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.sort_index()

    print(f"  [{symbol}] Building full feature matrix...")
    pipe = PredictorPipeline.from_config()
    pipe.load(cfg["model_dir"])
    feature_cols = pipe._feature_cols

    X_base, _ = pipe._fp.build(df_raw, fit=False)
    if pipe._enc is not None:
        latent = pipe._enc.transform(df_raw)
        shared = X_base.index.intersection(latent.index)
        X_full = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
    else:
        X_full = X_base
    X_full = _add_extra_features(df_raw, X_full)
    for c in feature_cols:
        if c not in X_full.columns:
            X_full[c] = 0.0
    X_full = X_full[feature_cols]
    print(f"  [{symbol}] Feature matrix: {X_full.shape[0]:,} rows × {X_full.shape[1]} features")

    boundaries = get_fold_boundaries(X_full.index)
    n_folds    = min(len(boundaries), len(fold_models))
    print(f"  [{symbol}] Generating OOS predictions for {n_folds} folds...")

    fold_dfs = []
    total_bars = 0

    for i in range(n_folds):
        fold_idx, train_end, test_end = boundaries[i]
        model = fold_models[i]

        X_oos = X_full[(X_full.index >= train_end) & (X_full.index < test_end)].copy()
        if len(X_oos) < 10:
            continue

        # Align features to fold model's expected columns
        X_in = X_oos.copy()
        for c in feature_cols:
            if c not in X_in.columns:
                X_in[c] = 0.0
        X_in = X_in[feature_cols]

        proba   = model.predict_proba(X_in)
        classes = model.classes_
        cls_map = {c: i for i, c in enumerate(classes)}
        p_buy  = proba[:, cls_map.get(1,  cls_map.get("buy",  0))]
        p_hold = proba[:, cls_map.get(0,  cls_map.get("hold", 1))]
        p_sell = proba[:, cls_map.get(-1, cls_map.get("sell", 2))]

        fold_df = pd.DataFrame({
            "candle_p_buy":  p_buy,
            "candle_p_hold": p_hold,
            "candle_p_sell": p_sell,
        }, index=X_oos.index)

        fold_dfs.append(fold_df)
        total_bars += len(fold_df)

        print(f"    fold {fold_idx:2d}  "
              f"{str(train_end.date()):>12} → {str(test_end.date()):<12}  "
              f"{len(X_oos):>5,} bars")

    if not fold_dfs:
        print(f"  [{symbol}] No predictions generated — check fold models")
        return

    out = pd.concat(fold_dfs).sort_index()
    # Remove any duplicate timestamps (shouldn't happen with non-overlapping folds)
    out = out[~out.index.duplicated(keep="last")]

    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"candle_signal_{symbol}.parquet"
    out.to_parquet(out_path)

    print(f"\n  [{symbol}] Saved {len(out):,} OOS predictions → {out_path}")
    print(f"    Date range : {out.index[0].date()} → {out.index[-1].date()}")
    print(f"    Mean P_buy : {out['candle_p_buy'].mean():.4f}")
    print(f"    Mean P_sell: {out['candle_p_sell'].mean():.4f}")
    # Show how many bars signal threshold ≥ 0.60
    n_signals = ((out["candle_p_buy"] >= 0.60) | (out["candle_p_sell"] >= 0.60)).sum()
    print(f"    Bars with signal (≥0.60): {n_signals:,}  "
          f"({n_signals / len(out):.1%} of total)")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate OOS candle signal parquet")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    args = p.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  BUILD CANDLE SIGNAL FEATURES (OOS, no-lookahead)")
    print(f"  Using {TRAIN_DAYS}d train / {TEST_DAYS}d test WF fold models")
    print(f"  Each bar's prediction comes from a model that NEVER saw that bar")
    print(f"{'='*72}\n")

    for sym in symbols:
        build_symbol(sym)

    print(f"\n{'='*72}")
    print(f"  Done. Use these parquets as features in the main champion pipeline:")
    print(f"    conda run -n envmt5 python scripts/retrain_champion.py")
    print(f"    conda run -n envmt5 python scripts/v5_validate_champion.py --symbol EURUSD")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
