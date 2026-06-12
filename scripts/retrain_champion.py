"""
Retrain the champion pipeline on ALL available data (no walk-forward splits).

This produces the "deploy-ready" model artifacts used by PipelineBot in live trading.
Walk-forward has already validated the champion config (+3.13 Sharpe over 2 years) —
this script trains on 100% of data to maximise the live model's statistical coverage.

Saved artifacts (→ data/models/pipeline/):
    scaler.joblib   — StandardScaler fitted on all bars
    encoder.pt      — Supervised enc8 trained on all bars
    model.joblib    — XGBoost calibrated on all bars
    meta.json       — Config + feature column names

Usage:
    conda run -n envmt5 python scripts/retrain_champion.py
    conda run -n envmt5 python scripts/retrain_champion.py --data data/EURUSD_M15.csv
    conda run -n envmt5 python scripts/retrain_champion.py --out data/models/pipeline_v2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain champion on all data")
    parser.add_argument("--data", default=None, help="Path to OHLCV CSV (overrides config)")
    parser.add_argument("--out", default=None, help="Artifact output directory (overrides config)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    # Build champion config (mirrors Phase 21 champion: XGBoost + supervised enc8)
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type         = "xgboost"
    cfg.encoder_mode       = "supervised"
    cfg.encoder_latent_dim = 8
    cfg.encoder_epochs     = 30
    cfg.candle_tokenizer_enabled = False

    if args.out:
        cfg.artifacts_dir = args.out

    data_path = args.data or cfg.data_path

    print(f"\n{'='*64}")
    print(f"  Retraining champion on ALL data")
    print(f"  Config : {cfg.model_type} + {cfg.encoder_mode} enc{cfg.encoder_latent_dim}")
    print(f"  Data   : {data_path}")
    print(f"  Out    : {cfg.artifacts_dir}")
    print(f"{'='*64}\n")

    t0 = time.time()

    # Load raw OHLCV
    df = _load_raw(data_path)
    print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    # Build pipeline
    pipe = PredictorPipeline(cfg)

    # Step 1: Build features with train_frac=1.0 so scaler + encoder fit on ALL data
    print("Step 1/3 — Building feature matrix (scaler + encoder fit on all data)...")
    X, y = pipe.build_features(df, train_frac=1.0)
    print(f"  Feature matrix: {X.shape}  classes={y.value_counts().to_dict()}\n")

    # Optional: inject OOS candle signal as features
    symbol_tag = Path(data_path).stem.split("_")[0]  # e.g. "EURUSD" from "EURUSD_M15"
    candle_feat_path = ROOT / "data" / "features" / f"candle_signal_{symbol_tag}.parquet"
    if candle_feat_path.exists():
        print(f"  Injecting OOS candle features from {candle_feat_path.name}...")
        cf = pd.read_parquet(candle_feat_path)[["candle_p_buy", "candle_p_sell"]]
        shared = X.index.intersection(cf.index)
        X = pd.concat([X.loc[shared], cf.loc[shared]], axis=1)
        y = y.reindex(shared)
        pipe._feature_cols = list(X.columns)   # keep meta.json in sync with 42-col model
        print(f"  Candle features injected: {len(shared):,} rows × {X.shape[1]} features\n")
    else:
        print(f"  (No candle features — run build_candle_features.py to add them)\n")

    # Step 2: Train model on full feature matrix
    print("Step 2/3 — Training XGBoost on full feature matrix...")
    pipe.fit_full(X, y)
    print()

    # Step 3: Save all artifacts
    print("Step 3/3 — Saving artifacts...")
    pipe.save()

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*64}")
    print(f"  Done in {elapsed:.1f} min")
    print(f"  Features : {len(pipe.feature_names())}  "
          f"({', '.join(pipe.feature_names()[:4])}...)")
    print(f"  Artifacts: {cfg.artifacts_dir}/")
    print(f"    scaler.joblib  encoder.pt  model.joblib  meta.json")
    print(f"\n  Run the bot:")
    print(f"  conda run -n envmt5 python src/bots/pipeline_bot.py --dry-run")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
