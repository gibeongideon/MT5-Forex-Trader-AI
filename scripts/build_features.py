"""
Build and save an ML-ready feature matrix from historical OHLCV data.

Runs the FeaturePipeline on a CSV file, saves:
  data/features/{SYMBOL}_{TIMEFRAME}_features.parquet   — feature matrix X
  data/features/{SYMBOL}_{TIMEFRAME}_labels.parquet     — label series y
  data/models/scaler.joblib                             — fitted StandardScaler

Usage:
    conda activate envmt5
    python scripts/build_features.py
    python scripts/build_features.py --data data/EURUSD_M15.csv
    python scripts/build_features.py --data data/EURUSD_M15.csv --validate
    python scripts/build_features.py --horizon 4 --threshold 0.0003 --train-frac 0.8

Arguments:
    --data        Path to OHLCV CSV (default: data/EURUSD_M15.csv)
    --horizon     Bars ahead for label generation (default: 4 = 1 hour on M15)
    --threshold   Minimum return to label as buy/sell (default: 0.0003 = 3 pips)
    --train-frac  Fraction of data used as training set for scaler fitting (default: 0.8)
    --validate    Run lookahead validation check after building
    --no-scale    Skip StandardScaler normalisation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.feature_pipeline import FeaturePipeline

DATA_DIR    = Path("data")
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR   = DATA_DIR / "models"


def main():
    p = argparse.ArgumentParser(description="Build ML feature matrix from OHLCV CSV")
    p.add_argument("--data",        default="data/EURUSD_M15.csv")
    p.add_argument("--horizon",     type=int,   default=4,
                   help="Bars ahead for labelling (4 bars × 15 min = 1 hour)")
    p.add_argument("--threshold",   type=float, default=0.0003,
                   help="Min forward return to label as buy/sell (0.0003 ≈ 3 pips on EURUSD)")
    p.add_argument("--train-frac",  type=float, default=0.8,
                   help="Fraction of data to use as training window for scaler fit")
    p.add_argument("--validate",    action="store_true",
                   help="Run lookahead validation after building")
    p.add_argument("--no-scale",    action="store_true",
                   help="Skip StandardScaler normalisation")
    args = p.parse_args()

    csv = Path(args.data)
    if not csv.exists():
        print(f"ERROR: {csv} not found.")
        print("Download data first:  python scripts/download_data.py")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading {csv}...")
    df = pd.read_csv(csv, index_col="time")
    df.index = pd.to_datetime(df.index)
    print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Split train / test ─────────────────────────────────────────────────
    split = int(len(df) * args.train_frac)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"Train: {len(df_train):,} bars  "
          f"({df_train.index[0].date()} → {df_train.index[-1].date()})")
    print(f"Test : {len(df_test):,} bars   "
          f"({df_test.index[0].date()} → {df_test.index[-1].date()})")

    # ── Build pipeline ─────────────────────────────────────────────────────
    pipeline = FeaturePipeline(
        label_horizon=args.horizon,
        label_threshold=args.threshold,
        scale=not args.no_scale,
    )

    print(f"\nBuilding features (horizon={args.horizon} bars, "
          f"threshold={args.threshold:.4f})...")

    X_train, y_train = pipeline.build(df_train, fit=True)
    X_test,  y_test  = pipeline.build(df_test,  fit=False)

    # Combine for full dataset save
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    print(f"\nFeature matrix shape : {X_all.shape}  "
          f"({X_all.shape[1]} features, {len(X_all):,} rows)")
    print(f"Features: {', '.join(pipeline.feature_names())}")

    # ── Label distribution ─────────────────────────────────────────────────
    counts = y_all.value_counts().sort_index()
    total  = len(y_all)
    print(f"\nLabel distribution:")
    label_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    for val, count in counts.items():
        print(f"  {label_names.get(val, val):4s} ({val:+d})  {count:6,}  "
              f"({count / total * 100:.1f}%)")

    # ── Lookahead validation ───────────────────────────────────────────────
    if args.validate:
        print("\nRunning lookahead validation...")
        try:
            pipeline.validate_no_lookahead(df_train)
        except AssertionError as e:
            print(f"FAIL: {e}")
            sys.exit(1)

    # ── Save outputs ───────────────────────────────────────────────────────
    stem = csv.stem  # e.g. "EURUSD_M15"
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_path = FEATURE_DIR / f"{stem}_features.parquet"
    y_path = FEATURE_DIR / f"{stem}_labels.parquet"
    X_all.to_parquet(x_path)
    y_all.to_frame("label").to_parquet(y_path)

    scaler_path = MODEL_DIR / "scaler.joblib"
    if not args.no_scale:
        pipeline.save_scaler(scaler_path)

    # Also save train/test split indices so Phase 4 can reuse the same split
    split_path = FEATURE_DIR / f"{stem}_split.parquet"
    pd.DataFrame({
        "index": X_all.index,
        "split": (["train"] * len(X_train)) + (["test"] * len(X_test)),
    }).set_index("index").to_parquet(split_path)

    print(f"\nSaved:")
    print(f"  Features  → {x_path}")
    print(f"  Labels    → {y_path}")
    print(f"  Split     → {split_path}")
    if not args.no_scale:
        print(f"  Scaler    → {scaler_path}")

    print(f"\nPhase 3 complete. Ready for Phase 4 (XGBoost training).")
    print(f"  Next: python scripts/train_model.py --features {x_path} --labels {y_path}")


if __name__ == "__main__":
    main()
