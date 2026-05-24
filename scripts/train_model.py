"""
Train an XGBoost model on the Phase 3 feature matrix and save it.

Usage:
    conda activate envmt5
    python scripts/train_model.py
    python scripts/train_model.py --features data/features/EURUSD_M15_features.parquet \
                                  --labels   data/features/EURUSD_M15_labels.parquet \
                                  --output   data/models/xgboost.joblib

Arguments:
    --features      Parquet file of feature matrix X (from build_features.py)
    --labels        Parquet file of label series y   (from build_features.py)
    --split         Parquet file of train/test split  (from build_features.py)
    --output        Where to save the trained model
    --n-estimators  XGBoost trees (default 300)
    --max-depth     Tree depth (default 4)
    --lr            Learning rate (default 0.05)
    --no-calibrate  Skip isotonic calibration (faster, less accurate probabilities)
    --no-importance Skip feature importance chart (for headless runs)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss

from src.models.xgboost_model import XGBoostModel


def main():
    p = argparse.ArgumentParser(description="Train XGBoost model on feature matrix")
    p.add_argument("--features",      default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--labels",        default="data/features/EURUSD_M15_labels.parquet")
    p.add_argument("--split",         default="data/features/EURUSD_M15_split.parquet")
    p.add_argument("--output",        default="data/models/xgboost.joblib")
    p.add_argument("--n-estimators",  type=int,   default=300)
    p.add_argument("--max-depth",     type=int,   default=4)
    p.add_argument("--lr",            type=float, default=0.05)
    p.add_argument("--subsample",     type=float, default=0.8)
    p.add_argument("--colsample",     type=float, default=0.8)
    p.add_argument("--no-calibrate",  action="store_true")
    p.add_argument("--no-importance", action="store_true")
    args = p.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading feature matrix...")
    X = pd.read_parquet(args.features)
    y = pd.read_parquet(args.labels)["label"]

    # Load train/test split
    split_path = Path(args.split)
    if split_path.exists():
        split_df = pd.read_parquet(split_path)
        train_idx = split_df[split_df["split"] == "train"].index
        test_idx  = split_df[split_df["split"] == "test"].index
        X_train, y_train = X.loc[X.index.isin(train_idx)], y.loc[y.index.isin(train_idx)]
        X_test,  y_test  = X.loc[X.index.isin(test_idx)],  y.loc[y.index.isin(test_idx)]
    else:
        # Fallback: 80/20 split by row
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    print(f"Train: {len(X_train):,} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Test : {len(X_test):,} rows   ({X_test.index[0].date()}  → {X_test.index[-1].date()})")
    print(f"Features: {X_train.shape[1]}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining XGBoost (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, lr={args.lr})...")
    print("Calibration: " + ("OFF" if args.no_calibrate else "ON (isotonic, 5-fold CV)"))

    model = XGBoostModel(
        n_estimators   = args.n_estimators,
        max_depth      = args.max_depth,
        learning_rate  = args.lr,
        subsample      = args.subsample,
        colsample      = args.colsample,
        calibration_cv = 0 if args.no_calibrate else 5,
    )
    model.train(X_train, y_train)
    print("Training complete.")

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("\n── Test Set Evaluation ──────────────────────────────────────────")
    proba_test = model.predict_proba(X_test)   # shape (n, 3): [P_buy, P_hold, P_sell]
    pred_test  = np.argmax(proba_test, axis=1)

    # Map back: argmax 0→buy(1), 1→hold(0), 2→sell(-1)
    label_map  = {0: 1, 1: 0, 2: -1}
    pred_labels = np.array([label_map[i] for i in pred_test])

    print(classification_report(
        y_test, pred_labels,
        target_names=["SELL (-1)", "HOLD (0)", "BUY (+1)"],
        zero_division=0,
    ))

    # Log-loss (lower = better calibrated probabilities)
    # sklearn expects proba columns in class order [-1, 0, 1]
    proba_sklearn = np.column_stack([proba_test[:, 2], proba_test[:, 1], proba_test[:, 0]])
    ll = log_loss(y_test, proba_sklearn, labels=[-1, 0, 1])
    print(f"Log-loss (probability calibration quality): {ll:.4f}")
    print("  (lower = better calibrated; random 3-class baseline ≈ 1.099)")

    # ── Confidence distribution ────────────────────────────────────────────
    max_proba = proba_test.max(axis=1)
    print(f"\nConfidence distribution (max class probability per bar):")
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pct = (max_proba >= threshold).mean() * 100
        print(f"  P >= {threshold:.2f} : {pct:.1f}% of bars would trigger a trade")

    # ── Feature importance ─────────────────────────────────────────────────
    if not args.no_calibrate and not args.no_importance:
        print("\nTop-15 feature importances not available with calibration wrapper.")
        print("Run with --no-calibrate to see importances.")
    elif args.no_calibrate and not args.no_importance:
        try:
            imp = model.feature_importance(top_n=15)
            print("\nTop-15 feature importances:")
            for feat, score in imp.items():
                bar = "█" * int(score * 200)
                print(f"  {feat:<22} {score:.4f}  {bar}")
        except Exception as e:
            print(f"  (Could not compute importances: {e})")

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(args.output)
    print(f"\nPhase 4 complete.")
    print(f"  Next: python scripts/walk_forward.py")


if __name__ == "__main__":
    main()
