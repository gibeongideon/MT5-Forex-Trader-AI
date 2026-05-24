"""
Train and evaluate the Phase 6 stacking ensemble.

Trains a two-layer ensemble:
  Layer 0: XGBoost + LightGBM + CatBoost + RandomForest (configurable)
  Layer 1: Logistic Regression or LightGBM meta-learner

Uses out-of-fold stacking to prevent leakage between layers.
Saves the full ensemble (base models + meta-learner) to disk.

Usage:
    conda activate envmt5
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --base xgboost lightgbm catboost --meta logistic
    python scripts/train_ensemble.py --no-catboost   # if CatBoost not installed

Arguments:
    --features   Parquet feature matrix
    --labels     Parquet label file
    --split      Parquet train/test split
    --base       Space-separated list of base model names (default: xgboost lightgbm catboost random_forest)
    --meta       Meta-learner type: logistic | lightgbm (default: logistic)
    --folds      CV folds for stacking (default: 5)
    --output     Path to save ensemble artifact
    --no-catboost  Skip CatBoost (use if not installed)
    --compare    Also run walk-forward on XGBoost alone for comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss

from src.ensemble import Ensemble
from src.model_registry import _build_model
from src.metrics import performance_report


def main():
    p = argparse.ArgumentParser(description="Train Phase 6 stacking ensemble")
    p.add_argument("--features",    default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--labels",      default="data/features/EURUSD_M15_labels.parquet")
    p.add_argument("--split",       default="data/features/EURUSD_M15_split.parquet")
    p.add_argument("--base",        nargs="+",
                   default=["xgboost", "lightgbm", "catboost", "random_forest"])
    p.add_argument("--meta",        default="logistic",
                   choices=["logistic", "lightgbm"])
    p.add_argument("--folds",       type=int, default=5)
    p.add_argument("--output",      default="data/models/ensemble.joblib")
    p.add_argument("--no-catboost", action="store_true",
                   help="Skip CatBoost (use if not installed)")
    args = p.parse_args()

    # Remove catboost if not available or --no-catboost flag
    base_names = list(args.base)
    if args.no_catboost and "catboost" in base_names:
        base_names.remove("catboost")
        print("[Warning] Skipping CatBoost (--no-catboost flag)")
    else:
        try:
            import catboost  # noqa: F401
        except ImportError:
            if "catboost" in base_names:
                base_names.remove("catboost")
                print("[Warning] CatBoost not installed — skipping. "
                      "Install: conda install -n envmt5 catboost -c conda-forge")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading features...")
    X = pd.read_parquet(args.features)
    y = pd.read_parquet(args.labels)["label"]

    split_path = Path(args.split)
    if split_path.exists():
        split_df  = pd.read_parquet(split_path)
        train_idx = split_df[split_df["split"] == "train"].index
        test_idx  = split_df[split_df["split"] == "test"].index
        X_train, y_train = X.loc[X.index.isin(train_idx)], y.loc[y.index.isin(train_idx)]
        X_test,  y_test  = X.loc[X.index.isin(test_idx)],  y.loc[y.index.isin(test_idx)]
    else:
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    print(f"Train: {len(X_train):,} rows  Test: {len(X_test):,} rows")
    print(f"Base models: {base_names}")
    print(f"Meta-learner: {args.meta}  |  Stacking folds: {args.folds}")
    print()

    # ── Build and train ensemble ───────────────────────────────────────────────
    base_models = [_build_model(name) for name in base_names]
    ensemble = Ensemble(
        base_models = base_models,
        meta_model  = args.meta,
        n_folds     = args.folds,
    )
    ensemble.train(X_train, y_train)

    # ── Evaluate on test set ───────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──────────────────────────────────────────")
    proba_test = ensemble.predict_proba(X_test)     # (n, 3): [P_buy, P_hold, P_sell]
    pred_idx   = np.argmax(proba_test, axis=1)
    label_map  = {0: 1, 1: 0, 2: -1}
    pred_labels = np.array([label_map[i] for i in pred_idx])

    print(classification_report(
        y_test, pred_labels,
        target_names=["SELL (-1)", "HOLD (0)", "BUY (+1)"],
        zero_division=0,
    ))

    # Log-loss (columns in sklearn order: [-1, 0, 1])
    proba_sklearn = np.column_stack([proba_test[:, 2], proba_test[:, 1], proba_test[:, 0]])
    ll = log_loss(y_test, proba_sklearn, labels=[-1, 0, 1])
    print(f"Log-loss: {ll:.4f}  (random baseline ≈ 1.099)")

    # Confidence distribution
    max_proba = proba_test.max(axis=1)
    print(f"\nConfidence distribution:")
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pct = (max_proba >= thr).mean() * 100
        print(f"  P >= {thr:.2f} : {pct:.1f}% of bars")

    # Meta-learner weights (logistic only)
    try:
        weights = ensemble.model_weights()
        print("\n── Meta-learner weights (how much each model is trusted) ────────")
        print(weights.to_string())
    except Exception:
        pass

    # ── Save ───────────────────────────────────────────────────────────────────
    ensemble.save(args.output)
    print(f"\nPhase 6 complete.")
    print(f"  Next: python scripts/walk_forward.py --model ensemble --threshold 0.40")


if __name__ == "__main__":
    main()
