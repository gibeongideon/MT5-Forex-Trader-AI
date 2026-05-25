"""
Train the Bar Language Model (BarLMModel) — Phase 9.

Trains a tiny Transformer (~200k params) on discretised OHLCV token sequences.
Produces data/models/bar_lm.pt and prints classification report.

Usage:
    conda activate envmt5
    python scripts/train_bar_lm.py
    python scripts/train_bar_lm.py --seq-len 32 --epochs 50 --d-model 64
    python scripts/train_bar_lm.py --features custom_features.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.bar_lm_model import BarLMModel
from src.features.bar_tokenizer import BarTokenizer


def main():
    p = argparse.ArgumentParser(description="Train BarLMModel")
    p.add_argument("--features",   default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--labels",     default="data/features/EURUSD_M15_labels.parquet")
    p.add_argument("--output",     default="data/models/bar_lm.pt")
    p.add_argument("--seq-len",    type=int,   default=32)
    p.add_argument("--d-model",    type=int,   default=32)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=4)
    p.add_argument("--ff-dim",     type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of data used for training (rest = validation)")
    p.add_argument("--no-eval",    action="store_true",
                   help="Skip evaluation after training")
    args = p.parse_args()

    print("Loading data...")
    X = pd.read_parquet(args.features)
    y = pd.read_parquet(args.labels)["label"]

    # Align
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    print(f"X: {X.shape}  Labels: buy={( y==1).sum():,}  hold={(y==0).sum():,}  sell={(y==-1).sum():,}")

    # Temporal train/val split
    split_idx = int(len(X) * args.train_frac)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {len(X_train):,} bars  Val: {len(X_val):,} bars")

    model = BarLMModel(
        seq_len    = args.seq_len,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        ff_dim     = args.ff_dim,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )

    # Count parameters
    try:
        import torch
        dummy = BarLMModel(seq_len=args.seq_len, d_model=args.d_model,
                           n_heads=args.n_heads, n_layers=args.n_layers,
                           ff_dim=args.ff_dim)
        dummy.train(X_train.head(100), y_train.head(100))
        from src.models.bar_lm_model import _BarTransformer
        n_params = sum(p.numel() for p in dummy._net.parameters() if p.requires_grad)
        print(f"\nModel parameters: {n_params:,}")
    except Exception:
        pass

    print("\nTraining BarLMModel...")
    model.train(X_train, y_train)
    model.save(args.output)
    print(f"\nModel saved → {args.output}")

    if not args.no_eval:
        print("\n--- Validation ---")
        proba = model.predict_proba(X_val)           # (N, 3)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        pred_idx = np.argmax(proba, axis=1)          # 0=buy, 1=hold, 2=sell
        # Map back: 0→1, 1→0, 2→-1 (P_buy, P_hold, P_sell order)
        pred_label = np.array([1, 0, -1])[pred_idx]
        true_label = y_val.values

        from sklearn.metrics import classification_report
        print(classification_report(
            true_label, pred_label,
            labels=[-1, 0, 1],
            target_names=["sell", "hold", "buy"],
            zero_division=0,
        ))

        # Confidence coverage
        max_prob = proba.max(axis=1)
        for thr in (0.40, 0.45, 0.50, 0.55):
            pct = (max_prob >= thr).mean() * 100
            print(f"  P ≥ {thr:.0%} coverage: {pct:.1f}%")

        # Distribution of token vocabulary (first 100 bars)
        try:
            tok = BarTokenizer()
            tok.fit(X_train)
            dist = tok.distribution(X_val.head(200))
            top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
            print(f"\nTop-5 bar tokens in val set: {top5}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
