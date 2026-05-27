"""
Train the latent feature encoder (supervised or autoencoder mode).

Reads raw OHLCV data, trains the encoder on the TRAINING portion only
(same 80/20 split used by build_features.py), and saves the encoder to disk.

Usage:
    conda run -n envmt5 python scripts/train_autoencoder.py
    conda run -n envmt5 python scripts/train_autoencoder.py --mode supervised --latent-dim 8
    conda run -n envmt5 python scripts/train_autoencoder.py --mode autoencoder --latent-dim 16

Flags:
    --mode          supervised (default) or autoencoder
    --data          Path to raw OHLCV CSV      (default: config.yaml data path)
    --labels        Path to labels parquet      (default: data/features/EURUSD_M15_labels.parquet)
    --output        Path to save .pt artifact  (default: config.latent_encoder.path)
    --window        OHLCV bars per window       (default: config.latent_encoder.window_size)
    --latent-dim    Latent vector size          (default: 8 for supervised, 16 for autoencoder)
    --epochs        Training epochs             (default: config.latent_encoder.epochs)
    --batch-size    Mini-batch size             (default: config.latent_encoder.batch_size)
    --lr            Learning rate               (default: config.latent_encoder.lr)
    --train-frac    Train/test split fraction   (default: 0.8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.latent_encoder import LatentEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()
    le_cfg = cfg.get("latent_encoder", {})

    p = argparse.ArgumentParser(description="Train latent feature encoder")
    p.add_argument("--mode",       default="supervised", choices=["supervised", "autoencoder"],
                   help="Training mode: supervised (default) or autoencoder")
    p.add_argument("--data",       default=str(ROOT / "data" / "EURUSD_M15.csv"),
                   help="Path to raw OHLCV CSV")
    p.add_argument("--labels",     default=str(ROOT / "data" / "features" / "EURUSD_M15_labels.parquet"),
                   help="Path to labels parquet (needed for supervised mode)")
    p.add_argument("--output",     default=le_cfg.get("path", "data/models/autoencoder.pt"),
                   help="Output path for .pt file")
    p.add_argument("--window",     type=int,   default=le_cfg.get("window_size", 50),
                   help="Window size (OHLCV bars per input)")
    p.add_argument("--latent-dim", type=int,   default=None,
                   help="Latent vector dimensionality (default: 8 for supervised, 16 for autoencoder)")
    p.add_argument("--epochs",     type=int,   default=le_cfg.get("epochs", 30),
                   help="Training epochs")
    p.add_argument("--batch-size", type=int,   default=le_cfg.get("batch_size", 4096),
                   help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=le_cfg.get("lr", 1e-3),
                   help="Learning rate")
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of data used for training (matches build_features.py)")
    args = p.parse_args()

    # Resolve latent dim default by mode
    if args.latent_dim is None:
        args.latent_dim = 8 if args.mode == "supervised" else 16

    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print("=" * 60)
    print(f"Latent Encoder Training  [{args.mode.upper()}]")
    print("=" * 60)
    print(f"  Mode        : {args.mode}")
    print(f"  Data        : {args.data}")
    print(f"  Window size : {args.window}")
    print(f"  Latent dim  : {args.latent_dim}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Train frac  : {args.train_frac}")
    print(f"  Output      : {output_path}")
    print()

    # Load raw OHLCV
    df = _load_ohlcv(args.data)
    print(f"  Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")

    # Split — train only (match build_features.py default 80/20)
    split = int(len(df) * args.train_frac)
    df_train = df.iloc[:split]
    split_date = df_train.index[-1]
    print(f"  Training on {len(df_train):,} bars (up to {split_date})")
    print(f"  Held-out    {len(df) - split:,} bars (encoder never sees these)")
    print()

    # Load labels if supervised
    y_train = None
    if args.mode == "supervised":
        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"ERROR: labels file not found: {labels_path}")
            print("Run scripts/build_features.py first to generate labels.")
            sys.exit(1)
        labels_all = pd.read_parquet(labels_path)["label"]
        y_train    = labels_all[labels_all.index.isin(df_train.index)]
        print(f"  Labels      : {len(y_train):,} aligned  "
              f"(sell={int((y_train==-1).sum())}, hold={int((y_train==0).sum())}, "
              f"buy={int((y_train==1).sum())})")
        print()

    # Train
    encoder = LatentEncoder(
        mode         = args.mode,
        window_size  = args.window,
        latent_dim   = args.latent_dim,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
    )
    encoder.fit(df_train, y=y_train)

    # Save
    encoder.save(str(output_path))
    print()
    print(f"Encoder saved → {output_path}")
    print(f"  metadata: {encoder.metadata()}")
    print()
    print("Next step:")
    print("  conda run -n envmt5 python scripts/build_latent_features.py")


if __name__ == "__main__":
    main()
