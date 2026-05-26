"""
Train the latent feature autoencoder (Phase 9 extension).

Reads raw OHLCV data, trains the autoencoder on the TRAINING portion only
(same 80/20 split used by build_features.py), and saves the encoder to disk.

Usage:
    conda run -n envmt5 python scripts/train_autoencoder.py
    conda run -n envmt5 python scripts/train_autoencoder.py --epochs 50 --latent-dim 32

Flags:
    --data          Path to raw OHLCV CSV      (default: config.yaml data path)
    --output        Path to save .pt artifact  (default: config.latent_encoder.path)
    --window        OHLCV bars per window       (default: config.latent_encoder.window_size)
    --latent-dim    Latent vector size          (default: config.latent_encoder.latent_dim)
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

    p = argparse.ArgumentParser(description="Train latent feature autoencoder")
    p.add_argument("--data",       default=str(ROOT / "data" / "EURUSD_M15.csv"),
                   help="Path to raw OHLCV CSV")
    p.add_argument("--output",     default=le_cfg.get("path", "data/models/autoencoder.pt"),
                   help="Output path for .pt file")
    p.add_argument("--window",     type=int,   default=le_cfg.get("window_size", 50),
                   help="Window size (OHLCV bars per input)")
    p.add_argument("--latent-dim", type=int,   default=le_cfg.get("latent_dim", 16),
                   help="Latent vector dimensionality")
    p.add_argument("--epochs",     type=int,   default=le_cfg.get("epochs", 30),
                   help="Training epochs")
    p.add_argument("--batch-size", type=int,   default=le_cfg.get("batch_size", 512),
                   help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=le_cfg.get("lr", 1e-3),
                   help="Learning rate")
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of data used for training (matches build_features.py)")
    args = p.parse_args()

    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print("=" * 60)
    print("Latent Autoencoder Training")
    print("=" * 60)
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
    print(f"  Held-out    {len(df) - split:,} bars (autoencoder never sees these)")
    print()

    # Train
    encoder = LatentEncoder(
        window_size  = args.window,
        latent_dim   = args.latent_dim,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
    )
    encoder.fit(df_train)

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
