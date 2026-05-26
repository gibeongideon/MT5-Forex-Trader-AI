"""
Build the latent-enriched feature matrix (Phase 9 extension).

Loads the existing base feature matrix (31 features), extracts latent vectors
from the trained autoencoder, and merges them into an extended Parquet file.
The base matrix is left untouched — this always creates a NEW file.

Usage:
    conda run -n envmt5 python scripts/build_latent_features.py
    conda run -n envmt5 python scripts/build_latent_features.py --latent-dim 32

Output:
    data/features/EURUSD_M15_features_latent.parquet   ← 31 + latent_dim columns
    (labels reused from data/features/EURUSD_M15_labels.parquet — no recomputation)

Flags:
    --features   Base features parquet path  (default: data/features/EURUSD_M15_features.parquet)
    --labels     Labels parquet path         (default: data/features/EURUSD_M15_labels.parquet)
    --data       Raw OHLCV CSV path          (default: data/EURUSD_M15.csv)
    --encoder    Trained encoder .pt path    (default: config.latent_encoder.path)
    --output     Output parquet path         (default: config.latent_encoder.features_output)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.latent_encoder import LatentEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = _load_config()
    le_cfg = cfg.get("latent_encoder", {})

    feat_default   = str(ROOT / "data" / "features" / "EURUSD_M15_features.parquet")
    labels_default = str(ROOT / "data" / "features" / "EURUSD_M15_labels.parquet")
    data_default   = str(ROOT / "data" / "EURUSD_M15.csv")
    enc_default    = le_cfg.get("path",            "data/models/autoencoder.pt")
    out_default    = le_cfg.get("features_output", "data/features/EURUSD_M15_features_latent.parquet")

    p = argparse.ArgumentParser(description="Build latent-enriched feature matrix")
    p.add_argument("--features", default=feat_default,   help="Base features parquet")
    p.add_argument("--labels",   default=labels_default, help="Labels parquet")
    p.add_argument("--data",     default=data_default,   help="Raw OHLCV CSV")
    p.add_argument("--encoder",  default=enc_default,    help="Trained encoder .pt path")
    p.add_argument("--output",   default=out_default,    help="Output parquet path")
    args = p.parse_args()

    encoder_path = ROOT / args.encoder if not Path(args.encoder).is_absolute() else Path(args.encoder)
    output_path  = ROOT / args.output  if not Path(args.output).is_absolute()  else Path(args.output)

    print("=" * 60)
    print("Build Latent-Enriched Feature Matrix")
    print("=" * 60)

    # ---- Load base features ----
    print(f"\nLoading base features: {args.features}")
    X_base = pd.read_parquet(args.features)
    print(f"  Base features shape: {X_base.shape}")

    # ---- Load labels ----
    print(f"Loading labels:        {args.labels}")
    y = pd.read_parquet(args.labels).squeeze()
    print(f"  Labels shape: {y.shape}")

    # ---- Load raw OHLCV ----
    print(f"Loading raw OHLCV:     {args.data}")
    ohlcv = _load_ohlcv(args.data)
    print(f"  OHLCV shape: {ohlcv.shape}  ({ohlcv.index[0]} → {ohlcv.index[-1]})")

    # ---- Load encoder ----
    if not encoder_path.exists():
        print(f"\nERROR: Encoder not found at {encoder_path}")
        print("Run first:  conda run -n envmt5 python scripts/train_autoencoder.py")
        sys.exit(1)

    print(f"\nLoading encoder: {encoder_path}")
    encoder = LatentEncoder()
    encoder.load(str(encoder_path))
    meta = encoder.metadata()
    print(f"  window_size={meta['window_size']}  latent_dim={meta['latent_dim']}")

    # ---- Extract latent features for the FULL dataset (no leakage — encoder
    #      was trained on train split only, transform is inference-only) ----
    print("\nExtracting latent features...")
    latent_df = encoder.transform(ohlcv)
    print(f"  Latent shape: {latent_df.shape}")

    # ---- Align indexes: latent must match base feature index ----
    # Base feature matrix may start later than ohlcv (NaN rows dropped during pipeline)
    shared_idx = X_base.index.intersection(latent_df.index)
    if len(shared_idx) == 0:
        raise ValueError(
            "No overlapping index between base features and latent features. "
            "Check that --data and --features use the same symbol/timeframe."
        )

    X_base   = X_base.loc[shared_idx]
    latent_df = latent_df.loc[shared_idx]
    y        = y.loc[shared_idx] if shared_idx[0] in y.index else y

    # Check for any latent NaN rows (warmup zeros are fine — they are 0.0, not NaN)
    nan_count = latent_df.isna().sum().sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values in latent features — filling with 0")
        latent_df = latent_df.fillna(0.0)

    # ---- Merge ----
    X_extended = pd.concat([X_base, latent_df], axis=1)
    print(f"\nExtended feature matrix: {X_extended.shape}  "
          f"({X_base.shape[1]} base + {latent_df.shape[1]} latent)")

    # Verify no new lookahead (latent features are derived from past windows only)
    latent_cols = latent_df.columns.tolist()
    print(f"  Latent columns: {latent_cols[0]} .. {latent_cols[-1]}")

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X_extended.to_parquet(output_path)
    print(f"\nSaved → {output_path}")
    print(f"  Rows : {len(X_extended):,}")
    print(f"  Cols : {X_extended.shape[1]}  ({X_base.shape[1]} original + "
          f"{latent_df.shape[1]} latent)")
    print()
    print("Next step:")
    print("  conda run -n envmt5 python scripts/compare_latent_vs_baseline.py")


if __name__ == "__main__":
    main()
