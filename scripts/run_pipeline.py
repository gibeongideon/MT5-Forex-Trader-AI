"""
End-to-end pipeline runner.

Modes
-----
  backtest  Build features + walk-forward evaluation. Prints fold table + metrics.
  train     Build features + fit model on full data + save artifacts for live use.
  predict   Load saved artifacts, run prediction on latest bars from data file.
  features  Build and cache the feature matrix only (no training). Useful for
            inspecting or feeding into other scripts.

Usage
-----
  conda run -n envmt5 python scripts/run_pipeline.py backtest
  conda run -n envmt5 python scripts/run_pipeline.py train
  conda run -n envmt5 python scripts/run_pipeline.py predict
  conda run -n envmt5 python scripts/run_pipeline.py features

Options
-------
  --config   Path to config.yaml  (default: config.yaml in project root)
  --data     Override pipeline.data_path
  --model    Override pipeline.model_type  (xgboost|lightgbm|catboost|ensemble)
  --no-enc   Disable latent encoder for this run (pipeline.encoder.enabled=false)
  --epochs   Override pipeline.encoder.epochs
  --latent   Override pipeline.encoder.latent_dim
  --bars     Number of recent bars to use for predict mode (default: 200)
  --out      Override pipeline.artifacts.directory

All other settings come from config.yaml → pipeline: section.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    return df.sort_index()


def _load_prices(data_path: str) -> pd.DataFrame:
    prices = pd.read_csv(data_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n", flush=True)


def _apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Patch cfg with CLI overrides."""
    if args.data:
        cfg.data_path = args.data
    if args.model:
        cfg.model_type = args.model
    if args.no_enc:
        cfg.encoder_enabled = False
    if args.epochs is not None:
        cfg.encoder_epochs = args.epochs
    if args.latent is not None:
        cfg.encoder_latent_dim = args.latent
    if args.out:
        cfg.artifacts_dir = args.out
    return cfg


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_backtest(pipe: PredictorPipeline, args: argparse.Namespace) -> None:
    """Build features → walk-forward → report."""
    _banner(f"PIPELINE BACKTEST  [{pipe.cfg.model_type.upper()}]")
    pipe.summary()
    print()

    df_raw = _load_raw(pipe.cfg.data_path)
    prices = _load_prices(pipe.cfg.data_path)

    _banner("Step 1 — Feature engineering")
    X, y = pipe.build_features(df_raw)

    _banner("Step 2 — Walk-forward evaluation")
    result = pipe.walk_forward(X, y, prices)
    result.report(title=f"Pipeline  [{pipe.cfg.model_type}]")

    eq  = result.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    print(f"\n  Final balance : ${eq.iloc[-1]:,.2f}  ({ret:+.1%})")
    print(f"  Sharpe        : {result.sharpe:+.2f}")
    print(f"  Max drawdown  : {result.drawdown:.1f}%")
    print(f"  Trades        : {len(result.trades)}")
    print()


def run_train(pipe: PredictorPipeline, args: argparse.Namespace) -> None:
    """Build features → fit on full data → save artifacts."""
    _banner(f"PIPELINE TRAIN  [{pipe.cfg.model_type.upper()}]")
    pipe.summary()
    print()

    df_raw = _load_raw(pipe.cfg.data_path)

    _banner("Step 1 — Feature engineering")
    X, y = pipe.build_features(df_raw)

    _banner("Step 2 — Fit model on full dataset")
    pipe.fit_full(X, y)

    _banner("Step 3 — Save artifacts")
    pipe.save()

    print(f"\n  Artifacts saved to: {pipe.cfg.artifacts_dir}")
    print(f"  Features          : {len(pipe.feature_names())} columns")
    print()
    print("  Ready for live prediction:")
    print(f"    python scripts/run_pipeline.py predict")
    print()


def run_predict(pipe: PredictorPipeline, args: argparse.Namespace) -> None:
    """Load saved artifacts → predict on latest bars."""
    _banner("PIPELINE PREDICT")

    artifact_dir = pipe.cfg.artifacts_dir
    print(f"  Loading artifacts from: {artifact_dir}")
    pipe.load(artifact_dir)
    print()

    df_raw = _load_raw(pipe.cfg.data_path)

    # Use the last N bars (enough for indicator warmup + encoder window)
    n_bars = max(args.bars, pipe.cfg.encoder_window * 3, 200)
    df_slice = df_raw.iloc[-n_bars:]

    print(f"  Using last {len(df_slice):,} bars "
          f"({df_slice.index[0].date()} → {df_slice.index[-1].date()})")
    print(f"  Predicting for: {df_slice.index[-1]}", flush=True)
    print()

    signal = pipe.predict(df_slice)

    direction = signal["signal"].upper()
    conf      = signal["confidence"]
    sizing    = signal["sizing"]

    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  Signal     : {direction:<22}│")
    print(f"  │  Confidence : {conf:<22.1%}│")
    print(f"  │  P_buy      : {signal['P_buy']:<22.4f}│")
    print(f"  │  P_hold     : {signal['P_hold']:<22.4f}│")
    print(f"  │  P_sell     : {signal['P_sell']:<22.4f}│")
    print(f"  │  Risk pct   : {sizing['risk_pct']:<22.2%}│")
    print(f"  │  Skip trade : {str(sizing['skip']):<22}│")
    print(f"  │  SL pips    : {sizing['sl_pips']:<22.1f}│")
    print(f"  └─────────────────────────────────────┘")
    print()


def run_features(pipe: PredictorPipeline, args: argparse.Namespace) -> None:
    """Build and save feature matrix only."""
    _banner("PIPELINE FEATURES")
    pipe.summary()
    print()

    df_raw = _load_raw(pipe.cfg.data_path)
    X, y = pipe.build_features(df_raw)

    out_dir = Path(pipe.cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / "features.parquet"
    lab_path  = out_dir / "labels.parquet"

    X.to_parquet(feat_path)
    y.to_frame("label").to_parquet(lab_path)

    print(f"\n  Saved features → {feat_path}  ({X.shape})")
    print(f"  Saved labels   → {lab_path}   ({y.shape})")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="MT5 end-to-end pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "mode",
        choices=["backtest", "train", "predict", "features"],
        help="Pipeline mode",
    )
    p.add_argument("--config",  default=str(ROOT / "config.yaml"),
                   help="Path to config.yaml")
    p.add_argument("--data",    default=None, help="Override data_path")
    p.add_argument("--model",   default=None,
                   help="Override model_type: xgboost|lightgbm|catboost|ensemble")
    p.add_argument("--no-enc",  action="store_true",
                   help="Disable latent encoder for this run")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override encoder epochs")
    p.add_argument("--latent",  type=int, default=None,
                   help="Override encoder latent_dim")
    p.add_argument("--bars",    type=int, default=200,
                   help="Bars to use for predict mode (default 200)")
    p.add_argument("--out",     default=None,
                   help="Override artifacts directory")
    args = p.parse_args()

    # Load pipeline from config
    pipe = PredictorPipeline.from_config(args.config)

    # Apply CLI overrides
    pipe.cfg = _apply_overrides(pipe.cfg, args)

    # Rebuild internal objects if encoder was toggled off
    if args.no_enc:
        pipe._enc = None

    # Dispatch
    {
        "backtest": run_backtest,
        "train":    run_train,
        "predict":  run_predict,
        "features": run_features,
    }[args.mode](pipe, args)


if __name__ == "__main__":
    main()
