"""
Retrain champion model for EURUSD and USDJPY on ALL available data.

Uses 30 epochs (proven) with early stopping (no early stopping) so training stops
at the optimal epoch — not too early, not overfit.

Each pair is completely independent:
  - enc8 trained ONLY on that pair's OHLCV
  - XGBoost trained ONLY on that pair's feature matrix
  - Saved to separate artifact directories

Artifacts saved:
  data/models/pipeline_EURUSD/  → scaler.joblib, encoder.pt, model.joblib, meta.json
  data/models/pipeline_USDJPY/  → scaler.joblib, encoder.pt, model.joblib, meta.json

Usage:
    conda run -n envmt5 python scripts/retrain_for_live.py
    conda run -n envmt5 python scripts/retrain_for_live.py --pair EURUSD
    conda run -n envmt5 python scripts/retrain_for_live.py --pair USDJPY
"""

from __future__ import annotations

import argparse
import json
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

# ── Pair definitions ──────────────────────────────────────────────────────────
PAIRS = {
    "EURUSD": dict(
        csv       = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
        spread    = 1.0,
        out_dir   = "data/models/pipeline_EURUSD",
    ),
    "USDJPY": dict(
        csv       = "data/USDJPY_M15.csv",
        pip_size  = 0.01,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
        spread    = 1.0,
        out_dir   = "data/models/pipeline_USDJPY",
    ),
}


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def retrain_pair(symbol: str, full_cfg: dict) -> dict:
    p = PAIRS[symbol]

    print(f"\n{'='*68}")
    print(f"  RETRAINING  {symbol}  — champion config + 30 epochs (proven) early stopping")
    print(f"{'='*68}")

    # Build config — champion hyperparameters + 30 epochs (proven) early stopping
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30        # proven from 25+ walk-forward experiments
    cfg.encoder_patience         = 0        # no early stopping — train on 100% of data
    cfg.candle_tokenizer_enabled = False
    cfg.artifacts_dir            = str(ROOT / p["out_dir"])
    cfg.bt_sl_pips               = p["sl_pips"]
    cfg.bt_tp_pips               = p["tp_pips"]
    cfg.bt_spread                = p["spread"]

    # Load all available data for this pair
    df = _load_raw(str(ROOT / p["csv"]))
    print(f"  Data  : {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Pair  : {symbol}  pip_size={p['pip_size']}  SL={p['sl_pips']}p  TP={p['tp_pips']}p")
    print(f"  Out   : {cfg.artifacts_dir}\n")

    t0 = time.time()

    pipe = PredictorPipeline(cfg)

    # Step 1: Build features — scaler + encoder fit on 100% of data
    print("Step 1/3 — Building features (scaler + enc8 on all data, 30 epochs (proven) + early stop)...")
    X, y = pipe.build_features(df, train_frac=1.0)
    n_feat = X.shape[1]
    print(f"  Feature matrix : {X.shape}  ({n_feat} features = 31 base + 8 latent)")
    print(f"  Label counts   : {y.value_counts().sort_index().to_dict()}\n")

    # Step 2: Train XGBoost on full feature matrix
    print("Step 2/3 — Training XGBoost on full feature matrix...")
    pipe.fit_full(X, y)
    print()

    # Step 3: Save artifacts
    print("Step 3/3 — Saving artifacts...")
    pipe.save()

    elapsed = time.time() - t0

    # Write pair-specific meta
    meta_path = Path(cfg.artifacts_dir) / "pair_meta.json"
    pair_meta = dict(
        symbol    = symbol,
        pip_size  = p["pip_size"],
        sl_pips   = p["sl_pips"],
        tp_pips   = p["tp_pips"],
        spread    = p["spread"],
        n_bars    = len(df),
        n_features= n_feat,
        trained_from = str(df.index[0].date()),
        trained_to   = str(df.index[-1].date()),
        trained_sec  = round(elapsed, 1),
    )
    meta_path.write_text(json.dumps(pair_meta, indent=2))

    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Artifacts: {cfg.artifacts_dir}/")
    print(f"    scaler.joblib  encoder.pt  model.joblib  meta.json  pair_meta.json")

    return pair_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["EURUSD", "USDJPY", "ALL"],
                        default="ALL", help="Which pair to retrain (default: ALL)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    pairs_to_run = list(PAIRS.keys()) if args.pair == "ALL" else [args.pair]

    print(f"\nRetrain for live deployment")
    print(f"Pairs   : {pairs_to_run}")
    print(f"Encoder : supervised enc8  latent_dim=8  epochs=100  no early stopping")
    print(f"Model   : XGBoost (trained on 100% of data after encoder converges)")

    results = {}
    for symbol in pairs_to_run:
        results[symbol] = retrain_pair(symbol, full_cfg)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*68}")
    print(f"  RETRAIN COMPLETE — Models ready for paper trading")
    print(f"{'='*68}")
    for sym, r in results.items():
        print(f"\n  {sym}:")
        print(f"    Artifact dir : {PAIRS[sym]['out_dir']}")
        print(f"    Bars trained : {r['n_bars']:,}  ({r['trained_from']} → {r['trained_to']})")
        print(f"    Features     : {r['n_features']}  (31 base + 8 latent)")
        print(f"    Train time   : {r['trained_sec']/60:.1f} min")

    print(f"\n{'='*68}")
    print(f"  PAPER TRADING STEPS")
    print(f"{'='*68}")
    print(f"""
  1. Verify models load correctly (dry run — no trades placed):

     conda run -n envmt5 python src/bots/pipeline_bot.py \\
         --model-dir data/models/pipeline_EURUSD --dry-run

     conda run -n envmt5 python src/bots/pipeline_bot.py \\
         --model-dir data/models/pipeline_USDJPY --symbol USDJPY --dry-run

  2. Start paper trading EURUSD (runs in background):

     conda run -n envmt5 python src/bots/pipeline_bot.py \\
         --model-dir data/models/pipeline_EURUSD \\
         > logs/bot_EURUSD.log 2>&1 &

  3. Start paper trading USDJPY (runs in background):

     conda run -n envmt5 python src/bots/pipeline_bot.py \\
         --model-dir data/models/pipeline_USDJPY --symbol USDJPY \\
         > logs/bot_USDJPY.log 2>&1 &

  4. Monitor both bots:

     tail -f logs/bot_EURUSD.log
     tail -f logs/bot_USDJPY.log

  5. Check MT5 demo account for live trades:
     Account: 52885998  Server: ICMarketsKE-Demo

  6. After 30 days — compare live Sharpe vs walk-forward Sharpe:
     EURUSD target: Sharpe > 0.8 live  (walk-forward: +2.31)
     USDJPY target: Sharpe > 0.8 live  (walk-forward: +2.76)

  7. If both pass → deploy to live account + build subscription API (MONETIZE.md)
""")


if __name__ == "__main__":
    main()
