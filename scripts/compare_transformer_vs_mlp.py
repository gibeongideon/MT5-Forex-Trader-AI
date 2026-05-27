"""
Transformer encoder vs MLP supervised encoder — direct comparison on 49k dataset.

Config A — MLP supervised enc8 (BASELINE)
  Phase 9 champion: Sharpe +3.13, MaxDD 13.3%, Return +358.3%, 524 trades
  Uses pre-built parquet — skipped by default (pass --run-a to re-run from scratch).

Config B — Transformer enc8 (EXPERIMENT)
  Drop-in replacement for _SupervisedEncoderNet.
  Same 49k dataset (May 2024 → May 2026), same latent_dim=8, same XGBoost downstream.
  Self-attention learns which bars within the 50-bar window matter most.

Usage
-----
  # Fast: skip A (known result), run transformer only
  conda run -n envmt5 python scripts/compare_transformer_vs_mlp.py

  # Full: re-run MLP baseline too (confirms +3.13 reproducibility)
  conda run -n envmt5 python scripts/compare_transformer_vs_mlp.py --run-a

Caches
------
  Config A (MLP)         : data/models/wf_cache_enc_compare/enc8/   (Phase 9 cache)
  Config B (Transformer) : data/models/wf_cache_transformer/
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig

# ── Known baseline (Phase 9 — MLP supervised enc8 on 49k dataset) ────────────
MLP_BASELINE = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524)

# ── Date filter to match 49k dataset exactly ─────────────────────────────────
FILTER_START = pd.Timestamp("2024-05-14 11:15:00")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    return df


def _load_prices(csv_path: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _make_cfg(full_cfg: dict, mode: str, cache_subdir: str) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type    = "xgboost"     # always XGBoost — isolate encoder variable
    cfg.encoder_mode  = mode
    cfg.encoder_latent_dim = 8
    cfg.encoder_epochs     = 30
    cfg.wf_cache_dir  = str(ROOT / "data/models" / cache_subdir)
    return cfg


def _run(label: str, mode: str, cfg: PipelineConfig,
         df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"  Encoder: {mode}  latent_dim={cfg.encoder_latent_dim}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*68}")
    t0 = time.time()

    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)
    r    = pipe.walk_forward(X, y, prices)

    eq  = r.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, mode=mode, sharpe=r.sharpe,
                maxdd=r.drawdown, ret=ret, trades=len(r.trades))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", action="store_true",
                        help="Re-run Config A (MLP baseline) from scratch instead of using cached result")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    csv_path = full_cfg.get("pipeline", {}).get("data_path", "data/EURUSD_M15.csv")
    prices   = _load_prices(csv_path)

    df_raw = _load_raw(csv_path)
    df_49k = df_raw[df_raw.index >= FILTER_START].copy()
    print(f"\nDataset after date filter: {len(df_49k):,} bars "
          f"({df_49k.index[0].date()} → {df_49k.index[-1].date()})")

    results = []

    # ── Config A: MLP supervised enc8 (baseline) ──────────────────────────
    if args.run_a:
        cfg_a = _make_cfg(full_cfg, "supervised", "wf_cache_enc_compare/enc8")
        results.append(_run(
            "Config A — MLP supervised  enc8  [BASELINE]",
            "supervised", cfg_a, df_49k, prices,
        ))
    else:
        print(f"\n  Config A skipped (use --run-a to re-run)  "
              f"[known result: Sharpe +{MLP_BASELINE['sharpe']:.2f}]")

    # ── Config B: Transformer enc8 (experiment) ───────────────────────────
    cfg_b = _make_cfg(full_cfg, "transformer", "wf_cache_transformer")
    results.append(_run(
        "Config B — Transformer enc8  d_model=32  n_heads=4  n_layers=2",
        "transformer", cfg_b, df_49k, prices,
    ))

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  TRANSFORMER vs MLP — enc8 (8 latent dims), XGBoost, 49k rows (May 2024+)")
    print("="*80)
    print(f"  {'Config':<50} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-"*74)

    # Baseline row (always shown)
    b = MLP_BASELINE
    print(f"  {'Config A  MLP supervised  enc8  [BASELINE / Phase 9]':<50} "
          f"{b['sharpe']:>+7.2f} {b['maxdd']:>6.1f}% {b['ret']:>+7.1%} {b['trades']:>7}")
    print("  " + "-"*74)

    for r in results:
        delta = r["sharpe"] - MLP_BASELINE["sharpe"]
        flag  = "  ✓ BEATS MLP" if delta > 0 else f"  ({delta:+.2f} vs MLP)"
        print(f"  {r['label']:<50} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>6.1f}% "
              f"{r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print("  " + "-"*74)
    print("="*80)
    print()

    # Print conclusion
    if results:
        tr = next((r for r in results if r["mode"] == "transformer"), None)
        if tr:
            delta = tr["sharpe"] - MLP_BASELINE["sharpe"]
            if delta > 0:
                print(f"  RESULT: Transformer BEATS MLP baseline: "
                      f"{tr['sharpe']:+.2f} vs +{MLP_BASELINE['sharpe']:.2f}  ({delta:+.2f})")
                print(f"  ACTION: Switch config.yaml encoder.mode to 'transformer' to deploy.")
            else:
                print(f"  RESULT: MLP baseline holds: "
                      f"+{MLP_BASELINE['sharpe']:.2f} vs {tr['sharpe']:+.2f}  ({delta:+.2f})")
                print(f"  ACTION: Keep config.yaml encoder.mode: supervised (default).")
    print()


if __name__ == "__main__":
    main()
