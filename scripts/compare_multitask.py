"""
Multi-task encoder vs supervised MLP — same 49k M15 dataset.

Config A — supervised MLP enc8 (BASELINE, skip with --skip-a)
  Phase 9 champion: Sharpe +3.13, MaxDD 13.3%
  Uses cached Phase 9 walk-forward models.

Config B — multi-task enc8
  Same MLP encoder + direction head (same as supervised)
  + auxiliary volatility head: predicts normalized next-bar move size
  Total loss = L_direction + 0.3 × L_volatility
  The shared encoder receives gradients from both objectives.

Both: XGBoost downstream, 49k M15 rows, 39 features, 19 folds.

Usage
-----
  conda run -n envmt5 python scripts/compare_multitask.py          # skip A
  conda run -n envmt5 python scripts/compare_multitask.py --run-a  # re-run A
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

MLP_BASELINE = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524)
FILTER_START = pd.Timestamp("2024-05-14 11:15:00")


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def _load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _make_cfg(full_cfg: dict, mode: str, alpha: float, cache_subdir: str) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = mode
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.encoder_multitask_alpha  = alpha
    cfg.wf_cache_dir             = str(ROOT / "data/models" / cache_subdir)
    return cfg


def _run(label: str, cfg: PipelineConfig, df_raw: pd.DataFrame,
         prices: pd.DataFrame) -> dict:
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  Encoder: {cfg.encoder_mode}  "
          f"(alpha={cfg.encoder_multitask_alpha if cfg.encoder_mode=='multitask' else 'n/a'})")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*66}")
    t0 = time.time()

    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)
    r    = pipe.walk_forward(X, y, prices)

    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, sharpe=r.sharpe, maxdd=r.drawdown,
                ret=ret, trades=len(r.trades))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", action="store_true",
                        help="Re-run Config A (supervised MLP) from scratch")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    csv_path = full_cfg["pipeline"]["data_path"]
    prices   = _load_prices(csv_path)
    df_raw   = _load_raw(csv_path)
    df_49k   = df_raw[df_raw.index >= FILTER_START].copy()

    print(f"\nDataset: {len(df_49k):,} bars "
          f"({df_49k.index[0].date()} → {df_49k.index[-1].date()})")

    results = []

    # ── Config A: supervised MLP (baseline) ──────────────────────────────────
    if args.run_a:
        cfg_a = _make_cfg(full_cfg, "supervised", 0.0, "wf_cache_enc_compare/enc8")
        results.append(_run("Config A — supervised MLP  enc8  [BASELINE]",
                            cfg_a, df_49k, prices))
    else:
        print(f"\n  Config A skipped  [known: +{MLP_BASELINE['sharpe']:.2f} Sharpe]")

    # ── Config B: multi-task enc8 ─────────────────────────────────────────────
    cfg_b = _make_cfg(full_cfg, "multitask", 0.3, "wf_cache_multitask")
    results.append(_run("Config B — multi-task  enc8  alpha=0.3",
                        cfg_b, df_49k, prices))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*76)
    print("  Multi-task vs Supervised MLP — enc8, 49k M15 rows (May 2024+)")
    print("="*76)
    print(f"  {'Config':<48} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-"*70)

    b = MLP_BASELINE
    print(f"  {'Supervised MLP enc8  [Phase 9 BASELINE]':<48} "
          f"{b['sharpe']:>+7.2f} {b['maxdd']:>6.1f}% {b['ret']:>+7.1%} {b['trades']:>7}")
    print("  " + "-"*70)

    for r in results:
        delta = r["sharpe"] - b["sharpe"]
        flag  = "  ✓ BEATS baseline!" if delta > 0 else f"  ({delta:+.2f} vs baseline)"
        print(f"  {r['label']:<48} {r['sharpe']:>+7.2f} {r['maxdd']:>6.1f}% "
              f"{r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print("  " + "-"*70)
    print("="*76)
    print()

    for r in results:
        if r["label"].startswith("Config B"):
            delta = r["sharpe"] - b["sharpe"]
            if delta > 0:
                print(f"  RESULT: Multi-task BEATS baseline: {r['sharpe']:+.2f} vs +{b['sharpe']:.2f}")
                print(f"  ACTION: Switch config.yaml encoder.mode: multitask")
            else:
                print(f"  RESULT: Supervised MLP holds: +{b['sharpe']:.2f} vs {r['sharpe']:+.2f}")
                print(f"  ACTION: Keep config.yaml encoder.mode: supervised")
    print()


if __name__ == "__main__":
    main()
