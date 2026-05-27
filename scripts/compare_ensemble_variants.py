"""
Ensemble variants comparison — Phase 10.

Runs walk-forward for 5 configurations on the same feature matrix and prints
a side-by-side Sharpe / MaxDD / Return / Trades table.

Configs tested
--------------
  A  XGBoost alone           (baseline — supervised enc 8-dim)
  B  XGBoost + CatBoost      blend  (50/50 weighted avg)
  C  XGBoost + CatBoost + LightGBM  blend  (equal weights)
  D  XGBoost + CatBoost      stack  (LightGBM meta-learner, 3-fold OOF)
  E  XGBoost alone           larger encoder: latent_dim=16, epochs=60

Usage
-----
  conda run -n envmt5 python scripts/compare_ensemble_variants.py

  # Skip Config E (encoder retrain takes ~20 min)
  conda run -n envmt5 python scripts/compare_ensemble_variants.py --skip-e

Timing estimates
----------------
  A–D  ~20–40 min total (walk-forward with 31 folds each)
  E    +15–25 min       (encoder retrain + separate walk-forward)
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


def _base_cfg(full_cfg: dict) -> PipelineConfig:
    """PipelineConfig with supervised 8-dim encoder and whatever model_type we set."""
    return PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )


def run_config(label: str, model_type: str, cfg: PipelineConfig,
               df_raw: pd.DataFrame, prices: pd.DataFrame,
               X_prebuilt: pd.DataFrame | None,
               y_prebuilt: pd.Series   | None,
               cache_dir: str) -> dict:
    """
    Run one walk-forward config and return a result dict.

    If X_prebuilt/y_prebuilt are given the feature-build step is skipped
    (encoder already fitted). Pass None to force a fresh build (Config E).
    """
    print(f"\n{'='*62}")
    print(f"  Config {label}  —  {model_type}")
    print(f"{'='*62}")
    t0 = time.time()

    cfg.model_type   = model_type
    cfg.wf_cache_dir = cache_dir
    pipe = PredictorPipeline(cfg)

    if X_prebuilt is not None:
        X, y = X_prebuilt, y_prebuilt
        pipe._fp  = _shared_fp      # reuse fitted FeaturePipeline
        pipe._enc = _shared_enc     # reuse fitted encoder
    else:
        print("  Building features (new encoder)…")
        X, y = pipe.build_features(df_raw)

    result = pipe.walk_forward(X, y, prices)

    eq  = result.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return {
        "label":    label,
        "model":    model_type,
        "sharpe":   result.sharpe,
        "maxdd":    result.drawdown,
        "ret":      ret,
        "trades":   len(result.trades),
        "elapsed":  elapsed,
    }


# ── Shared state (filled after first feature build) ───────────────────────────
_shared_fp  = None
_shared_enc = None


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global _shared_fp, _shared_enc

    p = argparse.ArgumentParser(description="Ensemble variants comparison")
    p.add_argument("--skip-e", action="store_true",
                   help="Skip Config E (encoder retrain)")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = p.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    data_path = full_cfg.get("pipeline", {}).get("data_path", "data/EURUSD_M15.csv")
    df_raw    = _load_raw(data_path)
    prices    = _load_prices(data_path)

    # ── Build shared feature matrix (supervised 8-dim encoder) ────────────────
    print("\n" + "="*62)
    print("  Building shared feature matrix (supervised enc, latent_dim=8)")
    print("="*62)
    base_cfg = _base_cfg(full_cfg)
    base_cfg.model_type = "xgboost"   # placeholder — only features matter here
    base_pipe = PredictorPipeline(base_cfg)
    X, y = base_pipe.build_features(df_raw)

    # Stash the fitted pipeline internals so other configs can reuse them
    _shared_fp  = base_pipe._fp
    _shared_enc = base_pipe._enc
    print(f"  Features: {X.shape[1]} columns  Rows: {len(X):,}")

    results = []
    cache_root = Path("/tmp/ens_compare_cache")

    # ── Config A: XGBoost alone ───────────────────────────────────────────────
    cfg_a = _base_cfg(full_cfg)
    results.append(run_config(
        "A", "xgboost", cfg_a, df_raw, prices,
        X, y, str(cache_root / "xgboost"),
    ))

    # ── Config B: XGBoost + CatBoost blend ───────────────────────────────────
    cfg_b = _base_cfg(full_cfg)
    results.append(run_config(
        "B", "ensemble_xgb_cat", cfg_b, df_raw, prices,
        X, y, str(cache_root / "blend2"),
    ))

    # ── Config C: XGBoost + CatBoost + LightGBM blend ────────────────────────
    cfg_c = _base_cfg(full_cfg)
    results.append(run_config(
        "C", "ensemble_xgb_cat_lgb", cfg_c, df_raw, prices,
        X, y, str(cache_root / "blend3"),
    ))

    # ── Config D: XGBoost + CatBoost stack (meta-learner) ────────────────────
    cfg_d = _base_cfg(full_cfg)
    results.append(run_config(
        "D", "ensemble_stack", cfg_d, df_raw, prices,
        X, y, str(cache_root / "stack"),
    ))

    # ── Config E: XGBoost, larger encoder (latent_dim=16, epochs=60) ─────────
    if not args.skip_e:
        cfg_e = _base_cfg(full_cfg)
        cfg_e.encoder_latent_dim = 16
        cfg_e.encoder_epochs     = 60
        cfg_e.model_type         = "xgboost"
        results.append(run_config(
            "E", "xgboost (enc16)", cfg_e, df_raw, prices,
            None, None,   # force fresh encoder build
            str(cache_root / "enc16"),
        ))
    else:
        print("\n  Config E skipped (--skip-e)")

    # ── Summary table ─────────────────────────────────────────────────────────
    prev_best = 3.13   # supervised encoder + XGBoost (Phase 9 best)

    print("\n\n" + "="*72)
    print("  ENSEMBLE VARIANTS — WALK-FORWARD COMPARISON")
    print("  (all use supervised 8-dim latent encoder, except Config E)")
    print("="*72)
    header = f"  {'Cfg':<4} {'Model':<30} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(header)
    print("  " + "-"*68)

    best_sharpe = max(r["sharpe"] for r in results)
    for r in results:
        marker = " ◄ BEST" if r["sharpe"] == best_sharpe else ""
        beat   = " ✓ BEATS PREV" if r["sharpe"] > prev_best else ""
        print(
            f"  {r['label']:<4} {r['model']:<30} "
            f"{r['sharpe']:>+7.2f} "
            f"{r['maxdd']:>6.1f}% "
            f"{r['ret']:>+7.1%} "
            f"{r['trades']:>7}"
            f"{marker}{beat}"
        )

    print("  " + "-"*68)
    print(f"  Previous best (Phase 9 supervised enc + XGBoost): Sharpe +{prev_best:.2f}")
    print("="*72)
    print()

    if best_sharpe > prev_best:
        winner = next(r for r in results if r["sharpe"] == best_sharpe)
        print(f"  NEW RECORD: Config {winner['label']} ({winner['model']})"
              f"  Sharpe {winner['sharpe']:+.2f}  (was +{prev_best:.2f})")
    else:
        print(f"  No config beat the Phase 9 record of +{prev_best:.2f}")
    print()


if __name__ == "__main__":
    main()
