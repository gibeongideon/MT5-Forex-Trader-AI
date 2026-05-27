"""
A/B/C walk-forward comparison: baseline vs. latent-enriched features.

Config A — Baseline: 31 features → XGBoost walk-forward
Config B — Unsupervised AE (16-dim): 31+16=47 features → XGBoost
Config C — Supervised encoder (8-dim): 31+8=39 features → XGBoost  ← main test

Usage:
    # Full comparison (requires all latent parquets built)
    conda run -n envmt5 python scripts/compare_latent_vs_baseline.py

    # Only A vs C (skip old unsupervised Config B)
    conda run -n envmt5 python scripts/compare_latent_vs_baseline.py --skip-b
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

from src.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig
from src.walk_forward import WalkForwardConfig, WalkForwardValidator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_BASE    = ROOT / "data" / "features" / "EURUSD_M15_features.parquet"
FEATURES_LATENT  = ROOT / "data" / "features" / "EURUSD_M15_features_latent.parquet"
FEATURES_SUP8    = ROOT / "data" / "features" / "EURUSD_M15_features_latent_sup8.parquet"
LABELS           = ROOT / "data" / "features" / "EURUSD_M15_labels.parquet"
PRICES           = ROOT / "data" / "EURUSD_M15.csv"

# ---------------------------------------------------------------------------
# Shared backtester config (Phase 8 winner: confidence-tiered risk)
# ---------------------------------------------------------------------------

BACKTEST_CFG = BacktestConfig(
    threshold         = 0.40,
    sl_pips           = 30.0,
    tp_pips           = 60.0,
    pip_size          = 0.0001,
    spread_pips       = 1.0,
    commission_pips   = 0.0,
    max_slippage_pips = 0.0,
    initial_balance   = 10_000.0,
    risk_pct          = 0.01,
    use_regime_filter = False,
    risk_manager      = RiskManager(RiskConfig()),
)

WF_CFG_BASE = dict(
    model_type   = "xgboost",
    window_type  = "expanding",
    train_days   = 180,
    test_days    = 30,
    backtest     = BACKTEST_CFG,
)


def _run(X: pd.DataFrame, y: pd.Series, prices: pd.DataFrame, label: str,
         cache_dir: str = "data/models/wf_cache"):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features: {X.shape[1]} columns")
    print(f"{'=' * 60}")
    cfg = WalkForwardConfig(**WF_CFG_BASE, cache_dir=cache_dir)
    result = WalkForwardValidator(verbose=True).run(X, y, prices, cfg)
    result.report(title=label)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-b", action="store_true",
                   help="Skip Config B (old unsupervised 16-dim latent)")
    args = p.parse_args()

    # ---- Pre-flight checks ----
    missing = []
    if not FEATURES_SUP8.exists():
        missing.append(f"  {FEATURES_SUP8.name}  — run: python scripts/build_latent_features.py --suffix sup8")
    if not args.skip_b and not FEATURES_LATENT.exists():
        print(f"  NOTE: {FEATURES_LATENT.name} not found — Config B will be skipped (use --skip-b to silence)")
        args.skip_b = True

    if missing:
        print("ERROR: Required files not found:")
        for m in missing:
            print(m)
        print("\nTo generate supervised features:")
        print("  1. conda run -n envmt5 python scripts/train_autoencoder.py --mode supervised --latent-dim 8")
        print("  2. conda run -n envmt5 python scripts/build_latent_features.py --suffix sup8")
        sys.exit(1)

    # ---- Load shared data ----
    print("Loading data...")
    y      = pd.read_parquet(LABELS)["label"]
    prices = pd.read_csv(PRICES, index_col="time")
    prices.index = pd.to_datetime(prices.index)

    X_base = pd.read_parquet(FEATURES_BASE)
    X_sup8 = pd.read_parquet(FEATURES_SUP8)

    print(f"  Base features      : {X_base.shape}")
    print(f"  Supervised 8-dim   : {X_sup8.shape}")
    print(f"  Prices             : {len(prices):,} bars")

    y_base = y.loc[y.index.isin(X_base.index)]
    y_sup8 = y.loc[y.index.isin(X_sup8.index)]

    # ---- Run configs ----
    results = {}

    results["A"] = _run(X_base, y_base, prices, "A — Baseline (31 features)",
                        cache_dir="data/models/wf_cache")

    if not args.skip_b:
        X_latent = pd.read_parquet(FEATURES_LATENT)
        y_latent = y.loc[y.index.isin(X_latent.index)]
        print(f"  Unsupervised 16-dim: {X_latent.shape}")
        results["B"] = _run(X_latent, y_latent, prices,
                            f"B — Unsupervised AE ({X_latent.shape[1]} features)",
                            cache_dir="data/models/wf_cache_latent")

    results["C"] = _run(X_sup8, y_sup8, prices,
                        f"C — Supervised enc 8-dim ({X_sup8.shape[1]} features)",
                        cache_dir="data/models/wf_cache_sup8")

    # ---- Summary table ----
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  LATENT ENCODER A/B/C COMPARISON" + " " * 34 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"  {'Config':<36} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-" * 63)

    config_labels = {
        "A": f"A Baseline (31 feat)",
        "B": f"B Unsupervised AE ({X_base.shape[1]+16} feat)",
        "C": f"C Supervised enc 8-dim ({X_sup8.shape[1]} feat)",
    }

    for key, res in results.items():
        eq  = res.equity
        ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
        label = config_labels[key]
        print(f"  {label:<36} {res.sharpe:>+7.2f} {res.drawdown:>6.1f}% "
              f"{ret:>+8.1%} {len(res.trades):>7}")

    print()
    res_a = results["A"]
    res_c = results["C"]
    sharpe_delta = res_c.sharpe - res_a.sharpe
    dd_delta     = res_c.drawdown - res_a.drawdown

    if sharpe_delta > 0.05:
        print(f"  ✓ Supervised latent IMPROVED Sharpe by {sharpe_delta:+.2f}")
    elif sharpe_delta < -0.05:
        print(f"  ✗ Supervised latent HURT Sharpe by {sharpe_delta:.2f} — "
              f"try latent_dim=4 or more epochs")
    else:
        print(f"  ~ Supervised latent neutral on Sharpe ({sharpe_delta:+.2f}) — "
              f"check drawdown and return for full picture")

    if dd_delta < -0.01:
        print(f"  ✓ Max drawdown REDUCED by {abs(dd_delta):.1f}%")
    elif dd_delta > 0.01:
        print(f"  ~ Max drawdown increased by {dd_delta:.1f}%")
    print()
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
