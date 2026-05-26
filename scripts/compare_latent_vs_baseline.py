"""
A/B walk-forward comparison: baseline features vs. latent-enriched features.

Runs the same XGBoost walk-forward on two feature sets and prints a side-by-side
table so you can see whether the autoencoder's latent features improve edge.

Usage:
    conda run -n envmt5 python scripts/compare_latent_vs_baseline.py

Requirements:
    1. Run scripts/train_autoencoder.py     (creates data/models/autoencoder.pt)
    2. Run scripts/build_latent_features.py (creates data/features/EURUSD_M15_features_latent.parquet)
    Then run this script.
"""

from __future__ import annotations

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

FEATURES_BASE   = ROOT / "data" / "features" / "EURUSD_M15_features.parquet"
FEATURES_LATENT = ROOT / "data" / "features" / "EURUSD_M15_features_latent.parquet"
LABELS          = ROOT / "data" / "features" / "EURUSD_M15_labels.parquet"
PRICES          = ROOT / "data" / "EURUSD_M15.csv"

# ---------------------------------------------------------------------------
# Shared backtester config (matches Phase 8 winner: confidence-tiered risk)
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
    risk_manager      = RiskManager(RiskConfig()),   # Phase 8 tiered risk
)

WF_CFG_BASE = dict(
    model_type   = "xgboost",
    window_type  = "expanding",
    train_days   = 180,
    test_days    = 30,
    backtest     = BACKTEST_CFG,
)


def _run(X: pd.DataFrame, y: pd.Series, prices: pd.DataFrame, label: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features: {X.shape[1]} columns")
    print(f"{'=' * 60}")
    cfg = WalkForwardConfig(**WF_CFG_BASE)
    result = WalkForwardValidator(verbose=True).run(X, y, prices, cfg)
    result.report(title=label)
    return result


def main() -> None:
    # ---- Pre-flight checks ----
    if not FEATURES_LATENT.exists():
        print(f"ERROR: Latent features not found at {FEATURES_LATENT}")
        print("\nRun these steps first:")
        print("  1. conda run -n envmt5 python scripts/train_autoencoder.py")
        print("  2. conda run -n envmt5 python scripts/build_latent_features.py")
        sys.exit(1)

    # ---- Load shared data ----
    print("Loading data...")
    y      = pd.read_parquet(LABELS)["label"]
    prices = pd.read_csv(PRICES, index_col="time")
    prices.index = pd.to_datetime(prices.index)

    X_base   = pd.read_parquet(FEATURES_BASE)
    X_latent = pd.read_parquet(FEATURES_LATENT)

    print(f"  Base features   : {X_base.shape}")
    print(f"  Latent features : {X_latent.shape}")
    print(f"  Prices          : {len(prices):,} bars")

    # ---- Align labels to each feature set ----
    y_base   = y.loc[y.index.isin(X_base.index)]
    y_latent = y.loc[y.index.isin(X_latent.index)]

    # ---- Run A/B ----
    res_a = _run(X_base,   y_base,   prices, "A — Baseline (31 features)")
    res_b = _run(X_latent, y_latent, prices,
                 f"B — Latent-enriched ({X_latent.shape[1]} features: "
                 f"31 base + {X_latent.shape[1] - X_base.shape[1]} latent)")

    # ---- Summary table ----
    print()
    print("╔" + "═" * 62 + "╗")
    print("║  LATENT ENCODER A/B COMPARISON" + " " * 31 + "║")
    print("╠" + "═" * 62 + "╣")
    print(f"  {'Config':<30} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-" * 57)

    for label, res in [
        (f"A Baseline (31 feat)",       res_a),
        (f"B Latent ({X_latent.shape[1]} feat)", res_b),
    ]:
        eq  = res.equity
        ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
        improvement = ""
        print(f"  {label:<30} {res.sharpe:>+7.2f} {res.drawdown:>7.1%} "
              f"{ret:>+8.1%} {len(res.trades):>7}")

    print()
    sharpe_delta = res_b.sharpe - res_a.sharpe
    dd_delta     = res_b.drawdown - res_a.drawdown
    if sharpe_delta > 0:
        print(f"  ✓ Latent features IMPROVED Sharpe by {sharpe_delta:+.2f}")
    elif sharpe_delta < -0.05:
        print(f"  ✗ Latent features HURT Sharpe by {sharpe_delta:.2f} — "
              f"consider reducing latent_dim or increasing window_size")
    else:
        print(f"  ~ Latent features neutral on Sharpe ({sharpe_delta:+.2f}) — "
              f"check drawdown and return for full picture")

    if dd_delta < -0.01:
        print(f"  ✓ Max drawdown REDUCED by {abs(dd_delta):.1%}")
    print()
    print("╚" + "═" * 62 + "╝")


if __name__ == "__main__":
    main()
