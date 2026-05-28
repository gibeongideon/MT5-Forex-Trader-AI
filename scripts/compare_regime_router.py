"""
RegimeRouter vs XGBoost — enc8 features, 49k M15 dataset.

Tests whether routing signals through regime-specialist XGBoost models
beats the single XGBoost champion (+3.13 Sharpe, 13.3% MaxDD).

RegimeRouter:
  - Detects 4 regimes via KMeans on [ATR-ratio, ADX, RSI]
  - Trains one specialist XGBoost per regime
  - Routes each bar's prediction to its regime specialist

Uses pre-built enc8 parquet — no encoder retraining needed.

Usage
-----
  conda run -n envmt5 python scripts/compare_regime_router.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

BACKTEST_CFG = BacktestConfig(
    initial_balance   = 10_000.0,
    sl_pips           = 30.0,
    tp_pips           = 60.0,
    spread_pips       = 1.0,
    risk_pct          = 0.01,
    max_slippage_pips = 0.0,
    use_regime_filter = False,
    risk_manager      = RiskManager(RiskConfig()),
)

XGB_BASELINE = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                    label="XGBoost          enc8  49k  [CHAMPION]")


def _load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _run(label: str, model_type: str, X: pd.DataFrame,
         y: pd.Series, prices: pd.DataFrame, cache_dir: str) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  Dataset: {X.index[0].date()} → {X.index[-1].date()}  ({X.shape[1]} features)")
    print(f"{'='*64}")
    t0 = time.time()

    cfg    = WalkForwardConfig(
        model_type  = model_type,
        window_type = "expanding",
        train_days  = 180,
        test_days   = 30,
        backtest    = BACKTEST_CFG,
        cache_dir   = cache_dir,
    )
    result = WalkForwardValidator(verbose=True).run(X, y, prices, cfg)

    eq      = result.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, model=model_type,
                sharpe=result.sharpe, maxdd=result.drawdown,
                ret=ret, trades=len(result.trades))


def main() -> None:
    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg.get("pipeline", {}).get("data_path", "data/EURUSD_M15.csv")
    prices   = _load_prices(csv_path)

    print("\nLoading pre-built enc8 features (39 cols, 49k rows) …")
    X = pd.read_parquet(ROOT / "data/features/EURUSD_M15_features_latent_sup8.parquet")
    y_full = pd.read_parquet(ROOT / "data/features/EURUSD_M15_labels.parquet")["label"]
    y = y_full.loc[y_full.index.isin(X.index)]
    print(f"  {X.shape}  ({X.index[0].date()} → {X.index[-1].date()})")

    result = _run(
        label      = "RegimeRouter     enc8  49k  4 regimes",
        model_type = "regime_router",
        X          = X,
        y          = y,
        prices     = prices,
        cache_dir  = str(ROOT / "data/models/wf_cache_regime_router"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*74)
    print("  RegimeRouter vs XGBoost — enc8 (39 features), 49k rows, M15")
    print("="*74)
    print(f"  {'Model':<44} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-"*68)

    b = XGB_BASELINE
    print(f"  {b['label']:<44} {b['sharpe']:>+7.2f} {b['maxdd']:>6.1f}% "
          f"{b['ret']:>+7.1%} {b['trades']:>7}")
    print("  " + "-"*68)

    delta = result["sharpe"] - b["sharpe"]
    flag  = "  ✓ BEATS XGBoost!" if delta > 0 else f"  ({delta:+.2f} vs XGBoost)"
    print(f"  {result['label']:<44} {result['sharpe']:>+7.2f} {result['maxdd']:>6.1f}% "
          f"{result['ret']:>+7.1%} {result['trades']:>7}{flag}")

    print("  " + "-"*68)
    print("="*74)
    print()

    if delta > 0:
        print(f"  RESULT: RegimeRouter BEATS XGBoost: {result['sharpe']:+.2f} vs +{b['sharpe']:.2f}")
        print(f"  ACTION: Update config.yaml model_type: regime_router")
    else:
        print(f"  RESULT: XGBoost holds: +{b['sharpe']:.2f} vs {result['sharpe']:+.2f}")
        print(f"  ACTION: Keep config.yaml model_type: xgboost (default).")
    print()


if __name__ == "__main__":
    main()
