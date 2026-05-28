"""
Walk-Forward Online Adaptation — Technique #3 from IMPROVEMENT.MD.

Tests whether more frequent model retraining improves out-of-sample performance
compared to the standard 30-day test fold.

Current champion: train=180d / test=30d → Sharpe +3.13, MaxDD 13.3%

The "adaptation" is implemented by shortening test_days: the model gets retrained
at every fold boundary using data up to that point. Shorter test windows = the
model adapts to regime changes faster within the 2-year test period.

Configs tested (all use train_days=180, same enc8 39-feature parquet):
  A  test_days=30   standard (baseline +3.13)          ~19 folds
  B  test_days=14   bi-weekly retraining               ~38 folds
  C  test_days=7    weekly retraining                  ~76 folds
  D  train_days=90  test_days=7  shorter window        ~76 folds

All use the pre-built enc8 parquet — no encoder retraining. Fast runtime.

Usage
-----
  conda run -n envmt5 python scripts/compare_adaptive_wf.py
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

BASELINE = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="XGBoost enc8  train=180d  test=30d  [CHAMPION]")

CONFIGS = [
    dict(tag="30d",  train=180, test=30,  label="train=180d  test=30d  (standard)"),
    dict(tag="14d",  train=180, test=14,  label="train=180d  test=14d  (bi-weekly)"),
    dict(tag="7d",   train=180, test=7,   label="train=180d  test= 7d  (weekly)"),
    dict(tag="90_7d",train= 90, test=7,   label="train= 90d  test= 7d  (short window+weekly)"),
]


def _load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _run(cfg_dict: dict, X: pd.DataFrame, y: pd.Series,
         prices: pd.DataFrame) -> dict:
    tag   = cfg_dict["tag"]
    label = cfg_dict["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()

    wf_cfg = WalkForwardConfig(
        model_type  = "xgboost",
        window_type = "expanding",
        train_days  = cfg_dict["train"],
        test_days   = cfg_dict["test"],
        backtest    = BACKTEST_CFG,
        cache_dir   = str(ROOT / f"data/models/wf_cache_adapt_{tag}"),
    )
    result = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)

    eq      = result.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Folds  : {len(result.folds)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, tag=tag,
                sharpe=result.sharpe, maxdd=result.drawdown,
                ret=ret, trades=len(result.trades), folds=len(result.folds))


def main() -> None:
    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg["pipeline"]["data_path"]
    prices   = _load_prices(csv_path)

    print("\nLoading pre-built enc8 features …")
    X = pd.read_parquet(ROOT / "data/features/EURUSD_M15_features_latent_sup8.parquet")
    y_full = pd.read_parquet(ROOT / "data/features/EURUSD_M15_labels.parquet")["label"]
    y = y_full.loc[y_full.index.isin(X.index)]
    print(f"  {X.shape}  ({X.index[0].date()} → {X.index[-1].date()})")

    results = []
    for cfg_dict in CONFIGS:
        results.append(_run(cfg_dict, X, y, prices))

    # ── Summary table ─────────────────────────────────────────────────────
    W = 80
    print(f"\n\n{'='*W}")
    print(f"  Walk-Forward Adaptation — enc8 49k M15 (May 2024+)")
    print(f"  Test: does more frequent retraining beat +3.13 champion?")
    print(f"{'='*W}")
    print(f"  {'Config':<44} {'Folds':>5} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print(f"  {'-'*(W-4)}")

    b = BASELINE
    print(f"  {b['label']:<44} {'19':>5} {b['sharpe']:>+7.2f} {b['maxdd']:>6.1f}% "
          f"{b['ret']:>+7.1%} {b['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    best = max(results, key=lambda r: r["sharpe"])
    for r in results:
        delta = r["sharpe"] - b["sharpe"]
        flag  = "  ✓ BEATS!" if delta > 0 else f"  ({delta:+.2f})"
        if r is best:
            flag += "  ← BEST"
        print(f"  {r['label']:<44} {r['folds']:>5} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    best_r = best
    delta  = best_r["sharpe"] - b["sharpe"]
    if delta > 0:
        print(f"  RESULT: Adaptation helps! Best: {best_r['sharpe']:+.2f} "
              f"({best_r['label'].strip()})")
        print(f"  Gains {delta:+.2f} Sharpe from more frequent retraining.")
    else:
        print(f"  RESULT: Standard 30d retraining holds as champion (+{b['sharpe']:.2f}).")
        print(f"  More frequent retraining does NOT improve Sharpe on this dataset.")
        if best_r["maxdd"] < b["maxdd"]:
            print(f"  However, {best_r['label'].strip()} has lower MaxDD "
                  f"({best_r['maxdd']:.1f}% vs {b['maxdd']:.1f}%).")
    print()


if __name__ == "__main__":
    main()
