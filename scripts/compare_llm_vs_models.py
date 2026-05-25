"""
Compare LLM signal vs XGBoost vs Ensemble on the cached period.

Runs the backtester directly (no retraining) on the period covered by the
LLM cache. Prints a side-by-side Sharpe / return / drawdown table.

Usage:
    conda activate envmt5
    python scripts/compare_llm_vs_models.py
    python scripts/compare_llm_vs_models.py --threshold 0.45
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.backtester import Backtester, BacktestConfig


def load_prices(path: str = "data/EURUSD_M15.csv") -> pd.DataFrame:
    prices = pd.read_csv(path, index_col="time", parse_dates=True)
    return prices



class _ProbaWrapper:
    """Minimal ModelInterface shim that serves a pre-computed probability array."""
    def __init__(self, probas: np.ndarray):
        self._probas = probas

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probas

    def train(self, *a, **kw): return self
    def save(self, *a, **kw): pass
    def load(self, *a, **kw): return self
    def metadata(self): return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--prices",    default="data/EURUSD_M15.csv")
    p.add_argument("--cache",     default="data/models/llm_cache.parquet")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--models",    nargs="+", default=["xgboost", "ensemble", "llm_signal"],
                   help="Models to compare")
    args = p.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    X = pd.read_parquet(args.features)
    prices = load_prices(args.prices)

    # Restrict to cached period so all models run on the same window
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: LLM cache not found at {args.cache}")
        print("Run scripts/precompute_llm_signals.py first.")
        sys.exit(1)

    cache = pd.read_parquet(cache_path)
    cache_start = cache.index.min()
    cache_end   = cache.index.max()
    print(f"Cache covers {cache_start.date()} → {cache_end.date()} ({len(cache):,} bars)")

    # Align all data to the cached window
    X      = X.loc[cache_start:cache_end]
    prices = prices.loc[cache_start:cache_end]
    print(f"Comparison window: {len(X):,} bars\n")

    cfg = BacktestConfig(
        threshold       = args.threshold,
        sl_pips         = 30.0,
        tp_pips         = 60.0,
        pip_size        = 0.0001,
        spread_pips     = 1.0,
        commission_pips = 0.5,
        initial_balance = 10_000.0,
        risk_pct        = 0.01,
    )

    # ── Run each model ─────────────────────────────────────────────────────────
    results = {}
    from src.model_registry import ModelRegistry
    registry = ModelRegistry.from_config("config.yaml", auto_load=True)

    # Collect raw probabilities for blend models
    raw_probas: dict[str, np.ndarray] = {}

    for name in args.models + ["xgb+llm"]:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            if name == "xgb+llm":
                # Simple equal-weight blend of XGBoost and LLM signals
                if "xgboost" not in raw_probas or "llm_signal" not in raw_probas:
                    raise KeyError("need both xgboost and llm_signal results first")
                blended = (raw_probas["xgboost"] + raw_probas["llm_signal"]) / 2.0
                model = _ProbaWrapper(blended)
            else:
                if name not in registry:
                    raise KeyError(f"not in registry (artifact missing?)")
                model = registry.get(name)

            # Collect raw probas for blending before backtest
            if name in ("xgboost", "llm_signal"):
                p = model.predict_proba(X)
                if p.ndim == 1:
                    p = p.reshape(1, -1)
                raw_probas[name] = p

            # Run backtest
            bt = Backtester()
            result = bt.run(model, X, prices.loc[X.index], cfg)
            trades = result.trades
            equity = result.equity

            if len(trades) == 0:
                metrics = {"trades": 0, "sharpe": 0.0, "return_pct": 0.0, "drawdown": 0.0, "win_rate": 0.0}
            else:
                total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
                wins = sum(1 for t in trades if t.get("pnl_dollars", t.get("pnl", 0)) > 0)
                metrics = {
                    "trades":     len(trades),
                    "sharpe":     round(float(result.sharpe), 3),
                    "return_pct": round(float(total_return), 2),
                    "drawdown":   round(float(result.drawdown), 2),
                    "win_rate":   round(wins / len(trades) * 100, 1),
                }
            results[name] = metrics
            print(f"done  ({metrics['trades']} trades, Sharpe={metrics['sharpe']})")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = None

    # ── Print comparison table ─────────────────────────────────────────────────
    w = 74
    print(f"\n{'─' * w}")
    print(f"  {'Model':<15}  {'Trades':>7}  {'Sharpe':>8}  {'Return%':>9}  {'Drawdown%':>10}  {'WinRate%':>9}")
    print(f"{'─' * w}")

    best_sharpe = max((r["sharpe"] for r in results.values() if r), default=0)
    for name, r in results.items():
        if r is None:
            print(f"  {name:<15}  {'ERROR':>7}")
            continue
        marker = "  ★" if r["sharpe"] == best_sharpe and best_sharpe > 0 else ""
        print(
            f"  {name:<15}  {r['trades']:>7}  {r['sharpe']:>8.3f}  "
            f"{r['return_pct']:>9.2f}  {r['drawdown']:>10.2f}  {r['win_rate']:>9.1f}{marker}"
        )
    print(f"{'─' * w}")
    print(f"\nThreshold={args.threshold}  Period: {cache_start.date()} → {cache_end.date()}")

    # ── Blend delta ────────────────────────────────────────────────────────────
    if results.get("xgb+llm"):
        blend_sharpe = results["xgb+llm"]["sharpe"]
        for name in ("xgboost", "llm_signal", "ensemble"):
            r = results.get(name)
            if r:
                delta = blend_sharpe - r["sharpe"]
                print(f"  xgb+llm vs {name:<12}: Sharpe {delta:+.3f}")


if __name__ == "__main__":
    main()
