"""
Stack (XGBoost + CatBoost, LightGBM meta) on two datasets.

Run 1 — 49k dataset (May 2024 → May 2026):
  Same data as Phase 9 (+3.13). Does stacking beat XGBoost alone?

Run 2 — 75k dataset (May 2023 → May 2026):
  Full history. Phase 10 Config D already ran this (+1.22) — cache reused.

Both use enc8 (31 base + 8 latent = 39 features), same as Phase 9.

Usage
-----
  conda run -n envmt5 python scripts/compare_stack_49k_75k.py
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
import yaml

from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

# ── Shared walk-forward / backtest config ─────────────────────────────────────
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

WF_BASE = dict(
    model_type  = "ensemble_stack",   # XGBoost + CatBoost → LightGBM meta, 3-fold OOF
    window_type = "expanding",
    train_days  = 180,
    test_days   = 30,
    backtest    = BACKTEST_CFG,
)

PHASE9_XGB  = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524)   # Phase 9 XGBoost+enc8
PHASE10_STK = dict(sharpe=1.22, maxdd=16.6, ret=1.009, trades=748)   # Phase 10 stack on 75k


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prices(csv_path: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _run_wf(label: str, X: pd.DataFrame, y: pd.Series,
            prices: pd.DataFrame, cache_dir: str) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  Features: {X.shape}   Model: ensemble_stack")
    print(f"  Dataset:  {X.index[0].date()} → {X.index[-1].date()}")
    print(f"{'='*64}")
    t0 = time.time()

    cfg = WalkForwardConfig(**WF_BASE, cache_dir=cache_dir)
    result = WalkForwardValidator(verbose=True).run(X, y, prices, cfg)

    eq  = result.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, sharpe=result.sharpe, maxdd=result.drawdown,
                ret=ret, trades=len(result.trades))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg.get("pipeline", {}).get("data_path", "data/EURUSD_M15.csv")
    prices   = _load_prices(csv_path)

    results = []

    # ── Run 1: stack on 49k (May 2024+) ── uses pre-built parquet from Phase 9
    print("\n" + "="*64)
    print("  RUN 1 — 49k dataset  (same as Phase 9, direct comparison to +3.13)")
    print("="*64)
    X_49k = pd.read_parquet(ROOT / "data/features/EURUSD_M15_features_latent_sup8.parquet")
    y_49k = pd.read_parquet(ROOT / "data/features/EURUSD_M15_labels.parquet")["label"]
    y_49k = y_49k.loc[y_49k.index.isin(X_49k.index)]

    results.append(_run_wf(
        "Stack — 49k (May 2024+)  enc8  39 feat",
        X_49k, y_49k, prices,
        cache_dir=str(ROOT / "data/models/wf_cache_stack_49k"),
    ))

    # ── Run 2: stack on 75k (full history) ── reuses Phase 10 cache if keys match
    print("\n" + "="*64)
    print("  RUN 2 — 75k dataset  (full history, Phase 10 cache reused if possible)")
    print("="*64)
    from src.pipeline import PredictorPipeline, PipelineConfig
    pipe_cfg = PipelineConfig.from_dict(cfg.get("pipeline", {}),
                                        rm_cfg=cfg.get("risk_manager", {}))
    pipe_cfg.model_type         = "ensemble_stack"
    pipe_cfg.encoder_latent_dim = 8
    pipe_cfg.encoder_epochs     = 30
    pipe_cfg.wf_cache_dir       = "/tmp/ens_compare_cache/stack"  # Phase 10 cache

    pipe = PredictorPipeline(pipe_cfg)

    df_raw = pd.read_csv(csv_path)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    time_col = next(c for c in df_raw.columns if "time" in c)
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    df_raw = df_raw.set_index(time_col).sort_index()

    print("  Building enc8 features on full 75k dataset...")
    X_75k, y_75k = pipe.build_features(df_raw)

    # Walk-forward via pipeline (will reuse Phase 10 cache for all 31 folds)
    print(f"\n  Feature matrix: {X_75k.shape}   ({X_75k.index[0].date()} → {X_75k.index[-1].date()})")
    t0 = time.time()
    r75 = pipe.walk_forward(X_75k, y_75k, prices)
    eq  = r75.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r75.sharpe:+.2f}")
    print(f"  MaxDD  : {r75.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r75.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")
    results.append(dict(label="Stack — 75k (full)       enc8  39 feat",
                        sharpe=r75.sharpe, maxdd=r75.drawdown,
                        ret=ret, trades=len(r75.trades)))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*78)
    print("  STACK COMPARISON — enc8 (39 features), XGBoost + CatBoost → LGB meta")
    print("="*78)
    print(f"  {'Config':<42} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-"*72)

    # Reference rows
    print(f"  {'Phase 9  XGBoost alone  49k  enc8 [REFERENCE]':<42} "
          f"{PHASE9_XGB['sharpe']:>+7.2f} {PHASE9_XGB['maxdd']:>6.1f}% "
          f"{PHASE9_XGB['ret']:>+7.1%} {PHASE9_XGB['trades']:>7}")
    print(f"  {'Phase 10 XGBoost alone  75k  enc8 [REFERENCE]':<42} "
          f"{1.12:>+7.2f} {22.5:>6.1f}% {0.599:>+7.1%} {753:>7}")

    print("  " + "-"*72)

    for r in results:
        beats49 = "  ✓ BEATS +3.13" if r["sharpe"] > 3.13 else ""
        print(f"  {r['label']:<42} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>6.1f}% "
              f"{r['ret']:>+7.1%} {r['trades']:>7}{beats49}")

    print("  " + "-"*72)
    print("="*78)
    print()

    r49 = results[0]
    r75 = results[1]

    print("  KEY FINDINGS:")
    delta49 = r49["sharpe"] - PHASE9_XGB["sharpe"]
    print(f"  49k stack vs XGBoost alone: {r49['sharpe']:+.2f} vs +3.13  ({delta49:+.2f})")
    delta75 = r75["sharpe"] - 1.12
    print(f"  75k stack vs XGBoost alone: {r75['sharpe']:+.2f} vs +1.12  ({delta75:+.2f})")
    print()


if __name__ == "__main__":
    main()
