"""
XGBoost + enc8 on M30 vs M15 champion.

M30 sits between M15 (best) and H1 (weaker). Two label configs tested:

  Config A — "Same as M15" (direct comparison)
    label_horizon=4 bars (2 hours on M30), threshold=0.0003 (3 pips)
    SL=30p / TP=60p — identical to M15 champion setup

  Config B — "M30-calibrated" (time-matched + wider threshold)
    label_horizon=2 bars (1 hour — same wall-clock as M15's 4×15min=1h)
    threshold=0.0003 (3 pips), SL=30p / TP=60p

Both use the same May 2024 → May 2026 date range as the M15 champion.

Usage
-----
  conda run -n envmt5 python scripts/compare_m30.py
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

from src.pipeline import PredictorPipeline, PipelineConfig
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

FILTER_START = pd.Timestamp("2024-05-14")

M15_CHAMPION = dict(
    label="M15  XGBoost+enc8  horizon=4  3pip  30p/60p  [CHAMPION]",
    sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
)

CONFIGS = [
    dict(
        tag       = "m30_same",
        label     = "M30  same-as-M15     horizon=4 (2h)  3pip",
        horizon   = 4,
        threshold = 0.0003,
    ),
    dict(
        tag       = "m30_timematch",
        label     = "M30  time-matched    horizon=2 (1h)  3pip",
        horizon   = 2,
        threshold = 0.0003,
    ),
]

STOP_VARIANTS = [
    (30.0, 60.0,  "SL=30p  TP=60p  "),
    (40.0, 80.0,  "SL=40p  TP=80p  "),
]


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


def _run_config(cfg_dict: dict, df_m30: pd.DataFrame,
                prices: pd.DataFrame, full_cfg: dict) -> list[dict]:
    """Build features once, then test 2 stop variants with shared fold cache."""
    tag       = cfg_dict["tag"]
    label     = cfg_dict["label"]
    horizon   = cfg_dict["horizon"]
    threshold = cfg_dict["threshold"]

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  horizon={horizon} bars ({horizon*30}min)  threshold={threshold} ({threshold*10000:.0f}pip)")
    print(f"{'='*64}")

    # Build features once with this label config
    cfg_build = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg_build.model_type         = "xgboost"
    cfg_build.encoder_mode       = "supervised"
    cfg_build.encoder_latent_dim = 8
    cfg_build.encoder_epochs     = 30
    cfg_build.label_horizon      = horizon
    cfg_build.label_threshold    = threshold

    t0   = time.time()
    pipe = PredictorPipeline(cfg_build)
    X, y = pipe.build_features(df_m30)
    print(f"\n  Features: {X.shape}  "
          f"buy={int((y==1).sum())}  hold={int((y==0).sum())}  sell={int((y==-1).sum())}")
    print(f"  Build time: {(time.time()-t0)/60:.1f} min")

    rm          = RiskManager(RiskConfig())
    cache_dir   = str(ROOT / f"data/models/wf_cache_m30_{tag}")
    results     = []

    for sl, tp, stop_label in STOP_VARIANTS:
        print(f"\n  --- {stop_label} ---")
        t0 = time.time()
        bt_cfg = BacktestConfig(
            initial_balance=10_000.0, sl_pips=sl, tp_pips=tp,
            spread_pips=1.0, risk_pct=0.01, max_slippage_pips=0.0,
            use_regime_filter=False, risk_manager=rm,
        )
        wf_cfg = WalkForwardConfig(
            model_type="xgboost", window_type="expanding",
            train_days=180, test_days=30,
            backtest=bt_cfg, cache_dir=cache_dir,
        )
        result  = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)
        eq      = result.equity
        ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
        elapsed = time.time() - t0

        print(f"\n  Sharpe : {result.sharpe:+.2f}")
        print(f"  MaxDD  : {result.drawdown:.1f}%")
        print(f"  Return : {ret:+.1%}")
        print(f"  Trades : {len(result.trades)}")
        print(f"  Time   : {elapsed/60:.1f} min")

        results.append(dict(
            label   = f"{label[:36]}  {stop_label.strip()}",
            sharpe  = result.sharpe,
            maxdd   = result.drawdown,
            ret     = ret,
            trades  = len(result.trades),
        ))

    return results


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    df_m30  = _load_raw(str(ROOT / "data/EURUSD_M30.csv"))
    df_m30  = df_m30[df_m30.index >= FILTER_START].copy()
    prices  = _load_prices(str(ROOT / full_cfg["pipeline"]["data_path"]))

    print(f"\nM30 dataset : {len(df_m30):,} bars "
          f"({df_m30.index[0].date()} → {df_m30.index[-1].date()})")
    print(f"M15 dataset : ~49,892 bars  (reference)")

    all_results = []
    for cfg_dict in CONFIGS:
        all_results.extend(_run_config(cfg_dict, df_m30, prices, full_cfg))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 88
    print(f"\n\n{'='*W}")
    print(f"  M30 vs M15 — XGBoost + enc8, May 2024 → May 2026")
    print(f"{'='*W}")
    print(f"  {'Config':<54} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print(f"  {'-'*(W-4)}")

    c = M15_CHAMPION
    print(f"  {c['label']:<54} {c['sharpe']:>+7.2f} {c['maxdd']:>6.1f}% "
          f"{c['ret']:>+7.1%} {c['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    best = max(all_results, key=lambda r: r["sharpe"])
    for r in all_results:
        delta = r["sharpe"] - c["sharpe"]
        flag  = "  ✓ BEATS M15!" if delta > 0 else f"  ({delta:+.2f})"
        if r is best:
            flag += "  ← BEST M30"
        print(f"  {r['label']:<54} {r['sharpe']:>+7.2f} {r['maxdd']:>6.1f}% "
              f"{r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    b = best
    delta = b["sharpe"] - c["sharpe"]
    if delta > 0:
        print(f"  M30 BEATS M15: {b['sharpe']:+.2f} vs +{c['sharpe']:.2f}  — update config.yaml!")
    else:
        print(f"  Best M30: {b['sharpe']:+.2f} / {b['maxdd']:.1f}% MaxDD"
              f"  vs M15 {c['sharpe']:+.2f}  ({delta:+.2f})")
        print(f"  M15 remains the deployment timeframe.")
    print()


if __name__ == "__main__":
    main()
