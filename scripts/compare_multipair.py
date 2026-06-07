"""
Multi-pair champion comparison — Phase 28.

Runs the champion config (XGBoost + supervised enc8, 39 features) independently
on EURUSD, GBPUSD, USDJPY, XAUUSD — same date range, same hyperparameters.

Each pair is completely isolated:
  - enc8 trained ONLY on that pair's raw OHLCV
  - XGBoost trained ONLY on that pair's feature matrix
  - Separate wf_cache directory per pair
  - No data ever crosses between pairs

Results show which pairs generalise the champion's edge before paper trading.

Usage:
    conda run -n envmt5 --no-capture-output python scripts/compare_multipair.py \\
        > logs/multipair_compare.log 2>&1 &
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
from src.evaluation.walk_forward import WalkForwardValidator, WalkForwardConfig
from src.evaluation.backtester import BacktestConfig

# ── Common date range — all pairs start 2024-01-08 ────────────────────────────
COMMON_START = pd.Timestamp("2024-01-08")

# ── Pair-specific configs ─────────────────────────────────────────────────────
#   pip_size  : price per pip (instrument-specific, never mixed)
#   sl / tp   : in pips for that instrument (scaled to give comparable % risk)
#   spread    : typical spread in pips for that instrument
#
#   EURUSD/GBPUSD : pip_size=0.0001, sl=30p → $0.003 stop (0.28% at 1.08)
#   USDJPY        : pip_size=0.01,   sl=30p → 0.30 yen   (0.20% at 148)
#   XAUUSD        : pip_size=0.01,   sl=300p→ $3.00 stop  (0.15% at 2000)
PAIRS = [
    dict(symbol="EURUSD", csv="data/EURUSD_M15.csv",
         pip_size=0.0001, sl=30,  tp=60,  spread=1.0,  commission=0.5),
    dict(symbol="GBPUSD", csv="data/GBPUSD_M15.csv",
         pip_size=0.0001, sl=30,  tp=60,  spread=2.0,  commission=0.5),
    dict(symbol="USDJPY", csv="data/USDJPY_M15.csv",
         pip_size=0.01,   sl=30,  tp=60,  spread=1.0,  commission=0.5),
    dict(symbol="XAUUSD", csv="data/XAUUSD_M15.csv",
         pip_size=0.01,   sl=300, tp=600, spread=30.0, commission=5.0),
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_df(path: str) -> pd.DataFrame:
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


# ── Champion PipelineConfig (pair-agnostic hyperparameters) ───────────────────

def _make_cfg(symbol: str, full_cfg: dict) -> PipelineConfig:
    """Champion config — model and encoder settings identical for every pair."""
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = False
    # Separate cache per pair — models never shared
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_multipair_{symbol}")
    return cfg


# ── Per-pair walk-forward ─────────────────────────────────────────────────────

def run_pair(pair: dict, full_cfg: dict) -> dict:
    symbol  = pair["symbol"]
    csv     = ROOT / pair["csv"]

    print(f"\n{'='*72}")
    print(f"  {symbol}  —  Champion XGBoost + enc8 (39 features)")
    print(f"{'='*72}")

    # Load and trim to common date range
    df_all  = _load_df(str(csv))
    df      = df_all[df_all.index >= COMMON_START].copy()
    prices  = _load_prices(str(csv))
    prices  = prices[prices.index >= COMMON_START]

    print(f"  Bars : {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Pip  : {pair['pip_size']}  SL={pair['sl']}p  TP={pair['tp']}p  "
          f"Spread={pair['spread']}p", flush=True)

    t0 = time.time()

    # Build features — enc8 trained exclusively on THIS pair's OHLCV
    cfg  = _make_cfg(symbol, full_cfg)
    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df)
    n_feat = X.shape[1]

    label_counts = y.value_counts().to_dict()
    print(f"  Features : {n_feat}   Labels : {label_counts}", flush=True)

    # Walk-forward with pair-specific BacktestConfig
    bt_cfg = BacktestConfig(
        threshold         = 0.40,
        pip_size          = pair["pip_size"],
        sl_pips           = pair["sl"],
        tp_pips           = pair["tp"],
        spread_pips       = pair["spread"],
        commission_pips   = pair["commission"],
        initial_balance   = 10_000.0,
        risk_pct          = 0.01,
        use_regime_filter = False,
    )
    wf_cfg = WalkForwardConfig(
        model_type  = "xgboost",
        window_type = "expanding",
        train_days  = 180,
        test_days   = 30,
        backtest    = bt_cfg,
        cache_dir   = cfg.wf_cache_dir,
    )

    r       = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)
    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe  : {r.sharpe:+.2f}")
    print(f"  MaxDD   : {r.drawdown:.1f}%")
    print(f"  Return  : {ret:+.1%}")
    print(f"  Trades  : {len(r.trades)}")
    print(f"  Time    : {elapsed/60:.1f} min")

    return dict(
        symbol  = symbol,
        n_feat  = n_feat,
        sharpe  = r.sharpe,
        maxdd   = r.drawdown,
        ret     = ret,
        trades  = len(r.trades),
        elapsed = elapsed,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    print(f"\nPhase 28 — Multi-Pair Champion Validation")
    print(f"Config  : XGBoost + supervised enc8  latent_dim=8  epochs=30  39 features")
    print(f"Period  : {COMMON_START.date()} → 2026-06-05  (same for all pairs)")
    print(f"WF      : expanding  180d train / 30d test")
    print(f"\nEach pair's enc8 and XGBoost are trained exclusively on that pair's data.")
    print(f"No data is shared between pairs.\n")

    results = []
    for pair in PAIRS:
        try:
            results.append(run_pair(pair, full_cfg))
        except Exception as e:
            print(f"\n  ERROR on {pair['symbol']}: {e}")
            results.append(dict(symbol=pair["symbol"], n_feat=0,
                                sharpe=float("nan"), maxdd=float("nan"),
                                ret=float("nan"), trades=0, elapsed=0))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 80
    print(f"\n\n{'='*W}")
    print(f"  Phase 28: Multi-Pair Champion Results — XGBoost + enc8, M15")
    print(f"{'='*W}")
    hdr = f"  {'Pair':<8} {'Feat':>4} {'Sharpe':>8} {'MaxDD':>7} {'Return':>9} {'Trades':>7} {'Time':>6}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    for r in results:
        if r["n_feat"] == 0:
            print(f"  {r['symbol']:<8}  ERROR")
            continue
        verdict = ""
        if r["sharpe"] > 1.5:
            verdict = "  ✓ DEPLOY"
        elif r["sharpe"] > 0.5:
            verdict = "  ~ MARGINAL"
        else:
            verdict = "  ✗ SKIP"
        print(
            f"  {r['symbol']:<8} {r['n_feat']:>4} {r['sharpe']:>+8.2f} "
            f"{r['maxdd']:>6.1f}% {r['ret']:>+8.1%} {r['trades']:>7} "
            f"{r['elapsed']/60:>5.1f}m{verdict}"
        )

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}")

    deploy = [r for r in results if r.get("sharpe", 0) > 1.5]
    skip   = [r for r in results if r.get("sharpe", 0) <= 0.5]

    print(f"\n  Deploy for paper trading : {[r['symbol'] for r in deploy] or 'none'}")
    print(f"  Skip                     : {[r['symbol'] for r in skip] or 'none'}")
    print(f"\n  Next: run retrain_champion.py for each pair marked DEPLOY,")
    print(f"        then start pipeline_bot.py with pair-specific config.\n")


if __name__ == "__main__":
    main()
