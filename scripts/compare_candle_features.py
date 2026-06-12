"""
compare_candle_features.py — Walk-forward comparison: with vs without candle features.

Tests whether adding OOS candle model predictions (candle_p_buy, candle_p_sell)
as features to the main XGBoost pipeline improves the WF OOS Sharpe above +3.13.

Prerequisites:
    conda run -n envmt5 python scripts/build_candle_features.py

Configs compared:
  A — Baseline:  XGBoost + enc8, no candle features  (champion +3.13 Sharpe)
  B — Candle:    XGBoost + enc8, + candle_p_buy + candle_p_sell  (+2 features)

Both use the same WF settings as the champion (expanding 180d/30d).

Usage:
    conda run -n envmt5 python scripts/compare_candle_features.py
    conda run -n envmt5 python scripts/compare_candle_features.py --symbol USDJPY
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig

BASELINE = dict(sharpe=3.13, maxdd=13.3, trades=524, label="Baseline  XGBoost+enc8  no candle feat")

FEAT_DIR = ROOT / "data" / "features"

# Default symbols to compare (can be overridden via CLI)
SYMBOL_DEFAULTS = {
    "EURUSD": dict(data_path="data/EURUSD_M15.csv", pip_size=0.0001),
    "USDJPY": dict(data_path="data/USDJPY_M15.csv", pip_size=0.01),
}


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


def _make_cfg(full_cfg: dict, cache_tag: str, pip_size: float = 0.0001) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = False
    cfg.bt_pip_size              = pip_size
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_candle_feat_{cache_tag}")
    return cfg


def _run_wf(label: str, cfg: PipelineConfig, df_raw: pd.DataFrame,
            prices: pd.DataFrame, candle_parquet: Path | None) -> dict:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    if candle_parquet:
        print(f"  Candle parquet: {candle_parquet.name}")
    print(f"{'='*70}")

    t0   = time.time()
    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)

    if candle_parquet and candle_parquet.exists():
        cf = pd.read_parquet(candle_parquet)[["candle_p_buy", "candle_p_sell"]]
        shared = X.index.intersection(cf.index)
        before = len(X)
        X = pd.concat([X.loc[shared], cf.loc[shared]], axis=1)
        y = y.reindex(shared)
        print(f"  Candle features injected: {len(shared):,}/{before:,} rows  "
              f"→ {X.shape[1]} total features")
    else:
        if candle_parquet:
            print(f"  WARNING: Candle parquet not found at {candle_parquet}")
            print(f"  Run: conda run -n envmt5 python scripts/build_candle_features.py")

    r = pipe.walk_forward(X, y, prices)

    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Feats  : {X.shape[1]}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, n_feat=X.shape[1],
                sharpe=r.sharpe, maxdd=r.drawdown,
                ret=ret, trades=len(r.trades))


def compare_symbol(symbol: str, full_cfg: dict) -> None:
    sym_cfg   = SYMBOL_DEFAULTS.get(symbol, {})
    data_path = sym_cfg.get("data_path", full_cfg["pipeline"]["data_path"])
    pip_size  = sym_cfg.get("pip_size", 0.0001)
    df_raw    = _load_raw(str(ROOT / data_path))
    prices    = _load_prices(str(ROOT / data_path))

    candle_path = FEAT_DIR / f"candle_signal_{symbol}.parquet"

    print(f"\n{'#'*70}")
    print(f"  CANDLE FEATURE INJECTION — {symbol}  (pip_size={pip_size})")
    print(f"{'#'*70}")

    cfg_a = _make_cfg(full_cfg, cache_tag=f"{symbol}_base",   pip_size=pip_size)
    cfg_b = _make_cfg(full_cfg, cache_tag=f"{symbol}_candle", pip_size=pip_size)

    results = []
    results.append(_run_wf(
        f"Config A  Baseline — {symbol}  no candle features",
        cfg_a, df_raw, prices, candle_parquet=None,
    ))
    results.append(_run_wf(
        f"Config B  + candle_p_buy/sell — {symbol}",
        cfg_b, df_raw, prices, candle_parquet=candle_path,
    ))

    # ── Summary ──────────────────────────────────────────────────────────────
    W = 78
    print(f"\n\n{'='*W}")
    print(f"  CANDLE FEATURE INJECTION COMPARISON — {symbol}")
    print(f"{'='*W}")
    print(f"  {'Config':<46} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print(f"  {'-'*(W-4)}")

    baseline_sharpe = results[0]["sharpe"]
    for r in results:
        delta = r["sharpe"] - baseline_sharpe if r is not results[0] else 0
        flag  = ""
        if r is not results[0]:
            flag = f"  (+{delta:+.2f})" if delta > 0 else f"  ({delta:+.2f})"
        print(f"  {r['label']:<46} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}")

    a, b = results[0], results[1]
    delta_sharpe = b["sharpe"] - a["sharpe"]
    delta_dd     = b["maxdd"]  - a["maxdd"]

    print(f"\n  RESULT for {symbol}:")
    if delta_sharpe > 0.10:
        print(f"  ✓ CANDLE FEATURES HELP: Sharpe {a['sharpe']:+.2f} → {b['sharpe']:+.2f} "
              f"({delta_sharpe:+.2f})")
        print(f"  Recommendation: run retrain_champion.py with candle features for {symbol}")
    elif delta_sharpe > 0:
        print(f"  ~ Marginal improvement: {a['sharpe']:+.2f} → {b['sharpe']:+.2f} "
              f"({delta_sharpe:+.2f}) — may not justify added complexity")
    else:
        print(f"  ✗ No improvement: {a['sharpe']:+.2f} → {b['sharpe']:+.2f} "
              f"({delta_sharpe:+.2f}) — candle features don't help {symbol}")
    if delta_dd < -0.5:
        print(f"  MaxDD improved: {a['maxdd']:.1f}% → {b['maxdd']:.1f}%")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="WF comparison: with/without candle features")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_DEFAULTS.keys()))
    args = p.parse_args()

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    symbols = [args.symbol] if args.symbol else list(SYMBOL_DEFAULTS.keys())

    print(f"\n{'='*70}")
    print(f"  CANDLE FEATURE INJECTION — WF COMPARISON")
    print(f"  Hypothesis: candle_p_buy/sell as XGBoost inputs improves Sharpe >+3.13")
    print(f"  Both configs: XGBoost + supervised enc8, same WF settings")
    print(f"{'='*70}")

    candle_parquet_missing = False
    for sym in symbols:
        path = FEAT_DIR / f"candle_signal_{sym}.parquet"
        if not path.exists():
            print(f"  MISSING: {path}")
            candle_parquet_missing = True

    if candle_parquet_missing:
        print(f"\n  Run first:")
        print(f"    conda run -n envmt5 python scripts/build_candle_features.py")
        print()

    for sym in symbols:
        compare_symbol(sym, full_cfg)


if __name__ == "__main__":
    main()
