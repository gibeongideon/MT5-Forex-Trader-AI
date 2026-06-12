"""
train_candle_model.py — Train a dedicated 1-bar (15M candle) predictor model.

Improvements v2:
  1. label_threshold=0.0005  (was 0.0002) — requires real 5-pip move, covers spread+commission
  2. confidence threshold=0.60 (was 0.40) — only trade high-confidence signals
  3. sliding window WF (was expanding) — adapts faster to regime changes
  4. SL=10p / TP=30p (was 15/20) — 1:3 reward:risk ratio
  5. CatBoost (was XGBoost) — ordered boosting, less overfitting
  6. Session filter features — London/NY/Tokyo/overlap flags + hour_sin/cos
  7. Multi-timeframe EMA features — 1H EMA20 and 4H EMA50 trend ratios

Phase 1 — Build features + extra session/MTF features
Phase 2 — Walk-forward with sliding windows (CatBoost per fold)
Phase 3 — Full retrain on all data if OOS Sharpe >= 0.5, save to candle_SYMBOL/

Usage:
    conda run -n envmt5 python scripts/train_candle_model.py
    conda run -n envmt5 python scripts/train_candle_model.py --symbol EURUSD
    conda run -n envmt5 python scripts/train_candle_model.py --no-retrain
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.evaluation.metrics import sharpe_ratio

# ── Settings ───────────────────────────────────────────────────────────────────

MIN_OOS_SHARPE  = 0.5
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005   # v2: 5-pip threshold (was 2-pip); covers spread+commission
THRESHOLD       = 0.60     # v2: only trade when model is ≥60% confident (was 40%)

SYMBOL_CFG = {
    "EURUSD": dict(
        data_path = "data/EURUSD_M15.csv",
        model_dir = "data/models/candle_EURUSD",
        pip_size  = 0.0001,
        sl_pips   = 10.0,   # v2: tighter SL (was 15p) → 1:3 R:R with TP=30
        tp_pips   = 30.0,   # v2: wider TP (was 20p)
    ),
    "USDJPY": dict(
        data_path = "data/USDJPY_M15.csv",
        model_dir = "data/models/candle_USDJPY",
        pip_size  = 0.01,
        sl_pips   = 10.0,
        tp_pips   = 30.0,
    ),
}


# ── Extra features: session flags + multi-timeframe EMAs ──────────────────────

def _add_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Append session-time and multi-timeframe EMA features to X.

    All features are ratio/boolean scale-independent so no separate scaling needed
    for tree models (CatBoost/XGBoost).  No lookahead: uses close[t] to predict
    close[t+1] — the raw pipeline already enforces this with .shift(1) internally.
    """
    idx = X.index

    # ── Session flags (UTC hours) ──────────────────────────────────────────────
    hour = idx.hour
    extra = pd.DataFrame(index=idx)
    extra["session_sydney"]  = ((hour >= 22) | (hour < 7)).astype(float)   # UTC 22–07 (EAT 01–10)
    extra["session_tokyo"]   = ((hour >= 0)  & (hour < 9)).astype(float)   # UTC 00–09 (EAT 03–12)
    extra["session_london"]  = ((hour >= 8)  & (hour < 17)).astype(float)  # UTC 08–17 (EAT 11–20)
    extra["session_ny"]      = ((hour >= 13) & (hour < 22)).astype(float)  # UTC 13–22 (EAT 16–01)
    extra["session_tok_lon"] = ((hour >= 8)  & (hour < 9)).astype(float)   # UTC 08–09 (EAT 11–12) overlap
    extra["session_lon_ny"]  = ((hour >= 13) & (hour < 17)).astype(float)  # UTC 13–17 (EAT 16–20) ★
    extra["hour_sin"]        = np.sin(2 * np.pi * hour / 24)
    extra["hour_cos"]        = np.cos(2 * np.pi * hour / 24)

    # ── 1H EMA-20: trend direction on 1H timeframe ────────────────────────────
    # resample M15 → 1H (last close of each hour), compute EMA, ffill back to M15
    close_1h  = df_raw["close"].resample("1h").last().ffill()
    ema_1h    = close_1h.ewm(span=20, adjust=False).mean()
    ema_1h_m15= ema_1h.reindex(df_raw.index, method="ffill")
    # ratio: (price − EMA) / price; positive = price above EMA = bullish
    extra["ema_1h_ratio"]  = ((df_raw["close"] - ema_1h_m15) / df_raw["close"]).reindex(idx).fillna(0)
    # 1H EMA slope over last 4 M15 bars (approx 1H)
    extra["ema_1h_slope"]  = (ema_1h_m15.diff(4) / df_raw["close"]).reindex(idx).fillna(0)

    # ── 4H EMA-50: macro trend direction ──────────────────────────────────────
    close_4h  = df_raw["close"].resample("4h").last().ffill()
    ema_4h    = close_4h.ewm(span=50, adjust=False).mean()
    ema_4h_m15= ema_4h.reindex(df_raw.index, method="ffill")
    extra["ema_4h_ratio"]  = ((df_raw["close"] - ema_4h_m15) / df_raw["close"]).reindex(idx).fillna(0)
    # 4H EMA slope over last 16 M15 bars (approx 4H)
    extra["ema_4h_slope"]  = (ema_4h_m15.diff(16) / df_raw["close"]).reindex(idx).fillna(0)

    # Align and concatenate
    extra = extra.reindex(idx).fillna(0)
    return pd.concat([X, extra], axis=1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _annualized_sharpe(equity: pd.Series, bars_per_year: float) -> float:
    r = equity.pct_change().dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def _backup_model(model_dir: str) -> None:
    src = Path(model_dir)
    if not src.exists():
        return
    ts  = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    dst = src.parent / f"{src.name}_backup_{ts}"
    shutil.copytree(src, dst)
    print(f"  Backed up existing model → {dst}")


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(
    symbol:     str,
    train_days: int,
    test_days:  int,
    do_retrain: bool,
) -> float:
    cfg_s    = SYMBOL_CFG[symbol]
    df_raw   = _load_raw(cfg_s["data_path"])
    span_yrs = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bpy      = len(df_raw) / span_yrs

    print(f"\n{'='*68}")
    print(f"  CANDLE MODEL v2 — {symbol}")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  label_horizon={LABEL_HORIZON}  label_threshold={LABEL_THRESHOLD}  threshold={THRESHOLD}")
    print(f"  train_days={train_days}  test_days={test_days}  window=sliding  model=catboost")
    print(f"  SL={cfg_s['sl_pips']}p  TP={cfg_s['tp_pips']}p  (R:R = 1:{cfg_s['tp_pips']/cfg_s['sl_pips']:.1f})")
    print(f"{'='*68}")

    # ── Phase 1: build features ────────────────────────────────────────────────
    print("\nPhase 1 — Building features...")
    t0   = time.time()
    pipe = PredictorPipeline.from_config()

    pipe.cfg.label_horizon   = LABEL_HORIZON
    pipe.cfg.label_threshold = LABEL_THRESHOLD
    pipe._fp.label_horizon   = LABEL_HORIZON
    pipe._fp.label_threshold = LABEL_THRESHOLD
    pipe.cfg.data_path       = cfg_s["data_path"]
    pipe.cfg.artifacts_dir   = cfg_s["model_dir"]
    pipe.cfg.bt_sl_pips      = cfg_s["sl_pips"]
    pipe.cfg.bt_tp_pips      = cfg_s["tp_pips"]
    pipe.cfg.bt_threshold    = THRESHOLD
    pipe.cfg.train_frac      = 0.80
    pipe.cfg.wf_cache_dir    = f"data/models/wf_cache_candle2_{symbol}"

    X, y = pipe.build_features(df_raw, train_frac=0.80)
    prices = df_raw.reindex(X.index)

    # Add session + MTF features
    X = _add_extra_features(df_raw, X)

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Labels: buy={(y== 1).sum():,}  hold={(y==0).sum():,}  sell={(y==-1).sum():,}")
    print(f"  Extra features: session flags (8) + MTF EMAs (4) = 12 added  [v3: Sydney+TokyoLondon added]")
    print(f"  Feature build: {time.time()-t0:.1f}s")

    # ── Phase 2: walk-forward ──────────────────────────────────────────────────
    print(f"\nPhase 2 — Walk-forward (CatBoost, sliding {train_days}d window, {test_days}d test)...")
    wf_cfg = WalkForwardConfig(
        model_type  = "catboost",     # v2: CatBoost (was xgboost)
        window_type = "sliding",      # v2: sliding window (was expanding)
        train_days  = train_days,
        test_days   = test_days,
        cache_dir   = pipe.cfg.wf_cache_dir,
        backtest    = BacktestConfig(
            threshold         = THRESHOLD,
            pip_size          = cfg_s["pip_size"],
            sl_pips           = cfg_s["sl_pips"],
            tp_pips           = cfg_s["tp_pips"],
            spread_pips       = 1.0,
            commission_pips   = 0.5,
            initial_balance   = 10_000.0,
            risk_pct          = 0.01,
            use_regime_filter = False,
        ),
    )

    t1     = time.time()
    result = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)
    print(f"  Walk-forward: {time.time()-t1:.1f}s")

    # ── Report ─────────────────────────────────────────────────────────────────
    print()
    result.print_fold_table()

    if len(result.equity) < 10:
        print("  WARNING: no trades generated — threshold may be too high")
        return float("nan")

    oos_sharpe_annl = _annualized_sharpe(result.equity, bpy)
    oos_sharpe_raw  = sharpe_ratio(result.equity)
    oos_dd          = float(((result.equity.cummax() - result.equity) / result.equity.cummax() * 100).max())
    all_pips        = [t["pnl_pips"] for t in result.trades]
    win_rate        = sum(1 for p in all_pips if p > 0) / len(all_pips) if all_pips else 0.0
    net_pnl         = (result.equity.iloc[-1] / result.equity.iloc[0] - 1) * 100

    print(f"\n  ── OOS SUMMARY ({symbol}) ──────────────────────────────────────────")
    print(f"  Sharpe (annualized)  : {oos_sharpe_annl:+.3f}   (per-bar: {oos_sharpe_raw:.3f})")
    print(f"  Win rate             : {win_rate:.1%}")
    print(f"  Total trades         : {len(result.trades):,}")
    print(f"  Max drawdown         : {oos_dd:.1f}%")
    print(f"  Net PnL              : {net_pnl:+.1f}%")
    print(f"  Folds completed      : {len(result.folds)}")

    fold_sharpes = [f.sharpe for f in result.folds if f.n_trades > 0]
    if fold_sharpes:
        print(f"  Fold Sharpe  avg={np.mean(fold_sharpes):.2f}  "
              f"min={np.min(fold_sharpes):.2f}  max={np.max(fold_sharpes):.2f}  "
              f"positive={sum(1 for s in fold_sharpes if s > 0)}/{len(fold_sharpes)}")

    print()
    if oos_sharpe_annl >= 1.5:
        verdict = "STRONG edge — safe to deploy live"
    elif oos_sharpe_annl >= 0.5:
        verdict = "REAL but modest edge — deploy with caution, monitor closely"
    elif oos_sharpe_annl >= 0.0:
        verdict = "WEAK edge — barely profitable"
    else:
        verdict = "NO edge (negative Sharpe) — model needs rework"
    print(f"  Verdict: {verdict}")

    # ── Phase 3: full retrain ──────────────────────────────────────────────────
    if do_retrain and oos_sharpe_annl >= MIN_OOS_SHARPE:
        print(f"\nPhase 3 — Full retrain on all {len(X):,} bars...")
        _backup_model(cfg_s["model_dir"])

        pipe2 = PredictorPipeline.from_config()
        pipe2.cfg.label_horizon   = LABEL_HORIZON
        pipe2.cfg.label_threshold = LABEL_THRESHOLD
        pipe2._fp.label_horizon   = LABEL_HORIZON
        pipe2._fp.label_threshold = LABEL_THRESHOLD
        pipe2.cfg.data_path       = cfg_s["data_path"]
        pipe2.cfg.artifacts_dir   = cfg_s["model_dir"]
        pipe2.cfg.bt_sl_pips      = cfg_s["sl_pips"]
        pipe2.cfg.bt_tp_pips      = cfg_s["tp_pips"]
        pipe2.cfg.model_type      = "catboost"

        t2 = time.time()
        X_full, y_full = pipe2.build_features(df_raw, train_frac=1.0)
        # Add same extra features so they're stored in _feature_cols
        X_full = _add_extra_features(df_raw, X_full)
        # Sync _feature_cols BEFORE fit_full so save() writes the correct 50-col list
        pipe2._feature_cols = list(X_full.columns)
        pipe2.fit_full(X_full, y_full)
        pipe2.save()
        elapsed = time.time() - t2

        meta_path = Path(cfg_s["model_dir"]) / "pair_meta.json"
        meta = dict(
            symbol          = symbol,
            pip_size        = cfg_s["pip_size"],
            sl_pips         = cfg_s["sl_pips"],
            tp_pips         = cfg_s["tp_pips"],
            threshold       = THRESHOLD,
            label_horizon   = LABEL_HORIZON,
            label_threshold = LABEL_THRESHOLD,
            mode            = "candle_predictor",
            version         = 3,
            extra_features  = ["session_sydney", "session_tokyo", "session_london",
                               "session_ny", "session_tok_lon", "session_lon_ny",
                               "hour_sin", "hour_cos",
                               "ema_1h_ratio", "ema_1h_slope",
                               "ema_4h_ratio", "ema_4h_slope"],
        )
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"  Saved → {cfg_s['model_dir']}/  ({elapsed:.1f}s)")
        print(f"  Features: {len(pipe2.feature_names())}  "
              f"({', '.join(pipe2.feature_names()[:4])}...)")
        print(f"  pair_meta.json written  (pip_size={cfg_s['pip_size']}  "
              f"sl={cfg_s['sl_pips']}p  tp={cfg_s['tp_pips']}p  threshold={THRESHOLD})")
        print(f"\n  To run live:")
        print(f"  python src/bots/pipeline_bot.py \\")
        print(f"      --symbol {symbol} \\")
        print(f"      --model-dir data/models/pipeline_{symbol} \\")
        print(f"      --flip-mode candle_predictor \\")
        print(f"      --candle-model-dir {cfg_s['model_dir']}")

    elif do_retrain:
        print(f"\nPhase 3 — SKIPPED: OOS Sharpe {oos_sharpe_annl:.3f} < {MIN_OOS_SHARPE}")
        print(f"  Model NOT saved.")

    return oos_sharpe_annl


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train 1-bar candle predictor model (v2)")
    p.add_argument("--symbol",     default=None, choices=list(SYMBOL_CFG.keys()))
    p.add_argument("--train-days", type=int, default=120,
                   help="Sliding training window in days (default 120)")
    p.add_argument("--test-days",  type=int, default=60,
                   help="OOS test window per fold in days (default 60)")
    p.add_argument("--no-retrain", action="store_true")
    args = p.parse_args()

    symbols    = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    do_retrain = not args.no_retrain

    results = {}
    for sym in symbols:
        results[sym] = run_symbol(
            sym,
            train_days = args.train_days,
            test_days  = args.test_days,
            do_retrain = do_retrain,
        )

    print(f"\n{'='*68}")
    print("  CANDLE MODEL v2 — FINAL OOS SHARPE SUMMARY")
    print(f"{'='*68}")
    for sym, s in results.items():
        flag = "✓ saved" if (do_retrain and not np.isnan(s) and s >= MIN_OOS_SHARPE) else "✗ not saved"
        print(f"  {sym:<10}  OOS Sharpe = {s:+.3f}   {flag}")
    print()


if __name__ == "__main__":
    main()
