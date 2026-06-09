"""
wf_retrain.py — Proper walk-forward validation + retrain for live deployment.

Phase 1 — Walk-forward (honest OOS):
  - Scaler + encoder fitted on first train_frac (80%) of bars
  - XGBoost RE-TRAINED at every fold boundary on expanding window
  - OOS predictions concatenated into a single equity curve
  - Sharpe reported on that curve (never saw training data)

Phase 2 — Full retrain (for live bot):
  - If OOS Sharpe >= MIN_OOS_SHARPE, retrain on ALL data and save artifacts
  - Backs up existing model first
  - Updates /tmp service files ready for sudo install

Usage:
    conda run -n envmt5 python scripts/wf_retrain.py
    conda run -n envmt5 python scripts/wf_retrain.py --symbol EURUSD
    conda run -n envmt5 python scripts/wf_retrain.py --no-retrain   # WF only
    conda run -n envmt5 python scripts/wf_retrain.py --train-days 120 --test-days 30
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline, PipelineConfig
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.evaluation.metrics import sharpe_ratio

# ── Settings ───────────────────────────────────────────────────────────────────

MIN_OOS_SHARPE = 0.5    # minimum OOS Sharpe to proceed with full retrain

SYMBOL_CFG = {
    "EURUSD": dict(
        data_path  = "data/EURUSD_M15.csv",
        model_dir  = "data/models/pipeline_EURUSD",
        pip_size   = 0.0001,
        sl_pips    = 30.0,
        tp_pips    = 60.0,
    ),
    "USDJPY": dict(
        data_path  = "data/USDJPY_M15.csv",
        model_dir  = "data/models/pipeline_USDJPY",
        pip_size   = 0.01,
        sl_pips    = 30.0,
        tp_pips    = 60.0,
    ),
}


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
    symbol:       str,
    train_days:   int,
    test_days:    int,
    do_retrain:   bool,
) -> float:
    """Returns the overall OOS Sharpe for this symbol."""
    cfg_s    = SYMBOL_CFG[symbol]
    df_raw   = _load_raw(cfg_s["data_path"])
    span_yrs = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bpy      = len(df_raw) / span_yrs    # bars per year

    print(f"\n{'='*68}")
    print(f"  {symbol}  —  Walk-Forward Validation + Retrain")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  train_days={train_days}  test_days={test_days}  bars/year≈{bpy:,.0f}")
    print(f"{'='*68}")

    # ── Phase 1: build features (scaler+encoder fit on train_frac=0.8 only) ───
    print("\nPhase 1 — Building features (scaler+encoder fit on first 80% of bars)...")
    t0   = time.time()
    pipe = PredictorPipeline.from_config()
    # Override to match this symbol's paths + deployed champion config
    pipe.cfg.data_path    = cfg_s["data_path"]
    pipe.cfg.artifacts_dir= cfg_s["model_dir"]
    pipe.cfg.bt_sl_pips   = cfg_s["sl_pips"]
    pipe.cfg.bt_tp_pips   = cfg_s["tp_pips"]
    pipe.cfg.bt_threshold = 0.40
    pipe.cfg.train_frac   = 0.80          # scaler+encoder see only first 80%
    pipe.cfg.wf_cache_dir = f"data/models/wf_cache_{symbol}"

    X, y = pipe.build_features(df_raw, train_frac=0.80)
    prices = df_raw.reindex(X.index)

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Labels: buy={( y== 1).sum():,}  hold={(y==0).sum():,}  sell={(y==-1).sum():,}")
    print(f"  Feature build: {time.time()-t0:.1f}s")

    # ── Phase 2: walk-forward (XGBoost retrained at every fold) ──────────────
    # NOTE: pipe.walk_forward() hard-codes pip_size=0.0001 via _make_backtest_cfg().
    # We bypass it and call WalkForwardValidator directly with the correct pip_size.
    print(f"\nPhase 2 — Walk-forward (XGBoost retrained every {test_days} days)...")
    wf_cfg = WalkForwardConfig(
        model_type  = "xgboost",
        window_type = "expanding",
        train_days  = train_days,
        test_days   = test_days,
        cache_dir   = pipe.cfg.wf_cache_dir,
        backtest    = BacktestConfig(
            threshold         = 0.40,
            pip_size          = cfg_s["pip_size"],   # critical: 0.0001 EUR, 0.01 JPY
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

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    result.print_fold_table()

    if len(result.equity) < 10:
        print("  WARNING: no trades generated — threshold may be too high")
        return float("nan")

    oos_sharpe_raw   = sharpe_ratio(result.equity)        # lib function (per-bar)
    oos_sharpe_annl  = _annualized_sharpe(result.equity, bpy)
    oos_dd           = float(((result.equity.cummax() - result.equity) / result.equity.cummax() * 100).max())
    all_pips         = [t["pnl_pips"] for t in result.trades]
    win_rate         = sum(1 for p in all_pips if p > 0) / len(all_pips) if all_pips else 0.0
    net_pnl          = (result.equity.iloc[-1] / result.equity.iloc[0] - 1) * 100

    print(f"\n  ── OOS SUMMARY ({symbol}) ──────────────────────────────────────────")
    print(f"  Sharpe (annualized)  : {oos_sharpe_annl:+.3f}   (per-bar: {oos_sharpe_raw:.3f})")
    print(f"  Win rate             : {win_rate:.1%}")
    print(f"  Total trades         : {len(result.trades):,}")
    print(f"  Max drawdown         : {oos_dd:.1f}%")
    print(f"  Net PnL              : {net_pnl:+.1f}%")
    print(f"  Folds completed      : {len(result.folds)}")

    # Fold-level Sharpe stats
    fold_sharpes = [f.sharpe for f in result.folds if f.n_trades > 0]
    if fold_sharpes:
        print(f"  Fold Sharpe  avg={np.mean(fold_sharpes):.2f}  "
              f"min={np.min(fold_sharpes):.2f}  max={np.max(fold_sharpes):.2f}  "
              f"positive={sum(1 for s in fold_sharpes if s > 0)}/{len(fold_sharpes)}")

    # ── Honest interpretation ─────────────────────────────────────────────────
    print()
    if oos_sharpe_annl >= 1.5:
        verdict = "STRONG edge — safe to deploy live"
    elif oos_sharpe_annl >= 0.5:
        verdict = "REAL but modest edge — deploy with caution, monitor closely"
    elif oos_sharpe_annl >= 0.0:
        verdict = "WEAK edge — barely profitable, needs improvement before live"
    else:
        verdict = "NO edge (negative Sharpe) — DO NOT deploy, model needs rework"
    print(f"  Verdict: {verdict}")

    # ── Phase 3: full retrain for live deployment ─────────────────────────────
    if do_retrain and oos_sharpe_annl >= MIN_OOS_SHARPE:
        print(f"\nPhase 3 — Full retrain on all {len(X):,} bars (for live deployment)...")
        _backup_model(cfg_s["model_dir"])

        pipe2 = PredictorPipeline.from_config()
        pipe2.cfg.data_path     = cfg_s["data_path"]
        pipe2.cfg.artifacts_dir = cfg_s["model_dir"]
        pipe2.cfg.bt_sl_pips    = cfg_s["sl_pips"]
        pipe2.cfg.bt_tp_pips    = cfg_s["tp_pips"]

        t2 = time.time()
        X_full, y_full = pipe2.build_features(df_raw, train_frac=1.0)
        pipe2.fit_full(X_full, y_full)
        pipe2.save()
        print(f"  Saved → {cfg_s['model_dir']}/  ({time.time()-t2:.1f}s)")
        print(f"  Features: {len(pipe2.feature_names())}  "
              f"({', '.join(pipe2.feature_names()[:4])}...)")

        # Update /tmp service file ready for sudo install
        _write_service(symbol, cfg_s["model_dir"])
        print(f"  Service file ready: /tmp/mt5-{symbol.lower()}.service")
        print(f"  Install with: sudo cp /tmp/mt5-{symbol.lower()}.service "
              f"/etc/systemd/system/ && sudo systemctl daemon-reload && "
              f"sudo systemctl restart mt5-{symbol.lower()}.service")

    elif do_retrain:
        print(f"\nPhase 3 — SKIPPED: OOS Sharpe {oos_sharpe_annl:.3f} < "
              f"threshold {MIN_OOS_SHARPE}")
        print(f"  Model NOT updated. Keeping current champion in {cfg_s['model_dir']}/")

    return oos_sharpe_annl


def _write_service(symbol: str, model_dir: str) -> None:
    sleep = 15 if symbol == "EURUSD" else 20
    svc = f"""[Unit]
Description=MT5 PipelineBot — {symbol}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rock
WorkingDirectory=/home/rock/Desktop/2026_Projects/MT5
Environment=PATH=/home/rock/anaconda3/envs/envmt5/bin:/home/rock/anaconda3/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep {sleep}
ExecStart=/home/rock/anaconda3/envs/envmt5/bin/python src/bots/pipeline_bot.py \\
    --symbol {symbol} \\
    --model-dir {model_dir} \\
    --flip-mode partial_close
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mt5-{symbol.lower()}

[Install]
WantedBy=multi-user.target
"""
    Path(f"/tmp/mt5-{symbol.lower()}.service").write_text(svc)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward validation + retrain")
    p.add_argument("--symbol",      default=None, choices=list(SYMBOL_CFG.keys()))
    p.add_argument("--train-days",  type=int,  default=180,
                   help="Initial expanding training window in days (default 180)")
    p.add_argument("--test-days",   type=int,  default=30,
                   help="OOS test window per fold in days (default 30)")
    p.add_argument("--no-retrain",  action="store_true",
                   help="Run walk-forward only, do not retrain for live")
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
    print("  FINAL OOS SHARPE SUMMARY")
    print(f"{'='*68}")
    for sym, s in results.items():
        flag = "✓ deployed" if (do_retrain and not np.isnan(s) and s >= MIN_OOS_SHARPE) else ""
        print(f"  {sym:<10}  OOS Sharpe = {s:+.3f}   {flag}")
    print()


if __name__ == "__main__":
    main()
