"""
Champion boost comparison: Baseline vs Champion + fractal_corr feature.

Trains two identical XGBoost pipelines on the same 80/20 split:
  Run 1 — Baseline:  champion config (39 features, forward-return labels)
  Run 2 — +Fractal:  same config + fractal_corr as feature 40

Both runs use the same backtester seed, same SL/TP, same risk manager.
Prints a side-by-side delta at the end.

Usage
-----
    conda run -n envmt5 python scripts/backtest_fractal_feature.py
    conda run -n envmt5 python scripts/backtest_fractal_feature.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_fractal_feature.py --fractal-min-win 6 --fractal-max-win 60
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

from src.pipeline       import PredictorPipeline, PipelineConfig
from src.core.risk_manager import RiskManager, RiskConfig
from src.evaluation.backtester import Backtester, BacktestConfig


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",          default="EURUSD")
    p.add_argument("--data",            default=None)
    p.add_argument("--fractal-min-win", type=int, default=6)
    p.add_argument("--fractal-max-win", type=int, default=60)
    return p.parse_args()


# ── Helper: build BacktestConfig from champion pair_meta ──────────────────────

def _bt_cfg(pipe: PredictorPipeline, rm: RiskManager) -> BacktestConfig:
    import json
    pm_path = Path(pipe.cfg.artifacts_dir) / "pair_meta.json"
    if pm_path.exists():
        pm = json.loads(pm_path.read_text())
        sl   = float(pm.get("sl_pips",  30.0))
        tp   = float(pm.get("tp_pips",  60.0))
        pips = float(pm.get("pip_size", 0.0001))
    else:
        sl, tp, pips = 30.0, 60.0, 0.0001
    return BacktestConfig(
        threshold       = pipe.cfg.bt_threshold,
        pip_size        = pips,
        sl_pips         = sl,
        tp_pips         = tp,
        spread_pips     = 1.0,
        commission_pips = 0.5,
        initial_balance = 10_000.0,
        risk_manager    = rm,
    )


# ── Helper: run one pipeline, return (result, elapsed_s) ─────────────────────

def _run(
    cfg:      PipelineConfig,
    df_raw:   pd.DataFrame,
    rm:       RiskManager,
    label:    str,
) -> tuple:
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    t0   = time.time()
    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)
    pipe.fit_full(X, y)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s  —  {X.shape[1]} features  {len(X):,} rows")

    class _M:
        def predict_proba(self, X):
            return pipe.predict_batch(X)[["P_buy", "P_hold", "P_sell"]].values

    prices = df_raw[["open", "high", "low", "close"]].reindex(X.index)
    bt_cfg = _bt_cfg(pipe, rm)
    result = Backtester(seed=42).run(_M(), X, prices, bt_cfg)
    result.report(title=label, extra={"Features": str(X.shape[1])})
    return result, elapsed


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(result) -> dict:
    if not result.trades:
        return {}
    trades = result.trades
    wins   = [t for t in trades if t["pnl_dollars"] > 0]
    pnl    = sum(t["pnl_dollars"] for t in trades)
    return {
        "trades":   len(trades),
        "win_rate": len(wins) / len(trades),
        "net_pnl":  pnl,
        "final":    result.config.initial_balance + pnl,
        "sharpe":   result.sharpe,
        "drawdown": result.drawdown,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = args.data or f"data/{args.symbol}_M15.csv"
    print(f"\nLoading {data_path} …")
    df_raw = pd.read_csv(data_path)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    time_col = next(c for c in df_raw.columns if "time" in c)
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    df_raw = df_raw.set_index(time_col).sort_index()
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # ── Load champion config from config.yaml ─────────────────────────────────
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    base_cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    # Override symbol path
    base_cfg = PipelineConfig(
        **{**base_cfg.__dict__,
           "data_path": data_path,
           "artifacts_dir": f"data/models/pipeline_{args.symbol}",
           "fractal_enabled": False,
        }
    )

    # +Fractal config: identical except fractal_enabled=True
    frac_cfg = PipelineConfig(
        **{**base_cfg.__dict__,
           "fractal_enabled": True,
           "fractal_min_win": args.fractal_min_win,
           "fractal_max_win": args.fractal_max_win,
        }
    )

    # Shared risk manager
    rm = RiskManager(RiskConfig(tiers=base_cfg.rm_tiers or []))

    # ── Run both ──────────────────────────────────────────────────────────────
    r_base, t_base = _run(base_cfg, df_raw, rm, "BASELINE (39 features)")
    r_frac, t_frac = _run(frac_cfg, df_raw, rm,
                          f"+FRACTAL_CORR ({args.fractal_min_win}–{args.fractal_max_win} win)")

    # ── Delta summary ─────────────────────────────────────────────────────────
    m_b = _metrics(r_base)
    m_f = _metrics(r_frac)

    if m_b and m_f:
        print(f"\n{'─'*56}")
        print("  DELTA  (+Fractal minus Baseline)")
        print(f"{'─'*56}")
        for k in ["net_pnl", "final", "sharpe", "drawdown", "win_rate", "trades"]:
            b, f = m_b[k], m_f[k]
            if k == "drawdown":
                arrow = "↓ better" if f <= b else "↑ worse"
            else:
                arrow = "↑ better" if f >= b else "↓ worse"
            print(f"  {k:<12}: baseline={b:>10.4f}  fractal={f:>10.4f}"
                  f"  Δ={f-b:+.4f}  {arrow}")
        print(f"  {'train_time':<12}: baseline={t_base:>10.0f}s  fractal={t_frac:>10.0f}s")
        print(f"{'─'*56}\n")


if __name__ == "__main__":
    main()
