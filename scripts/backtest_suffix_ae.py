"""
Backtest: Baseline vs Suffix Automaton + Autoencoder position sizing.

Loads the trained EURUSD pipeline, runs a bar-by-bar simulation twice on
the same data, and prints a side-by-side performance comparison.

  Run 1 — Baseline:  confidence-tier risk manager only
  Run 2 — SA+AE:     same model + SA+AE lot multiplier on top

Usage:
    conda run -n envmt5 python scripts/backtest_suffix_ae.py
    conda run -n envmt5 python scripts/backtest_suffix_ae.py --data data/EURUSD_M15.csv
    conda run -n envmt5 python scripts/backtest_suffix_ae.py --symbol USDJPY --model-dir data/models/pipeline_USDJPY
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd

from src.core.risk_manager import RiskManager, RiskConfig
from src.core.suffix_ae_sizer import SuffixAESizer
from src.evaluation.backtester import Backtester, BacktestConfig
from src.pipeline import PredictorPipeline


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default=None, help="Path to OHLCV CSV")
    p.add_argument("--symbol",    default="EURUSD")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--algo-mode", type=int, default=1, choices=[1, 2, 3, 4],
                   help="SA algo mode: 1=linear 2=conservative 3=aggressive 4=mean-reversion")
    p.add_argument("--no-ae",     action="store_true", help="Disable autoencoder gate")
    p.add_argument("--history",   type=int, default=150, help="SA history length (bars)")
    p.add_argument("--dna-window",type=int, default=16,  help="SA DNA window (bars)")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    symbol = args.symbol
    data_path = args.data or f"data/{symbol}_M15.csv"
    print(f"\nLoading {data_path} …")
    df_raw = _load_csv(data_path)
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # ── 2. Load pipeline ──────────────────────────────────────────────────────
    print("\nLoading pipeline model …")
    pipe = PredictorPipeline.from_config()
    model_dir = args.model_dir or f"data/models/pipeline_{symbol}"
    if not Path(model_dir).exists():
        model_dir = "data/models/pipeline"
    pipe.load(model_dir)

    # Read sl/tp from pair_meta if available
    import json
    pm_path = Path(model_dir) / "pair_meta.json"
    if pm_path.exists():
        pm = json.loads(pm_path.read_text())
        sl_pips  = float(pm.get("sl_pips", 30.0))
        tp_pips  = float(pm.get("tp_pips", 60.0))
        pip_size = float(pm.get("pip_size", 0.0001))
    else:
        sl_pips, tp_pips, pip_size = 30.0, 60.0, 0.0001

    print(f"  model_dir={model_dir}  SL={sl_pips:.0f}p  TP={tp_pips:.0f}p")

    # ── 3. Build features ─────────────────────────────────────────────────────
    print("\nBuilding features …")
    X, _ = pipe.build_features(df_raw)

    # Build a model-like wrapper so Backtester can call predict_proba(X)
    class _PipeModel:
        def predict_proba(self, X):
            import numpy as np
            signals = pipe.predict_batch(X)
            proba = signals[["P_buy", "P_hold", "P_sell"]].values
            return proba

    model = _PipeModel()

    # Align prices to feature index (feature builder may drop warmup rows)
    prices = df_raw[["open", "high", "low", "close"]].reindex(X.index)

    # ── 4. Shared config ──────────────────────────────────────────────────────
    rm = RiskManager(RiskConfig())   # confidence-tier risk manager

    base_cfg = BacktestConfig(
        threshold      = pipe.cfg.bt_threshold,
        pip_size       = pip_size,
        sl_pips        = sl_pips,
        tp_pips        = tp_pips,
        spread_pips    = 1.0,
        commission_pips= 0.5,
        initial_balance= 10_000.0,
        risk_manager   = rm,
    )

    # ── 5. Run baseline ───────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RUN 1 — Baseline (confidence-tier risk manager only)")
    print("═" * 60)
    bt = Backtester(seed=42)
    result_base = bt.run(model, X, prices, base_cfg)
    result_base.report(
        title="BASELINE",
        extra={"SA+AE": "OFF"},
    )

    # ── 6. Run with SA+AE ─────────────────────────────────────────────────────
    sizer = SuffixAESizer(
        history_length = args.history,
        dna_window     = args.dna_window,
        algo_mode      = args.algo_mode,
        use_ae         = not args.no_ae,
    )

    sa_cfg = BacktestConfig(
        threshold       = pipe.cfg.bt_threshold,
        pip_size        = pip_size,
        sl_pips         = sl_pips,
        tp_pips         = tp_pips,
        spread_pips     = 1.0,
        commission_pips = 0.5,
        initial_balance = 10_000.0,
        risk_manager    = rm,
        suffix_ae_sizer = sizer,
    )

    mode_labels = {1: "linear", 2: "conservative", 3: "aggressive", 4: "mean-reversion"}
    print("\n" + "═" * 60)
    print(f"  RUN 2 — SA+AE  (mode={args.algo_mode}/{mode_labels[args.algo_mode]}  "
          f"hist={args.history}  dna={args.dna_window}  ae={'on' if not args.no_ae else 'off'})")
    print("═" * 60)
    bt2 = Backtester(seed=42)
    result_sa = bt2.run(model, X, prices, sa_cfg)
    result_sa.report(
        title="SA+AE",
        extra={
            "SA+AE":     "ON",
            "Algo mode": f"{args.algo_mode} ({mode_labels[args.algo_mode]})",
            "History":   str(args.history),
            "DNA window":str(args.dna_window),
            "AE gate":   "on" if not args.no_ae else "off",
        },
    )

    # ── 7. Delta summary ──────────────────────────────────────────────────────
    import numpy as np

    def _metrics(result) -> dict:
        if not result.trades:
            return {}
        trades = result.trades
        wins  = [t for t in trades if t["pnl_dollars"] > 0]
        total_pnl = sum(t["pnl_dollars"] for t in trades)
        final_bal = result.config.initial_balance + total_pnl
        return {
            "trades":   len(trades),
            "win_rate": len(wins) / len(trades),
            "net_pnl":  total_pnl,
            "final":    final_bal,
            "sharpe":   result.sharpe,
            "drawdown": result.drawdown,
        }

    m_b = _metrics(result_base)
    m_s = _metrics(result_sa)

    if m_b and m_s:
        print("\n" + "─" * 52)
        print("  DELTA  (SA+AE minus Baseline)")
        print("─" * 52)
        for k in ["net_pnl", "final", "sharpe", "drawdown", "win_rate"]:
            b, s = m_b[k], m_s[k]
            sign = "+" if s >= b else ""
            if k == "drawdown":
                sign = "+" if s <= b else ""   # lower drawdown is better
            print(f"  {k:<12}: baseline={b:>10.4f}  sa+ae={s:>10.4f}  Δ={sign}{s-b:+.4f}")
        print("─" * 52 + "\n")


if __name__ == "__main__":
    main()
