"""V5 H4 CTA backtest runner — writes Lumibot-style artifacts per run.

Primary pre-registered run:
    python scripts/v5_h4_cta_backtest.py --run-id h4-cta-v5

Stress runs (reproducible flags, report all — never best-only):
    --spread-mult 2.0        # double costs
    --entry-delay-bars 2     # one extra bar of execution delay
    --eval-start 2021-01-01  # recent-half subsample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from src.v5.artifacts import V5ArtifactWriter
from src.v5.h4_cta import (CONFIG, PERIODS_PER_YEAR, SYMBOLS, h4_pnl,
                           h4_positions, load_h4_panel)

# Pre-registered evaluation start: slowest EWMAC span is 256 trading days,
# so 2015 data warms up through 2016 and scoring starts 2017-01-01.
EVAL_START = "2017-01-01"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval-start", default=EVAL_START)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--entry-delay-bars", type=int, default=1)
    ap.add_argument("--buffer-frac", type=float, default=CONFIG["buffer_frac"])
    ap.add_argument("--target-vol", type=float, default=CONFIG["target_vol"])
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    args = ap.parse_args()

    close, spread = load_h4_panel(args.data_dir)
    cfg = {**CONFIG, "buffer_frac": args.buffer_frac, "target_vol": args.target_vol,
           "entry_delay_bars": args.entry_delay_bars,
           "spread_cost_mult": args.spread_mult}
    positions = h4_positions(close, cfg)
    pnl = h4_pnl(positions, close, spread,
                 entry_delay_bars=args.entry_delay_bars,
                 spread_cost_mult=args.spread_mult)

    pnl = pnl.loc[args.eval_start:]
    positions_eval = positions.loc[args.eval_start:]
    equity = (1.0 + pnl["net"].fillna(0.0)).cumprod() * args.initial_equity
    equity.index.name = "time"

    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    bar_sharpe = (pnl["net"].mean() / pnl["net"].std() * np.sqrt(PERIODS_PER_YEAR)
                  if pnl["net"].std() > 0 else 0.0)
    ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)

    yearly = (1.0 + pnl["net"].fillna(0.0)).groupby(pnl.index.year).prod() - 1.0
    sym_sharpe = {}
    for sym in SYMBOLS:
        r = pnl[f"net_{sym}"].dropna()
        sym_sharpe[sym] = float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)) if r.std() > 0 else 0.0

    stats = {
        "eval_start": str(pnl.index[0]),
        "eval_end": str(pnl.index[-1]),
        "years": round(years, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(equity.iloc[-1] / args.initial_equity - 1.0) * 100, 3),
        "cagr_pct": round(((equity.iloc[-1] / args.initial_equity) ** (1 / years) - 1.0) * 100, 3),
        "sharpe_bar_ann": round(float(bar_sharpe), 3),
        "sharpe_daily_ann": round(sharpe_ratio(equity), 3),
        "sortino_daily_ann": round(sortino_ratio(equity), 3),
        "sharpe_daily_ci95": [round(ci_lo, 3), round(ci_hi, 3)],
        "max_drawdown_pct": round(max_drawdown(equity), 3),
        "ann_turnover": round(float(pnl["turnover"].sum() / years), 2),
        "ann_cost_drag_pct": round(float(pnl["cost"].sum() / years) * 100, 3),
        "yearly_net_return_pct": {int(y): round(v * 100, 2) for y, v in yearly.items()},
        "per_symbol_sharpe": {k: round(v, 3) for k, v in sym_sharpe.items()},
        "avg_gross_exposure": round(float(positions_eval.abs().sum(axis=1).mean()), 3),
    }

    # position-change events stand in for discrete trades in this MTM framework
    delta = positions_eval.diff()
    events = delta.stack()
    events = events[events.abs() > 1e-9]
    trades = [dict(time=str(t), symbol=s, position_change=round(float(v), 6),
                   new_position=round(float(positions_eval.at[t, s]), 6))
              for (t, s), v in events.items()]

    settings = {"strategy": "h4_cta_ewmac_voltarget", "symbols": list(SYMBOLS),
                "timeframe": "H4", "eval_start": args.eval_start,
                "initial_equity": args.initial_equity, **cfg}
    run_dir = V5ArtifactWriter().write_run(
        run_id=args.run_id, settings=settings, trades=trades, equity=equity,
        stats=stats,
        reconciliation={"status": "research_replay",
                        "note": "no fitted components; lookahead guarded by "
                                "tests/test_v5_h4_cta.py"})

    print(f"run_dir: {run_dir}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
