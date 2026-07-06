"""V5 XAUUSD lock-and-reopen A/B — pre-registered in V5_PLAN.MD.

Compares the live trailing exit against close-at-target-then-reopen (`lock`)
at several profit targets. Same signal/sizing/costs; identical metric.

    python scripts/v5_xau_lock_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown
from src.v5.artifacts import V5ArtifactWriter
from src.v5.xau_trend import run_trades

EVAL_START = "2017-01-01"
DATA = "data/XAUUSD_H4_long.csv"
CONF_RISK = {"low": 0.5, "med": 1.0, "high": 1.5}


def evaluate(df, exit_mode, extra, label, equity0=3000.0):
    params = {"conf_risk_scale": CONF_RISK, **extra}
    res = run_trades(df, equity0=equity0, exit_mode=exit_mode,
                     flip_mode="confidence", params=params)
    eq = res["equity"].loc[EVAL_START:].dropna()
    trades = res["trades"]
    trades = trades[trades["close_time"] >= EVAL_START] if len(trades) else trades
    daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    st = dict(variant=label, exit_mode=exit_mode, **extra,
              sharpe=round(sharpe, 3),
              sharpe_ci95=[round(ci[0], 3), round(ci[1], 3)],
              cagr_pct=round(((eq.iloc[-1] / equity0) ** (1 / years) - 1) * 100, 2),
              total_return_pct=round((eq.iloc[-1] / equity0 - 1) * 100, 1),
              max_dd_pct=round(max_drawdown(eq), 2),
              n_trades=int(len(trades)),
              win_rate_pct=round(len(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
              profit_factor=round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 2)
              if len(losses) and losses["pnl"].sum() else float("inf"),
              exit_reasons=trades["exit_reason"].value_counts().to_dict() if len(trades) else {})
    print(f"{label:20s} Sharpe {st['sharpe']:+.3f} CI {st['sharpe_ci95']} "
          f"ret {st['total_return_pct']:+.1f}% DD {st['max_dd_pct']:.1f}% "
          f"trades {st['n_trades']} win {st['win_rate_pct']}% PF {st['profit_factor']}")
    return st, eq


def main():
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    writer = V5ArtifactWriter()
    cells = [
        ("trail", {}, "baseline_trail"),
        ("lock", {"lock_r": 1.0}, "lock_1R"),
        ("lock", {"lock_r": 2.0}, "lock_2R"),
        ("lock", {"lock_r": 3.0}, "lock_3R"),
    ]
    for exit_mode, extra, label in cells:
        st, eq = evaluate(df, exit_mode, extra, label)
        writer.write_run(run_id=f"xau-{label.replace('_', '-')}",
                         settings={"strategy": "xau_lock_compare", "symbol": "XAUUSD",
                                   "exit_mode": exit_mode, **extra},
                         trades=[], equity=eq, stats=st,
                         reconciliation={"status": "research_replay"})


if __name__ == "__main__":
    main()
