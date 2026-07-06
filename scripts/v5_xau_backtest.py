"""V5 XAUUSD standalone trend-trade backtest — pre-registered variant grid.

Runs all six declared variants (exit {flip, trail, sltp} x flip-mode
{confidence, always}) and reports every one. Stress flags reproduce the
battery for whichever variant is being examined.

    python scripts/v5_xau_backtest.py                       # full grid
    python scripts/v5_xau_backtest.py --exit trail --flip confidence \
        --spread-mult 2.0 --run-id xau-trend-trail-conf-costx2   # stress cell
"""
from __future__ import annotations

import argparse
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


def load_xau(path: str = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def evaluate(df: pd.DataFrame, exit_mode: str, flip_mode: str,
             equity0: float, overrides: dict) -> tuple[dict, dict]:
    ov = dict(overrides)
    sym = ov.pop("_symbol", "XAUUSD")
    res = run_trades(df, equity0=equity0, exit_mode=exit_mode,
                     flip_mode=flip_mode, params=ov, symbol=sym)
    eq = res["equity"].loc[EVAL_START:].dropna()
    trades = res["trades"]
    trades = trades[trades["close_time"] >= EVAL_START] if len(trades) else trades
    daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    by_conf = (trades.groupby("confidence")["pnl"]
               .agg(["count", "mean"]).round(2).to_dict("index")) if len(trades) else {}
    stats = dict(
        exit_mode=exit_mode, flip_mode=flip_mode,
        eval_start=str(eq.index[0]), eval_end=str(eq.index[-1]),
        years=round(years, 2),
        final_equity=round(float(eq.iloc[-1]), 2),
        total_return_pct=round(float(eq.iloc[-1] / eq.loc[:EVAL_START].iloc[0]
                                     if False else eq.iloc[-1] / equity0 - 1) * 100, 2),
        cagr_pct=round(((eq.iloc[-1] / equity0) ** (1 / years) - 1) * 100, 2),
        sharpe_daily=round(sharpe, 3),
        sharpe_ci95=[round(ci_lo, 3), round(ci_hi, 3)],
        max_dd_pct=round(max_drawdown(eq), 2),
        n_trades=int(len(trades)),
        win_rate_pct=round(len(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
        profit_factor=round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 2)
        if len(losses) and losses["pnl"].sum() != 0 else float("inf"),
        avg_r=round(float(trades["r_multiple"].mean()), 2) if len(trades) else 0.0,
        exit_reasons=trades["exit_reason"].value_counts().to_dict() if len(trades) else {},
        pnl_by_confidence=by_conf,
    )
    return stats, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--symbol", default="XAUUSD",
                    choices=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--equity", type=float, default=3000.0)
    ap.add_argument("--exit", choices=["flip", "trail", "sltp"], default=None)
    ap.add_argument("--flip", choices=["confidence", "always"], default=None)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--entry-delay-bars", type=int, default=1)
    ap.add_argument("--conf-risk", action="store_true",
                    help="round-2 variant: confidence-scaled risk 0.5/1.0/1.5%")
    args = ap.parse_args()

    df = load_xau(args.data or f"data/{args.symbol}_H4_long.csv")
    overrides = dict(spread_cost_mult=args.spread_mult,
                     entry_delay_bars=args.entry_delay_bars)
    if args.conf_risk:
        overrides["conf_risk_scale"] = {"low": 0.5, "med": 1.0, "high": 1.5}
    overrides["_symbol"] = args.symbol
    writer = V5ArtifactWriter()

    cells = ([(args.exit, args.flip)] if args.exit and args.flip else
             [(e, f) for e in ("flip", "trail", "sltp")
              for f in ("confidence", "always")])

    for exit_mode, flip_mode in cells:
        stats, res = evaluate(df, exit_mode, flip_mode, args.equity, overrides)
        run_id = args.run_id or f"{args.symbol.lower()}-trend-{exit_mode}-{flip_mode[:4]}"
        eq = res["equity"].loc[EVAL_START:].dropna()
        writer.write_run(
            run_id=run_id,
            settings={"strategy": "xau_h4_trend_trades", "symbol": "XAUUSD",
                      "timeframe": "H4", "equity0": args.equity,
                      "exit_mode": exit_mode, "flip_mode": flip_mode,
                      **overrides},
            trades=res["trades"].to_dict("records"),
            equity=eq, stats=stats,
            reconciliation={"status": "research_replay",
                            "note": "signal = validated H4 EWMAC forecast; "
                                    "no fitted components"})
        print(f"{run_id:28s} Sharpe {stats['sharpe_daily']:+.3f} "
              f"CI {stats['sharpe_ci95']} CAGR {stats['cagr_pct']:+.1f}% "
              f"DD {stats['max_dd_pct']:.1f}% trades {stats['n_trades']} "
              f"win {stats['win_rate_pct']}% PF {stats['profit_factor']} "
              f"avgR {stats['avg_r']}")


if __name__ == "__main__":
    main()
