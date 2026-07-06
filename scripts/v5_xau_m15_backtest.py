"""V5 XAUUSD M15-execution experiment — pre-registered cells E0..E4.

    python scripts/v5_xau_m15_backtest.py --cell all
    python scripts/v5_xau_m15_backtest.py --cell E2a --spread-mult 2.0 \
        --run-id xau-m15-e2a-costx2                       # stress a winner
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
from src.v5.xau_m15_exec import resample_h4, run_trades_m15
from src.v5.xau_trend import run_trades

EVAL_START = "2017-01-01"
DATA = "data/XAUUSD_M15_spliced.csv"
CONF_RISK = {"low": 0.5, "med": 1.0, "high": 1.5}

CELLS = {
    "E0": dict(),                                   # H4 engine reference
    "E1": dict(limit_k=None, trail_source="h4"),
    "E2a": dict(limit_k=0.25, trail_source="h4"),
    "E2b": dict(limit_k=0.50, trail_source="h4"),
    "E3": dict(limit_k=None, trail_source="m15"),
    "E4": dict(limit_k=None, trail_source="h4", session_block=(0, 8)),
    "E5a": dict(limit_k=None, trail_source="h4", confirm_rule="1h"),
    "E5b": dict(limit_k=None, trail_source="h4", confirm_rule="30min"),
}


def load_m15() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def metrics(eq: pd.Series, trades: pd.DataFrame, equity0: float, label: str) -> dict:
    eq = eq.loc[EVAL_START:].dropna()
    trades = trades[trades["close_time"] >= EVAL_START] if len(trades) else trades
    daily = eq.resample("D").last().pct_change(fill_method=None).dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    return dict(cell=label,
                sharpe=round(sharpe, 3),
                sharpe_ci95=[round(ci[0], 3), round(ci[1], 3)],
                cagr_pct=round(((eq.iloc[-1] / equity0) ** (1 / years) - 1) * 100, 2),
                max_dd_pct=round(max_drawdown(eq), 2),
                n_trades=int(len(trades)),
                win_rate_pct=round(len(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
                profit_factor=round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 2)
                if len(losses) and losses["pnl"].sum() else float("inf"),
                exit_reasons=trades["exit_reason"].value_counts().to_dict()
                if len(trades) else {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="all", choices=[*CELLS, "all"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--equity", type=float, default=3000.0)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--entry-delay-h4", type=int, default=0)
    ap.add_argument("--limit-penetration-pips", type=float, default=0.0)
    args = ap.parse_args()

    m15 = load_m15()
    params = dict(spread_cost_mult=args.spread_mult, conf_risk_scale=CONF_RISK)
    writer = V5ArtifactWriter()
    names = list(CELLS) if args.cell == "all" else [args.cell]

    for name in names:
        if name == "E0":
            h4 = resample_h4(m15)
            h4["spread"] = m15["spread"].resample("4h", label="left",
                                                  closed="left").mean().reindex(h4.index)
            res = run_trades(h4.dropna(), equity0=args.equity, exit_mode="trail",
                             flip_mode="confidence",
                             params={**params,
                                     "entry_delay_bars": 1 + args.entry_delay_h4})
        else:
            res = run_trades_m15(m15, equity0=args.equity, **CELLS[name],
                                 params=params,
                                 entry_delay_h4=args.entry_delay_h4,
                                 limit_penetration_pips=args.limit_penetration_pips)
        st = metrics(res["equity"], res["trades"], args.equity, name)
        run_id = args.run_id or f"xau-m15-{name.lower()}"
        writer.write_run(
            run_id=run_id,
            settings={"strategy": "xau_m15_exec", "cell": name,
                      "data": DATA, "equity0": args.equity,
                      "spread_mult": args.spread_mult,
                      "entry_delay_h4": args.entry_delay_h4,
                      "limit_penetration_pips": args.limit_penetration_pips,
                      **{k: str(v) for k, v in CELLS[name].items()}},
            trades=res["trades"].to_dict("records"),
            equity=res["equity"].loc[EVAL_START:].dropna(), stats=st,
            reconciliation={"status": "research_replay",
                            "data_gate": "spliced M15 == H4 CSV outside "
                                         "declared 2023-04..09 window"})
        print(f"{run_id:22s} Sharpe {st['sharpe']:+.3f} CI {st['sharpe_ci95']} "
              f"CAGR {st['cagr_pct']:+.2f}% DD {st['max_dd_pct']:.1f}% "
              f"trades {st['n_trades']} win {st['win_rate_pct']}% "
              f"PF {st['profit_factor']}")


if __name__ == "__main__":
    main()
