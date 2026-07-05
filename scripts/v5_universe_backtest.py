"""V5 universe-widening backtest — same pre-registered engine, nested D1 universes.

Measures the diversification impact on the H4 CTA engine using the v4 CTA D1
data (2008-2026, spread column already in price units = cost_bps/1e4 * close).

Tiers (nested; declared before any run):
  core4     EURUSD GBPUSD USDJPY GOLD          (frequency ablation vs H4 run)
  fx-metals all 18 FX pairs + 5 metals          (MT5/HFM tradeable)
  mt5-cfd   + energy + equity index CFDs        (MT5/HFM tradeable)
  full48    + rates, ags, crypto                (diversification ceiling)

Engine parameters are identical to the pre-registered H4 config, expressed in
daily bars: EWMAC speeds (8,32)..(64,256), cluster-equal risk across asset
classes, 10% vol target, halflife 42d, causal buffer 0.10, shift(1) execution.
Pre-registered evaluation start 2010-01-01 (2008-2009 warm-up); use
--eval-start 2017-01-01 to compare like-for-like with the H4 runs.

    python scripts/v5_universe_backtest.py --tier full48 --run-id d1-cta-v5-full48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.cta.panel import asset_classes, build_panels
from src.cta.universe import UNIVERSE
from src.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from src.v5.artifacts import V5ArtifactWriter
from src.cta.signals import combine, xsmom
from src.cta.strategy import rebalance_hold
from src.v5.h4_cta import (buffer_band_causal, cluster_inv_vol,
                           mtm_pnl_price_units, vol_target_h4)

PERIODS_PER_YEAR = 252
ANN = np.sqrt(PERIODS_PER_YEAR)
# "fast" = the pre-registered H4 engine set; "slow" = the v4 locked champion set
# (src/cta/strategy.py TREND_SPEEDS) for reproducing the v4 CTA evidence.
SPEED_SETS = {
    "fast": ((8, 32), (16, 64), (32, 128), (64, 256)),
    "slow": ((32, 128), (64, 256)),
}
SPEEDS = SPEED_SETS["fast"]
EVAL_START = "2010-01-01"

FX = [a for a, v in UNIVERSE.items() if v["asset_class"].startswith("FX")]
METALS = [a for a, v in UNIVERSE.items() if v["asset_class"] == "METAL"]
ENERGY = [a for a, v in UNIVERSE.items() if v["asset_class"] == "ENERGY"]
EQ = [a for a, v in UNIVERSE.items() if v["asset_class"] == "EQ_INDEX"]

TIERS = {
    "core4": ["EURUSD", "GBPUSD", "USDJPY", "GOLD"],
    "fx-metals": FX + METALS,
    "mt5-cfd": FX + METALS + ENERGY + EQ,
    "full48": list(UNIVERSE),
    # v4 locked SMALL_BASKET (src/cta/strategy.py): one instrument per class
    "basket5": ["GOLD", "UST10Y", "SPX", "WTI", "EURUSD"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tier", choices=sorted(TIERS), default="full48")
    ap.add_argument("--speeds", choices=sorted(SPEED_SETS), default="fast")
    ap.add_argument("--sleeve", choices=["trend", "combined"], default="trend",
                    help="combined = 50/50 EWMAC + cross-sectional momentum (v4 champion sleeve)")
    ap.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="daily")
    ap.add_argument("--eval-start", default=EVAL_START)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--entry-delay-bars", type=int, default=1)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--buffer-frac", type=float, default=0.10)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    args = ap.parse_args()

    close, spread, kept = build_panels(TIERS[args.tier], tf="D1")
    classes = asset_classes(kept)
    speeds = SPEED_SETS[args.speeds]
    from src.cta.signals import ewmac
    returns = close.pct_change(fill_method=None)
    sig = ewmac(close, speeds=speeds)
    if args.sleeve == "combined":
        sig = combine(sig, xsmom(close))
    raw = cluster_inv_vol(sig, returns, classes, args.target_vol, 42, ann=ANN)
    positions = vol_target_h4(raw, returns, args.target_vol, 42, ann=ANN)
    positions = rebalance_hold(positions, args.rebalance)
    positions = buffer_band_causal(positions, args.buffer_frac)
    pnl = mtm_pnl_price_units(positions, close, spread,
                              entry_delay_bars=args.entry_delay_bars,
                              spread_cost_mult=args.spread_mult)

    pnl = pnl.loc[args.eval_start:]
    positions_eval = positions.loc[args.eval_start:]
    equity = (1.0 + pnl["net"].fillna(0.0)).cumprod() * args.initial_equity
    equity.index.name = "time"

    daily = equity.pct_change(fill_method=None).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    bar_sharpe = (pnl["net"].mean() / pnl["net"].std() * ANN
                  if pnl["net"].std() > 0 else 0.0)
    ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)

    yearly = (1.0 + pnl["net"].fillna(0.0)).groupby(pnl.index.year).prod() - 1.0
    cls_sharpe = {}
    for c in sorted(set(classes.values())):
        cols = [f"net_{a}" for a in kept if classes[a] == c]
        r = pnl[cols].sum(axis=1)
        cls_sharpe[c] = round(float(r.mean() / r.std() * ANN), 3) if r.std() > 0 else 0.0

    stats = {
        "tier": args.tier,
        "speeds": args.speeds,
        "sleeve": args.sleeve,
        "rebalance": args.rebalance,
        "n_instruments": len(kept),
        "eval_start": str(pnl.index[0]), "eval_end": str(pnl.index[-1]),
        "years": round(years, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(equity.iloc[-1] / args.initial_equity - 1.0) * 100, 2),
        "cagr_pct": round(((equity.iloc[-1] / args.initial_equity) ** (1 / years) - 1.0) * 100, 3),
        "sharpe_daily_ann": round(float(bar_sharpe), 3),
        "sharpe_resampled": round(sharpe_ratio(equity), 3),
        "sortino": round(sortino_ratio(equity), 3),
        "sharpe_ci95": [round(ci_lo, 3), round(ci_hi, 3)],
        "max_drawdown_pct": round(max_drawdown(equity), 3),
        "ann_turnover": round(float(pnl["turnover"].sum() / years), 2),
        "ann_cost_drag_pct": round(float(pnl["cost"].sum() / years) * 100, 3),
        "per_class_sharpe": cls_sharpe,
        "yearly_net_return_pct": {int(y): round(v * 100, 2) for y, v in yearly.items()},
        "avg_gross_exposure": round(float(positions_eval.abs().sum(axis=1).mean()), 3),
    }

    delta = positions_eval.diff()
    events = delta.stack()
    events = events[events.abs() > 1e-9]
    trades = [dict(time=str(t), symbol=s, position_change=round(float(v), 6),
                   new_position=round(float(positions_eval.at[t, s]), 6))
              for (t, s), v in events.items()]

    settings = {"strategy": "d1_cta_ewmac_voltarget", "tier": args.tier,
                "instruments": kept, "timeframe": "D1",
                "speeds": [list(x) for x in speeds],
                "eval_start": args.eval_start, "target_vol": args.target_vol,
                "buffer_frac": args.buffer_frac,
                "entry_delay_bars": args.entry_delay_bars,
                "spread_cost_mult": args.spread_mult,
                "initial_equity": args.initial_equity}
    run_dir = V5ArtifactWriter().write_run(
        run_id=args.run_id, settings=settings, trades=trades, equity=equity,
        stats=stats,
        reconciliation={"status": "research_replay",
                        "note": "universe-widening ablation of the V5 H4 CTA "
                                "engine; costs are cost_bps per unit turnover"})

    print(f"run_dir: {run_dir}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
