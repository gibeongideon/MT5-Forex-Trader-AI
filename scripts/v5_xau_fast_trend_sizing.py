"""Does SIZING / trade-SELECTION beat the spread-edge problem for the fast
intraday trend bot? Discrete engine, real $0.36 cent spread, eval 2017+.

Tests:
  * flat sizing (all trades = 1 unit)          -> shows sizing is Sharpe-neutral
  * conviction scaling (size ~ forecast bucket) -> low/med/high risk multipliers
  * trade selection (raise enter threshold)      -> skip weak trades = pay spread
                                                    less, keep only big-move ones
The only way sizing can lift NET Sharpe is if signal strength predicts a better
cost-adjusted payoff (strong trend -> big move -> $0.36 is a smaller fraction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.v5.xau_trend as xt
from scripts.v5_xau_fast_trend_discrete import make_fast_champion, load, metrics

SPREAD_USD = 0.36
SLIP_USD = 0.02
TF, SPEED = "M30", "fast"

# (label, enter_thresh, flip_thresh, conf_risk_scale)
CONFIGS = [
    ("baseline enter0.5 conf.5/1/1.5", 0.5, 1.0, {"low": 0.5, "med": 1.0, "high": 1.5}),
    ("FLAT sizing (all=1.0)",          0.5, 1.0, {"low": 1.0, "med": 1.0, "high": 1.0}),
    ("concentrate .25/1/2.5",          0.5, 1.0, {"low": 0.25, "med": 1.0, "high": 2.5}),
    ("select enter1.0 (skip low)",     1.0, 1.5, {"low": 0.5, "med": 1.0, "high": 1.5}),
    ("select enter1.5 (high only)",    1.5, 1.8, {"low": 0.5, "med": 1.0, "high": 1.5}),
    ("select1.0 + concentrate 0/1/2",  1.0, 1.5, {"low": 0.0, "med": 1.0, "high": 2.0}),
    ("select1.5 + concentrate 0/1/3",  1.5, 1.8, {"low": 0.0, "med": 1.0, "high": 3.0}),
]


def run(enter, flip, conf, equity=3000.0):
    df = load(TF)[["open", "high", "low", "close"]].copy()
    df["spread"] = SPREAD_USD / 0.1
    orig = xt.xau_signal
    xt.xau_signal = make_fast_champion(TF, SPEED)
    try:
        res = xt.run_trades(
            df, equity0=equity, exit_mode="trail", flip_mode="confidence",
            params=dict(conf_risk_scale=conf, risk_frac=0.01,
                        slippage_pips=SLIP_USD / 0.1, spread_cost_mult=1.0,
                        entry_delay_bars=1, enter_thresh=enter, flip_thresh=flip,
                        sl_atr=3.0, trail_atr=3.0))
    finally:
        xt.xau_signal = orig
    return metrics(res, equity, "x")


def main():
    print(f"FAST intraday TREND — sizing/selection sweep  "
          f"({TF}/{SPEED}, spread ${SPREAD_USD}+${SLIP_USD} slip, eval 2017+)\n")
    print(f"{'config':34s} {'Sharpe':>7} {'CI95':>15} {'tr/mo':>6} "
          f"{'DD%':>7} {'PF':>5}  worst-yr")
    for label, en, fl, conf in CONFIGS:
        st = run(en, fl, conf)
        py = st.get("per_year", {})
        worst = min(py.values()) if py else 0.0
        print(f"{label:34s} {st['sharpe']:+7.3f} "
              f"{str(st['ci95']):>15} {st['trades_per_mo']:6.1f} "
              f"{st['max_dd_pct']:7.1f} {st['pf']:5.2f}  {worst:+.2f}")


if __name__ == "__main__":
    main()
