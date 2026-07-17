"""Spread frontier for the fast intraday trend bot WITH conviction selection.
For each candidate account spread, find the best entry-threshold and report the
achievable net Sharpe + drawdown. Answers: 'what spread do I need?'.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.v5.xau_trend as xt
from scripts.v5_xau_fast_trend_discrete import make_fast_champion, load, metrics

TF, SPEED, SLIP = "M30", "fast", 0.02
CONF = {"low": 0.5, "med": 1.0, "high": 1.5}
SPREADS = [0.06, 0.10, 0.14, 0.18, 0.24, 0.30, 0.36]
THRESHOLDS = [(0.5, 1.0), (1.0, 1.5), (1.5, 1.8)]


def run(spread, enter, flip, equity=3000.0):
    df = load(TF)[["open", "high", "low", "close"]].copy()
    df["spread"] = spread / 0.1
    orig = xt.xau_signal
    xt.xau_signal = make_fast_champion(TF, SPEED)
    try:
        res = xt.run_trades(df, equity0=equity, exit_mode="trail",
            flip_mode="confidence",
            params=dict(conf_risk_scale=CONF, risk_frac=0.01,
                        slippage_pips=SLIP / 0.1, spread_cost_mult=1.0,
                        entry_delay_bars=1, enter_thresh=enter, flip_thresh=flip,
                        sl_atr=3.0, trail_atr=3.0))
    finally:
        xt.xau_signal = orig
    return metrics(res, equity, "x")


def main():
    print(f"SPREAD FRONTIER — fast intraday trend + conviction sizing "
          f"({TF}/{SPEED}, +${SLIP} slip, eval 2017+)\n")
    print(f"{'spread$':>8} | best net Sharpe (enter thr) | tr/mo | DD%  | worst-yr")
    print("-" * 68)
    for sp in SPREADS:
        best = None
        for en, fl in THRESHOLDS:
            st = run(sp, en, fl)
            if best is None or st["sharpe"] > best["sharpe"]:
                best = {**st, "enter": en}
        py = best.get("per_year", {})
        worst = min(py.values()) if py else 0.0
        tag = ""
        if best["sharpe"] >= 0.9: tag = "  <-- matches champion"
        elif best["sharpe"] >= 0.8: tag = "  <-- deployable"
        print(f"{sp:8.2f} | {best['sharpe']:+6.3f}  (enter {best['enter']}) "
              f"        | {best['trades_per_mo']:5.1f} | {best['max_dd_pct']:5.1f}"
              f"| {worst:+.2f}{tag}")


if __name__ == "__main__":
    main()
