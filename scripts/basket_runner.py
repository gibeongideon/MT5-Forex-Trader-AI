"""basket_runner.py — daily target-position runner for the locked 5-instrument CTA basket.

Computes TODAY's target positions for GOLD/UST10Y/SPX/WTI/EURUSD using the exact validated
champion pipeline (src/cta/strategy.champion_positions), reports a per-symbol order ticket vs
the last run, persists state (restart-proof), and appends an audit-trail CSV.

ADVISORY ONLY — it does not place live orders (standing rule: never auto-run live bots). It
prints what to hold; a separate executor / you act on it. Refresh the daily CSVs first
(scripts/download_universe.py) so the signal uses current data.

Usage:
    python scripts/basket_runner.py                 # compute + report today's targets
    python scripts/basket_runner.py --validate      # also re-check full-period Sharpe == backtest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.panel import build_panels, daily_returns, asset_classes, pip_series
from src.cta.strategy import champion_positions, BASKET, CONFIG
from src.cta.pnl import portfolio_pnl

STATE = ROOT / "data" / "basket_state.json"
HISTORY = ROOT / "data" / "basket_signals.csv"
STALE_DAYS = 4

# alias → broker symbol. VERIFY these exist on your broker before trading (HFM uses .Z suffix
# for some tradables). UST10Y has no common retail CFD — left None (drop or substitute a bond ETF/future).
MT5_MAP = {
    "GOLD":   "XAUUSD",
    "EURUSD": "EURUSD",
    "WTI":    "USOIL",     # HFM crude CFD — confirm exact ticker
    "SPX":    "US500",     # S&P500 cash CFD — confirm exact ticker
    "UST10Y": None,        # no standard retail CFD — verify/substitute or trade model-only
}


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(date: str, positions: dict):
    STATE.write_text(json.dumps({"date": date, "positions": positions}, indent=2))


def _append_history(date: str, positions: dict):
    row = pd.DataFrame([{"date": date, **positions}])
    if HISTORY.exists():
        row.to_csv(HISTORY, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY, index=False)


def _action(prev: float, tgt: float, eps: float = 1e-6) -> str:
    if abs(tgt) < eps and abs(prev) < eps:    return "flat"
    if abs(prev) < eps:                        return "OPEN"
    if abs(tgt) < eps:                         return "CLOSE"
    if np.sign(tgt) != np.sign(prev):          return "FLIP"
    if abs(tgt) > abs(prev) + eps:             return "ADD"
    if abs(tgt) < abs(prev) - eps:             return "TRIM"
    return "hold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="recompute full-period net Sharpe to confirm == backtest (+0.746)")
    args = ap.parse_args()

    close, spread, kept = build_panels(BASKET, "D1")
    missing = [a for a in BASKET if a not in kept]
    returns = daily_returns(close)
    classes = asset_classes(kept)
    pos = champion_positions(close, returns, classes,
                             target_vol=CONFIG["target_vol"], trend_speeds=CONFIG["trend_speeds"],
                             risk=CONFIG["risk"], rebalance=CONFIG["rebalance"], buffer=CONFIG["buffer"])

    last_date = pos.index[-1]
    today = pos.iloc[-1]
    age = (pd.Timestamp.utcnow().tz_localize(None) - last_date).days
    state = _load_state()
    prev = state.get("positions", {})

    print(f"\n{'='*78}\n  CTA BASKET — target positions as of {last_date.date()}  "
          f"(config: {CONFIG['trend_speeds']} speeds, buffer {CONFIG['buffer']}, "
          f"{CONFIG['rebalance']} rebalance, {CONFIG['target_vol']:.0%} vol)\n{'='*78}")
    if missing:
        print(f"  ⚠ missing data (excluded): {missing}")
    if age > STALE_DAYS:
        print(f"  ⚠ DATA IS {age}d STALE — refresh data/*_D1_long.csv (scripts/download_universe.py) before trading")

    gross = today.abs().sum()
    print(f"\n  {'alias':8} {'broker':9} {'target':>8} {'weight':>7} {'dir':>6} {'action':>7}  {'prev':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*6} {'-'*7}  {'-'*8}")
    out_positions = {}
    for a in kept:
        tgt = float(today[a]); pv = float(prev.get(a, 0.0))
        out_positions[a] = round(tgt, 4)
        w = (abs(tgt) / gross * 100) if gross > 0 else 0.0
        d = "LONG" if tgt > 1e-6 else "SHORT" if tgt < -1e-6 else "flat"
        sym = MT5_MAP.get(a) or "(model-only)"
        print(f"  {a:8} {sym:9} {tgt:>+8.3f} {w:>6.0f}% {d:>6} {_action(pv,tgt):>7}  {pv:>+8.3f}")

    print(f"\n  gross exposure (Σ|pos|) = {gross:.2f}   net = {today.sum():+.2f}   "
          f"(positions are vol-scaled units; convert to lots via your contract specs)")

    if args.validate:
        pnl = portfolio_pnl(pos, returns, spread, pip_series(kept), close)
        net = pnl["net"]
        sh = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
        nc = net[net.index >= "2022-01-01"]
        shc = float(nc.mean() / nc.std(ddof=1) * np.sqrt(252))
        print(f"\n  [validate] full net Sharpe={sh:+.3f}  confirm={shc:+.3f}  "
              f"(expect full≈+0.75) turnover={pnl['turnover'].mean()*252*100:.0f}%/yr")

    _save_state(str(last_date.date()), out_positions)
    _append_history(str(last_date.date()), out_positions)
    print(f"\n  state → {STATE.name}   history appended → {HISTORY.name}\n  (advisory only — no live orders placed)")


if __name__ == "__main__":
    main()
