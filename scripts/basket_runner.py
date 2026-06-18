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
from src.cta.sizing import target_lots, min_viable_equity, gross_exposure, DEFAULT_CONTRACT, DEFAULT_VOL

STATE = ROOT / "data" / "basket_state.json"
HISTORY = ROOT / "data" / "basket_signals.csv"
STALE_DAYS = 4

# alias → HFM symbol. VERIFY the exact tickers in the live terminal (HFM appends .Z to some
# tradables, e.g. XAUUSD.Z). All five trade on HFM — UST10Y maps to HFM's US 10Y T-Note bond CFD.
MT5_MAP = {
    "GOLD":   "XAUUSD",
    "EURUSD": "EURUSD",
    "WTI":    "USOIL",     # HFM crude CFD — confirm exact ticker (USOIL/WTI)
    "SPX":    "US500",     # HFM S&P500 cash CFD — confirm exact ticker
    "UST10Y": "US10YR",    # HFM US 10-Year T-Note bond CFD (1 lot=100u, 1:50, spread ~0.06)
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


def _build_specs(kept, close, live):
    """Per-alias {symbol,contract_size,price,vol_*}. live=query the broker (exact), else offline
    (panel close + DEFAULT_CONTRACT — indicative, VERIFY in terminal)."""
    specs, conn = {}, None
    if live:
        from src.core.connector import get_connector
        conn = get_connector("mt5"); conn.connect()
    for a in kept:
        sym = MT5_MAP.get(a) or a
        if live:
            try:
                si = conn.symbol_info(sym); tk = conn.get_tick(sym)
                specs[a] = dict(symbol=sym,
                                contract_size=float(getattr(si, "trade_contract_size", DEFAULT_CONTRACT.get(sym, 1.0))),
                                price=float((tk.ask + tk.bid) / 2) or float(close.iloc[-1][a]),
                                vol_min=float(getattr(si, "volume_min", 0.01)),
                                vol_step=float(getattr(si, "volume_step", 0.01)),
                                vol_max=float(getattr(si, "volume_max", 1e6)))
                continue
            except Exception as e:
                print(f"  ⚠ live spec for {sym} failed ({e}) — falling back to offline")
        specs[a] = dict(symbol=sym, contract_size=DEFAULT_CONTRACT.get(sym, 1.0),
                        price=float(close[a].dropna().iloc[-1]), **DEFAULT_VOL)
    if conn:
        conn.disconnect()
    return specs


def _print_sizing(units, kept, close, equity, live):
    specs = _build_specs(kept, close, live)
    res = target_lots(units, equity, specs)
    src = "LIVE broker specs" if live else "OFFLINE (panel close + default contract sizes — VERIFY in terminal)"
    print(f"\n  ── LOTS for equity ${equity:,.0f}  [{src}] ──")
    print(f"  {'alias':8} {'symbol':8} {'price':>10} {'ideal':>8} {'LOTS':>7} {'notional$':>11} {'err':>6}  flag")
    for a in kept:
        r = res[a]; flag = "ROUND→0" if r["rounded_zero"] else ("CAPPED" if r["capped"] else "")
        print(f"  {a:8} {r['symbol']:8} {specs[a]['price']:>10.2f} {r['ideal_lots']:>+8.3f} "
              f"{r['lots']:>+7.2f} {r['actual_notional']:>+11,.0f} {r['err_frac']*100:>+5.1f}%  {flag}")
    g = gross_exposure(res)
    mve = min_viable_equity(units, specs)
    print(f"  gross notional ${g['gross_notional']:,.0f}  (={g['gross_notional']/equity:.2f}× equity)  "
          f"net ${g['net_notional']:+,.0f}")
    zeros = [a for a in kept if res[a]["rounded_zero"]]
    if zeros:
        print(f"  ⚠ legs rounding to ZERO at this equity: {zeros}")
    print(f"  ⚠ min viable equity for all 5 legs ≥ 1 min-lot: ${mve:,.0f}"
          + ("  (current OK)" if equity >= mve else "  (UNDER — vol target distorted)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="recompute full-period net Sharpe to confirm == backtest (+0.746)")
    ap.add_argument("--equity", type=float, default=None,
                    help="account equity (USD) → print target LOTS per symbol")
    ap.add_argument("--live", action="store_true",
                    help="with --equity: query the MT5 terminal for exact contract specs + price")
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
          f"(positions are vol-scaled units; pass --equity to convert to lots)")

    if args.equity:
        _print_sizing({a: float(today[a]) for a in kept}, kept, close, args.equity, args.live)

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
