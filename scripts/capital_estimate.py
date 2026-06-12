"""
capital_estimate.py — Dollar return estimates at different capital levels.

Uses the verified OOS walk-forward metrics (no-lookahead) to project
expected monthly and annual returns in USD at different starting capitals.

Three scenarios:
  Optimistic  — verified OOS (full encoder, upper bound)
  Base        — WF OOS Sharpe (80% encoder, most honest)
  Conservative — 40% of WF metrics (live-trading discount: slippage,
                 regime change, stale model, real execution)

Usage:
    conda run -n envmt5 python scripts/capital_estimate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Metrics from OOS verification ─────────────────────────────────────────────

# Per-symbol verified metrics (from verify_candle_oos.py run)
METRICS = {
    "EURUSD": dict(
        pip_size          = 0.0001,
        pip_value_per_lot = 10.0,        # USD per pip per standard lot
        sl_pips           = 10.0,
        tp_pips           = 30.0,
        spread_comm_pips  = 1.5,         # spread 1.0 + commission 0.5
        risk_pct          = 0.01,
        # OOS verified (full encoder)
        opt_trades_per_yr = 1092 / 2.4,  # ≈ 455/year
        opt_win_rate      = 0.783,
        # WF reported (80% encoder — most honest)
        wf_trades_per_yr  = 701  / 2.4,  # ≈ 292/year
        wf_win_rate       = 0.502,
        wf_max_dd_pct     = 0.090,       # 9.0% from WF
        # Conservative: 40% of WF metrics
        con_trades_per_yr = (701 / 2.4) * 0.40,
        con_win_rate      = 0.502,
    ),
    "USDJPY": dict(
        pip_size          = 0.01,
        pip_value_per_lot = 6.25,        # USD per pip at ~160 USDJPY
        sl_pips           = 10.0,
        tp_pips           = 30.0,
        spread_comm_pips  = 1.5,
        risk_pct          = 0.01,
        # OOS verified
        opt_trades_per_yr = 3241 / 2.4,  # ≈ 1350/year
        opt_win_rate      = 0.712,
        # WF reported
        wf_trades_per_yr  = 1797 / 2.4,  # ≈ 749/year
        wf_win_rate       = 0.569,
        wf_max_dd_pct     = 0.140,       # 14.0%
        # Conservative
        con_trades_per_yr = (1797 / 2.4) * 0.40,
        con_win_rate      = 0.569,
    ),
}

CAPITAL_LEVELS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]


def expected_pips_per_trade(win_rate: float, tp: float, sl: float, cost: float) -> float:
    return win_rate * (tp - cost) + (1 - win_rate) * -(sl + cost)


def annual_return(capital: float, trades_per_yr: float, win_rate: float,
                  tp: float, sl: float, cost: float,
                  pip_value: float, risk_pct: float) -> dict:
    """Non-compounding annual dollar return estimate."""
    lot = (capital * risk_pct) / (sl * pip_value)
    lot = max(0.01, round(lot / 0.01) * 0.01)  # snap to nearest 0.01, minimum 0.01
    at_min_lot = lot <= 0.01 and (capital * risk_pct) / (sl * pip_value) < 0.01

    e_pips = expected_pips_per_trade(win_rate, tp, sl, cost)
    dollar_per_trade = e_pips * pip_value * lot
    annual_usd = trades_per_yr * dollar_per_trade
    annual_pct = annual_usd / capital * 100
    monthly_usd = annual_usd / 12

    return dict(
        lot          = lot,
        at_min_lot   = at_min_lot,
        annual_usd   = annual_usd,
        annual_pct   = annual_pct,
        monthly_usd  = monthly_usd,
        e_pips       = e_pips,
    )


def print_symbol_table(symbol: str) -> None:
    m = METRICS[symbol]
    tp   = m["tp_pips"]
    sl   = m["sl_pips"]
    cost = m["spread_comm_pips"]
    pv   = m["pip_value_per_lot"]
    r    = m["risk_pct"]

    scenarios = [
        ("Optimistic",   m["opt_trades_per_yr"], m["opt_win_rate"]),
        ("WF Base",      m["wf_trades_per_yr"],  m["wf_win_rate"]),
        ("Conservative", m["con_trades_per_yr"], m["con_win_rate"]),
    ]

    for sc_name, trades_yr, wr in scenarios:
        e_pips_trade = expected_pips_per_trade(wr, tp, sl, cost)

        print(f"\n  ── {symbol}  [{sc_name}] ─────────────────────────────────────────")
        print(f"     Trades/year: {trades_yr:.0f}   Win rate: {wr:.1%}   "
              f"E[pips/trade]: {e_pips_trade:+.2f}")
        print()
        print(f"  {'Capital':>12}  {'Lots':>6}  {'MinLot?':>8}  "
              f"{'Annual $':>11}  {'Annual %':>9}  {'Monthly $':>10}  "
              f"{'MaxDD $':>10}  {'MaxDD %':>8}")
        print(f"  {'─'*82}")

        for cap in CAPITAL_LEVELS:
            res = annual_return(cap, trades_yr, wr, tp, sl, cost, pv, r)
            dd_pct = m["wf_max_dd_pct"] if "wf" in sc_name.lower() else m["wf_max_dd_pct"]
            # Conservative MaxDD is same % (slightly higher in live)
            if sc_name == "Conservative":
                dd_pct = m["wf_max_dd_pct"] * 1.5  # 50% worse in live
            dd_usd = cap * dd_pct
            min_flag = " ← floor" if res["at_min_lot"] else ""

            print(f"  ${cap:>11,}  {res['lot']:>6.2f}  "
                  f"{'YES'+min_flag:>8}  " if res["at_min_lot"] else
                  f"  ${cap:>11,}  {res['lot']:>6.2f}  "
                  f"{'':>8}  "
                  , end="")
            print(f"${res['annual_usd']:>10,.0f}  {res['annual_pct']:>8.1f}%  "
                  f"${res['monthly_usd']:>9,.0f}  "
                  f"${dd_usd:>9,.0f}  {dd_pct*100:>7.1f}%")


def print_symbol_table_clean(symbol: str) -> None:
    m = METRICS[symbol]
    tp   = m["tp_pips"]
    sl   = m["sl_pips"]
    cost = m["spread_comm_pips"]
    pv   = m["pip_value_per_lot"]
    r    = m["risk_pct"]

    scenarios = [
        ("Optimistic (upper bound)",  m["opt_trades_per_yr"], m["opt_win_rate"],  m["wf_max_dd_pct"]),
        ("WF Base (most honest)",     m["wf_trades_per_yr"],  m["wf_win_rate"],   m["wf_max_dd_pct"]),
        ("Conservative (live est.)",  m["con_trades_per_yr"], m["con_win_rate"],  m["wf_max_dd_pct"] * 1.5),
    ]

    W = 96
    print(f"\n{'═'*W}")
    print(f"  {symbol}  —  SL={sl}p  TP={tp}p  Risk=1%/trade  "
          f"Spread+Comm={cost}p  PipValue=${pv}/lot")
    print(f"{'═'*W}")

    for sc_name, trades_yr, wr, dd_pct in scenarios:
        e_pips = expected_pips_per_trade(wr, tp, sl, cost)
        print(f"\n  [{sc_name}]  {trades_yr:.0f} trades/yr  "
              f"WinRate {wr:.1%}  E[{e_pips:+.1f}p/trade]")
        print(f"  {'Capital':>12}  {'Lot/trade':>10}  "
              f"{'Ann. Return':>14}  {'Ann. Return%':>13}  "
              f"{'Monthly $':>10}  {'Max DD $':>10}  {'Max DD%':>8}")
        print(f"  {'─'*90}")

        for cap in CAPITAL_LEVELS:
            res = annual_return(cap, trades_yr, wr, tp, sl, cost, pv, r)
            dd_usd = cap * dd_pct
            effective_risk_pct = (res["lot"] * sl * pv) / cap * 100
            flag = " ⚠ floor" if res["at_min_lot"] else ""

            print(f"  ${cap:>11,}  {res['lot']:>9.2f}  "
                  f"  ${res['annual_usd']:>11,.0f}  {res['annual_pct']:>12.1f}%  "
                  f"${res['monthly_usd']:>9,.0f}  "
                  f"${dd_usd:>9,.0f}  {dd_pct*100:>7.1f}%{flag}")


def main() -> None:
    print(f"\n{'═'*96}")
    print(f"  CANDLE PREDICTOR — RETURN ESTIMATES BY CAPITAL")
    print(f"  Based on 2.4 years OOS walk-forward (Jan 2024 – Jun 2026)")
    print(f"  Compounding NOT applied — all figures are flat-rate (non-compounded) annual")
    print(f"  Actual compounded returns would be MUCH higher but unrealistic to project")
    print(f"{'═'*96}")

    print(f"""
  SCENARIO DEFINITIONS:
  ─────────────────────────────────────────────────────────────────────────────
  Optimistic   — verified OOS using full encoder (upper bound, some encoder leak)
  WF Base      — walk-forward OOS using 80%-data encoder (most honest estimate)
  Conservative — 40% of WF trade count (live slippage, regime drift, stale model)
  ─────────────────────────────────────────────────────────────────────────────
  ⚠  Min lot floor: below ~$100 (EURUSD) or ~$63 (USDJPY) the 1% risk rule
     hits the 0.01 minimum lot → effective risk becomes >1%, return% overstated
  ─────────────────────────────────────────────────────────────────────────────
""")

    for sym in ["EURUSD", "USDJPY"]:
        print_symbol_table_clean(sym)

    print(f"\n{'═'*96}")
    print(f"  COMBINED (EURUSD + USDJPY running simultaneously)")
    print(f"{'═'*96}")

    combined_scenarios = [
        ("Optimistic",   "opt"),
        ("WF Base",      "wf"),
        ("Conservative", "con"),
    ]

    for sc_name, key in combined_scenarios:
        print(f"\n  [{sc_name}]")
        print(f"  {'Capital':>12}  {'EURUSD$/yr':>13}  {'USDJPY$/yr':>13}  "
              f"{'COMBINED$/yr':>14}  {'Combined%':>10}  {'Monthly $':>10}")
        print(f"  {'─'*80}")

        eu = METRICS["EURUSD"]
        uj = METRICS["USDJPY"]
        trades_eu = eu[f"{key}_trades_per_yr"]
        wr_eu     = eu[f"{key}_win_rate"]
        trades_uj = uj[f"{key}_trades_per_yr"]
        wr_uj     = uj[f"{key}_win_rate"]

        for cap in CAPITAL_LEVELS:
            res_eu = annual_return(cap, trades_eu, wr_eu,
                                   eu["tp_pips"], eu["sl_pips"], eu["spread_comm_pips"],
                                   eu["pip_value_per_lot"], eu["risk_pct"])
            res_uj = annual_return(cap, trades_uj, wr_uj,
                                   uj["tp_pips"], uj["sl_pips"], uj["spread_comm_pips"],
                                   uj["pip_value_per_lot"], uj["risk_pct"])

            combined_annual = res_eu["annual_usd"] + res_uj["annual_usd"]
            combined_pct    = combined_annual / cap * 100
            combined_monthly = combined_annual / 12

            print(f"  ${cap:>11,}  ${res_eu['annual_usd']:>11,.0f}  "
                  f"${res_uj['annual_usd']:>11,.0f}  "
                  f"  ${combined_annual:>11,.0f}  {combined_pct:>9.1f}%  "
                  f"${combined_monthly:>9,.0f}")

    print(f"\n{'═'*96}")
    print(f"  IMPORTANT DISCLAIMERS:")
    print(f"  1. These are NON-COMPOUNDED estimates. Actual compounding would be far higher.")
    print(f"  2. Conservative scenario is the most realistic for live trading.")
    print(f"  3. Past OOS performance does not guarantee future results.")
    print(f"  4. Both bots trade the SAME account → combined capital exposure is doubled.")
    print(f"     Size your risk accordingly (e.g. use 0.5% risk/trade instead of 1%)")
    print(f"{'═'*96}\n")


if __name__ == "__main__":
    main()
