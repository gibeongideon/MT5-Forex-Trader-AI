"""v5_overnight_verify.py — verify the OVERNIGHT edge is REAL on the broker's own data.

The overnight sleeve (backtest OOS Sharpe ~1.19) assumes you can trade the close->open
gap. On a 24h CFD there is no real gap and the edge is a mirage. This tool pulls the
BROKER's actual daily bars for the candidate instruments and checks, per instrument:
  * is there a genuine overnight gap, and is it BIGGER than the spread?
  * overnight (close->open) net-of-spread Sharpe / hit-rate vs intraday
  * verdict: REAL tradeable edge, or NO EDGE (24h / gap < cost)

Read-only (no orders). Connect a DEMO account in the terminal first, then:
    conda run -n envmt5 python scripts/v5_overnight_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.mt5_connector import MT5Connector

# candidate broker-name patterns (resolve to whatever the account actually offers)
CANDIDATES = {
    "US500": ["US500", "US500.F", "SPX500", "SP500"],
    "US100": ["US100", "US100.F", "USTEC", "NAS100"],
    "US30":  ["US30", "US30.F", "DJ30", "WS30"],
    "GER40": ["GER40", "GER40.F", "DE40", "DAX40"],
    "JP225": ["JP225", "JPN225", "JP225.F"],
    "AUS200": ["AUS200", "AU200"],
    "GOLD":  ["XAUUSD", "GOLD"],
    "SILVER": ["XAGUSD", "SILVER"],
    "NATGAS": ["NATGAS", "NGAS", "XNGUSD"],
}


def resolve(conn, names):
    for n in names:
        if conn.symbol_info(n) is not None:
            return n
    return None


def stats(df, spread_px):
    c = df["close"]
    o = df["open"]
    sp = spread_px / c
    on = ((o - c.shift(1)) / c.shift(1) - sp).dropna()          # net overnight
    idr = ((c - o) / o - sp).dropna()                            # net intraday
    gap = (o - c.shift(1)).abs()                                 # raw gap size
    gap_vs_spread = float((gap / spread_px).dropna().mean())     # >1 = gap beats cost
    def sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0
    return dict(on_sr=sh(on), on_pos=float((on > 0).mean()), on_ann=float(on.mean() * 252 * 100),
                id_sr=sh(idr), gap_x=gap_vs_spread, n=len(on))


def main():
    conn = MT5Connector()
    conn.connect()
    try:
        a = conn.account_info()
        demo = "demo" in str(getattr(a, "server", "")).lower()
        print(f"account {a.login} @ {a.server}  bal {a.balance} {a.currency}  "
              f"{'[DEMO ok]' if demo else '[NOT a demo — verify only, do not size on this]'}")
        print(f"\n{'instrument':10} {'broker sym':12} {'ON_SR':>6} {'ON_pos%':>7} "
              f"{'ON_ann%':>7} {'gap/spread':>10} {'ID_SR':>6}  verdict")
        rows = []
        for label, names in CANDIDATES.items():
            sym = resolve(conn, names)
            if not sym:
                print(f"{label:10} {'—':12} not available on this account")
                continue
            conn._mt5.symbol_select(sym, True)
            df = conn.get_rates(sym, "D1", count=500)
            if df is None or len(df) < 100:
                print(f"{label:10} {sym:12} insufficient bars")
                continue
            info = conn.symbol_info(sym)
            spread_px = float(getattr(info, "spread", 0)) * float(getattr(info, "point", 0)) or \
                float(df.get("spread", pd.Series([0])).median()) * float(getattr(info, "point", 1e-5))
            if spread_px <= 0:
                spread_px = float((df["high"] - df["low"]).median()) * 0.02   # fallback
            s = stats(df, spread_px)
            real = s["on_sr"] > 0.3 and s["gap_x"] > 1.5 and s["on_ann"] > 0
            verdict = "REAL edge" if real else ("weak" if s["on_sr"] > 0 else "NO EDGE")
            rows.append((label, s, real))
            print(f"{label:10} {sym:12} {s['on_sr']:+6.2f} {s['on_pos']*100:6.0f}% "
                  f"{s['on_ann']:+7.1f} {s['gap_x']:10.1f} {s['id_sr']:+6.2f}  {verdict}")
        real_n = sum(1 for _, _, r in rows if r)
        print(f"\nSUMMARY: {real_n}/{len(rows)} instruments show a REAL tradeable overnight edge "
              f"(net SR>0.3, gap>1.5x spread).")
        print("If >=4 are REAL, the overnight sleeve is worth building. If gaps are ~1x spread "
              "(24h instruments), the backtest 1.19 is a mirage — ship reversal-only instead.")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
