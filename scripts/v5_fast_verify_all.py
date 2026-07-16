"""v5_fast_verify_all.py — verify ALL fast signals on the BROKER's real tradeable
instruments (not downloaded cash data). Read-only. Connect a DEMO with indices first.

Per instrument, net-of-REAL-spread Sharpe (2017+ if available) for:
  ON   overnight (close[t-1]->open[t])
  ID   intraday (open[t]->close[t])
  R1/R2/R5  short-term reversal (fade own N-day close-to-close move)
  BH   buy-and-hold (drift reference — to catch signals that are just re-packaged long)
Then builds equal-risk ensembles of the POSITIVE-net signals and checks correlation
to the trend book.  A signal only counts if net SR>0.3 AND not ~= buy-hold (distinct).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.mt5_connector import MT5Connector

CANDIDATES = {
    "US500": ["US500", "US500.F", "SPX500"], "US100": ["US100", "US100.F", "USTEC"],
    "US30": ["US30", "US30.F"], "GER40": ["GER40", "GER40.F"], "JP225": ["JP225", "JPN225"],
    "AUS200": ["AUS200"], "GOLD": ["XAUUSD"], "SILVER": ["XAGUSD"],
}


def resolve(conn, names):
    for n in names:
        if conn.symbol_info(n) is not None:
            return n
    return None


def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 100 and x.std() > 0 else 0.0


def signals(df, sp):
    """Return dict of net-of-spread daily return streams for each fast signal."""
    c, o = df["close"], df["open"]
    ret = c.pct_change()
    cost = sp / c
    out = {}
    out["ON"] = ((o - c.shift(1)) / c.shift(1) - cost)                     # overnight
    out["ID"] = ((c - o) / o - cost)                                       # intraday
    for lb in (1, 2, 5):
        pos = -np.sign(ret.rolling(lb).sum()).shift(1)                     # fade N-day
        out[f"R{lb}"] = pos * ret - pos.diff().abs().fillna(0) * cost
    out["BH"] = ret                                                        # buy-hold ref
    return {k: v.dropna() for k, v in out.items()}


def z(d, tv=0.10):
    sd = d.std() * np.sqrt(252)
    return d * (tv / sd) if sd > 0 else d


def main():
    conn = MT5Connector()
    conn.connect()
    try:
        a = conn.account_info()
        print(f"account {a.login} @ {a.server}  [{'DEMO' if 'demo' in str(a.server).lower() else 'LIVE'}]")
        cols = ["ON", "ID", "R1", "R2", "R5", "BH"]
        print(f"\n{'instr':7} {'sym':10} " + " ".join(f"{c:>6}" for c in cols))
        allsig = {c: {} for c in cols}
        for label, names in CANDIDATES.items():
            sym = resolve(conn, names)
            if not sym:
                continue
            conn._mt5.symbol_select(sym, True)
            df = conn.get_rates(sym, "D1", count=1500)
            if df is None or len(df) < 200:
                continue
            info = conn.symbol_info(sym)
            sp = float(getattr(info, "spread", 0)) * float(getattr(info, "point", 0))
            if sp <= 0:
                sp = float((df["high"] - df["low"]).median()) * 0.02
            sig = signals(df, sp)
            row = {k: sh(v.loc["2017-01-01":]) for k, v in sig.items()}
            print(f"{label:7} {sym:10} " + " ".join(f"{row[c]:+6.2f}" for c in cols))
            for c in cols:
                allsig[c][label] = sig[c].loc["2016-01-01":]
        # ensembles of each signal-type across instruments (equal risk), net
        print(f"\n{'signal':8} {'ensemble SR':>11} {'vs buy-hold':>12}  note")
        bh_ens = z(sum(z(v.fillna(0)) for v in allsig["BH"].values()) / max(len(allsig["BH"]), 1))
        for c in ["ON", "ID", "R1", "R2", "R5"]:
            d = allsig[c]
            if not d:
                continue
            ens = z(sum(z(v.fillna(0)) for v in d.values()) / len(d))
            j = pd.concat([ens, bh_ens], axis=1).dropna()
            corr = j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) > 100 else np.nan
            ens_sr = sh(ens.loc["2017-01-01":])
            note = "just drift" if corr > 0.6 else ("distinct" if ens_sr > 0.3 else "no edge")
            print(f"{c:8} {ens_sr:+11.2f} {corr:+12.2f}  {note}")
        print("\nRULE: a fast signal is REAL only if ensemble SR>0.3 AND corr-to-buy-hold<0.6 (distinct from just being long).")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
