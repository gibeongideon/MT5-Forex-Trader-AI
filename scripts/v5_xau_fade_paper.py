"""v5_xau_fade_paper.py — PARALLEL, VIRTUAL paper executor for the M15 fade signal.

Fully isolated from the live XAU trend bot (v5_xau_demo):
  * places NO broker orders — it only READS M15 prices via the shared bridge, so
    it cannot touch the trend bot's position, equity, or margin;
  * own state file + own CSV journal (separate from live_trades.db);
  * read-only: safe to run alongside anything.

Signal (from the win-rate study, `scripts/v5_xau_fade_backtest.py`):
  fade extreme closes on M15 — close near bar LOW -> long next bar; close near
  bar HIGH -> short next bar; restricted to good session hours. One-bar hold:
  enter at next bar OPEN, exit at that bar CLOSE.

Because entry and exit are the same bar's open/close, each trade is fully
resolved the instant that bar closes — no intra-bar order timing needed. This
pass is idempotent: it processes every newly-closed bar since the last run and
records BOTH gross and net-of-real-spread P&L, so we can watch the spread drag
live (expected net-negative at this account's ~$0.34 gold spread).

    conda run -n envmt5 python scripts/v5_xau_fade_paper.py            # one pass
    conda run -n envmt5 python scripts/v5_xau_fade_paper.py --backfill 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.mt5_connector import MT5Connector
from scripts.v5_xau_demo import resolve_symbol

STATE = ROOT / "data" / "v5_runs" / "fade_paper_state.json"
JOURNAL = ROOT / "data" / "v5_runs" / "fade_paper_trades.csv"
COLS = ["held_time", "signal_time", "hour", "dir", "open", "close",
        "spread_usd", "gross_usd", "cost_usd", "net_usd", "balance", "recorded_at"]


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1, default=str))


def append_rows(rows: list[dict]) -> None:
    new = not JOURNAL.exists()
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good-hours", default="8,20,22")
    ap.add_argument("--lo-thr", type=float, default=0.2)
    ap.add_argument("--hi-thr", type=float, default=0.8)
    ap.add_argument("--stake", type=float, default=1000.0, help="virtual $ notional per trade")
    ap.add_argument("--start-balance", type=float, default=10000.0)
    ap.add_argument("--backfill", type=int, default=0,
                    help="on a fresh state, seed history N bars back (default 0 = start clean)")
    ap.add_argument("--count", type=int, default=400, help="M15 bars to pull")
    args = ap.parse_args()

    good = {int(x) for x in args.good_hours.split(",")}
    conn = MT5Connector()
    conn.connect()
    try:
        symbol = resolve_symbol(conn)
        point = float(conn.symbol_info(symbol).point)
        df = conn.get_rates(symbol, "M15", count=args.count).sort_index()
    finally:
        conn.disconnect()

    closed = df.iloc[:-1]                       # drop the still-forming last bar
    if len(closed) < 3:
        print("not enough closed bars"); return

    st = load_state()
    bal = st.get("balance", args.start_balance)
    last = pd.Timestamp(st["last_time"]) if st.get("last_time") else None
    if last is None:
        # fresh: start clean from the newest closed bar (or seed `backfill` bars)
        seed_i = max(1, len(closed) - args.backfill)
        last = closed.index[seed_i - 1]
        print(f"fresh state: seeding from {last} (backfill {args.backfill})")

    rng = (closed.high - closed.low)
    close_pos = (closed.close - closed.low) / rng.replace(0, pd.NA)
    idx = closed.index
    rows, n_fire = [], 0
    for i in range(1, len(closed)):
        H = idx[i]
        if H <= last:
            continue
        S = idx[i - 1]                          # signal bar = predecessor
        hr = S.hour
        cp = close_pos.iloc[i - 1]
        if pd.isna(cp) or hr not in good:
            continue
        d = 1 if cp < args.lo_thr else (-1 if cp > args.hi_thr else 0)
        if d == 0:
            continue
        o, c = float(closed.open.iloc[i]), float(closed.close.iloc[i])
        spread_usd = float(closed.spread.iloc[i]) * point
        gross = args.stake * d * (c - o) / o
        cost = args.stake * spread_usd / o
        net = gross - cost
        bal += net
        n_fire += 1
        rows.append({"held_time": H, "signal_time": S, "hour": hr, "dir": d,
                     "open": round(o, 2), "close": round(c, 2),
                     "spread_usd": round(spread_usd, 3),
                     "gross_usd": round(gross, 2), "cost_usd": round(cost, 2),
                     "net_usd": round(net, 2), "balance": round(bal, 2),
                     "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    if rows:
        append_rows(rows)
    save_state({"last_time": idx[-1], "balance": bal,
                "start_balance": st.get("start_balance", args.start_balance),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    # summary from the full journal
    print(f"symbol {symbol}  spread now ~${float(df.spread.iloc[-1])*point:.2f}  "
          f"processed up to {idx[-1]}")
    print(f"this pass: {n_fire} new trade(s)")
    if JOURNAL.exists():
        j = pd.read_csv(JOURNAL)
        g, nt = j["gross_usd"].sum(), j["net_usd"].sum()
        wr_g = (j["gross_usd"] > 0).mean() * 100
        print(f"cumulative ({len(j)} trades): GROSS ${g:+.2f} ({wr_g:.0f}% win)  "
              f"NET ${nt:+.2f}  spread drag ${g - nt:.2f}  balance ${bal:.2f}")


if __name__ == "__main__":
    main()
