"""v5_xau_fade_exec.py — REAL-order (DEMO-only) executor for the M15 fade signal.

Places actual market orders on the demo account, isolated from the trend bot:
  * DISTINCT magic (360530) — the account is RETAIL_HEDGING, so these positions
    coexist with the trend bot's (magic 360520) as separate tickets; each bot
    manages ONLY its own magic. No netting, no interference.
  * HARD-ABORTS unless the logged-in account is a demo.

Signal (from `scripts/v5_xau_fade_backtest.py`): fade extreme M15 closes, one-bar
hold. Run once per M15 boundary; each pass:
  1. CLOSE any open fade position (its one-bar hold is complete);
  2. read the just-closed bar; if close_pos<lo (long) / >hi (short) AND its hour
     is a good session hour, OPEN a market position for the new bar;
  3. a protective SL is attached as a fail-safe if a later pass is missed.

HONEST NOTE: net-negative long-run at this account's ~$0.34 gold spread (backtest
gross Sharpe 1.9 -> net -7.5). This exists to see the signal trade for real on
demo, not to profit.

    conda run -n envmt5 python scripts/v5_xau_fade_exec.py --execute
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.mt5_connector import MT5Connector
from scripts.v5_xau_demo import require_demo, resolve_symbol

FADE_MAGIC = 360530                       # distinct from trend bot (360520)
JOURNAL = ROOT / "data" / "v5_runs" / "fade_real_trades.csv"
COLS = ["ts", "event", "signal_time", "hour", "dir", "lots", "price",
        "ticket", "profit_acct", "comment"]   # profit in ACCOUNT currency (KES)


def journal(rows: list[dict]) -> None:
    new = not JOURNAL.exists()
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="actually send orders")
    ap.add_argument("--good-hours", default="8,20,22")
    ap.add_argument("--lo-thr", type=float, default=0.2)
    ap.add_argument("--hi-thr", type=float, default=0.8)
    ap.add_argument("--lots", type=float, default=0.01)
    ap.add_argument("--protect-sl-usd", type=float, default=8.0,
                    help="fail-safe SL distance in $ (wide; only catches a missed close)")
    args = ap.parse_args()
    good = {int(x) for x in args.good_hours.split(",")}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    conn = MT5Connector()
    conn.connect()
    rows = []
    try:
        require_demo(conn)                # hard-abort if not demo
        symbol = resolve_symbol(conn)

        # 1) close any open fade position (one-bar hold complete) --------------
        mine = [p for p in (conn.get_positions(magic=FADE_MAGIC) or [])
                if p.symbol == symbol]
        for p in mine:
            pdir = 1 if p.type == 0 else -1
            print(f"  CLOSE fade {p.volume}@dir{pdir:+d} ticket={p.ticket} "
                  f"profit={p.profit:+.2f}")
            if args.execute:
                conn.close_position(p)
            rows.append(dict(ts=ts, event="close", signal_time="", hour="",
                             dir=pdir, lots=p.volume, price="", ticket=p.ticket,
                             profit_acct=round(p.profit, 2), comment="1-bar exit"))

        # 2) evaluate the just-closed bar and open if it fired -----------------
        df = conn.get_rates(symbol, "M15", count=5).sort_index()
        S = df.iloc[-2]                   # last fully-closed bar (df[-1] = forming)
        s_time = df.index[-2]
        rng = float(S.high - S.low)
        cp = (float(S.close) - float(S.low)) / rng if rng > 0 else 0.5
        hr = int(s_time.hour)
        d = 1 if cp < args.lo_thr else (-1 if cp > args.hi_thr else 0)
        fires = d != 0 and hr in good
        print(f"  signal bar {s_time}  hour {hr}  close_pos {cp:.2f}  -> "
              f"{'LONG' if d>0 else 'SHORT' if d<0 else 'flat'}"
              f"{'' if fires else '  (no trade: '+('hour' if d!=0 else 'not extreme')+')'}")

        if fires:
            side = "buy" if d > 0 else "sell"
            tick = conn.get_tick(symbol)
            entry = tick.ask if d > 0 else tick.bid
            sl = round(entry - d * args.protect_sl_usd, 2)
            print(f"  OPEN {side.upper()} {args.lots} {symbol} ~{entry:.2f} SL {sl}")
            if args.execute:
                r = conn.open_position(symbol, side, args.lots, sl=sl,
                                       comment="fade", magic=FADE_MAGIC)
                tk = r.get("order")
            else:
                tk = None
            rows.append(dict(ts=ts, event="open", signal_time=s_time, hour=hr,
                             dir=d, lots=args.lots, price=round(float(entry), 2),
                             ticket=tk, profit_acct="", comment="fade entry"))
    finally:
        conn.disconnect()

    if rows and args.execute:
        journal(rows)
    if JOURNAL.exists():
        j = pd.read_csv(JOURNAL)
        cl = j[j.event == "close"]
        if len(cl):
            tot = cl.profit_acct.sum(); wr = (cl.profit_acct > 0).mean() * 100
            print(f"  cumulative closed: {len(cl)} trades  net {tot:+.2f} (acct ccy)  "
                  f"{wr:.0f}% win")


if __name__ == "__main__":
    main()
