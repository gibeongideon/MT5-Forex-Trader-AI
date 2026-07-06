"""v5_xau_watch.py — live console monitor for the XAUUSD demo position.

Read-only: polls the MT5 bridge every few seconds and prints one status
line — bid/ask, the bot's position (magic 360520), floating PnL in account
currency, and distance to the stop. Recomputes the engine forecast when a
new 4H bar completes. Ctrl+C to stop; never sends orders.

    conda run -n envmt5 python scripts/v5_xau_watch.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.mt5_connector import MT5Connector
from src.v5.xau_trend import run_trades

CONFIG_FILE = ROOT / "configs" / "v5_xau_trader.json"
CSV = ROOT / "data" / "XAUUSD_H4_long.csv"


def engine_forecast(cfg: dict) -> float:
    df = pd.read_csv(CSV, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    res = run_trades(df, exit_mode=cfg["exit_mode"], flip_mode=cfg["flip_mode"],
                     params=cfg.get("params"))
    return float(res["signal"].iloc[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    magic = cfg["magic"]
    conn = MT5Connector()
    conn.connect()
    forecast = engine_forecast(cfg)
    seen_bar = None
    print(f"watching XAUUSD (magic {magic})  forecast {forecast:+.2f}  "
          f"— Ctrl+C to stop\n")
    try:
        while True:
            tick = conn.get_tick("XAUUSD")
            positions = [p for p in (conn.get_positions(magic=magic) or [])
                         if p.symbol.startswith("XAUUSD")]
            now = datetime.now().strftime("%H:%M:%S")
            if not positions:
                line = (f"{now}  bid {tick.bid:9.2f}  ask {tick.ask:9.2f}  "
                        f"flat  forecast {forecast:+.2f}")
            else:
                p = positions[0]
                side = "LONG " if p.type == 0 else "SHORT"
                mark = tick.ask if p.type == 1 else tick.bid
                to_sl = abs(mark - p.sl) if p.sl else float("nan")
                line = (f"{now}  bid {tick.bid:9.2f}  {side} {p.volume:.2f} "
                        f"@ {p.price_open:.2f}  SL {p.sl:.2f} "
                        f"({to_sl:6.2f} away)  PnL {p.profit:+10.2f} "
                        f"{'KES' if p.profit else ''}  forecast {forecast:+.2f}")
            print(line, flush=True)
            # refresh forecast when a new completed H4 bar appears
            bars = conn.get_rates("XAUUSD", "H4", count=2)
            latest_completed = bars.index[-2]
            if seen_bar is None:
                seen_bar = latest_completed
            elif latest_completed != seen_bar:
                seen_bar = latest_completed
                forecast = engine_forecast(cfg)
                print(f"--- new 4H bar {latest_completed}: forecast refreshed "
                      f"{forecast:+.2f} (run v5_xau_demo.py --execute to "
                      f"reconcile) ---")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
