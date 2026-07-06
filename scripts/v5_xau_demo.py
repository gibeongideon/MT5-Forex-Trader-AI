"""v5_xau_demo.py — DEMO-ONLY MT5 executor for the validated XAUUSD engine.

Phase-4 demo gate for `xau-trend-trail-conf-riskscaled`. Each invocation
(run it once per completed 4H bar, e.g. via cron or /loop):

  1. connects to the running MT5 terminal (./start_mt5.sh) and HARD-ABORTS
     unless the logged-in account is a DEMO account;
  2. resolves the broker's tradable gold symbol (XAUUSD -> XAUUSD.Z etc.);
  3. pulls fresh H4 bars from the terminal and merges them with the
     validated CSV history (same engine, fresh data);
  4. replays the deterministic engine to get the desired state (direction,
     stop, pending action) and sizes lots from the ACTUAL account equity;
  5. reconciles with the broker: open / close / flip / move the trailing SL,
     capped at --max-lot; journals every action.

Default is a dry run (prints the reconciliation plan). Orders are sent only
with --execute, and only on a demo account — there is no override.

    python scripts/v5_xau_demo.py                    # plan only
    python scripts/v5_xau_demo.py --execute          # trade on demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.mt5_connector import MT5Connector
from src.core.trade_journal import TradeJournal
from src.v5.xau_trend import (CONTRACT, PARAMS, confidence_bucket, run_trades,
                              wilder_atr)

CONFIG_FILE = ROOT / "configs" / "v5_xau_trader.json"
CSV = ROOT / "data" / "XAUUSD_H4_long.csv"
RUN_ID = "v5-xau-demo"
SL_TOLERANCE = 0.5  # USD; don't spam modify_position for sub-tolerance moves


def require_demo(conn: MT5Connector):
    info = conn.account_info()
    if info is None:
        raise SystemExit("ABORT: no account logged in on the MT5 terminal")
    is_demo = getattr(info, "trade_mode", None) == 0 or "demo" in str(info.server).lower()
    if not is_demo:
        raise SystemExit(
            f"ABORT: account {info.login} on '{info.server}' is NOT a demo "
            "account. This executor refuses to trade non-demo accounts.")
    print(f"  demo account OK: {info.login} on {info.server}  "
          f"equity ${info.equity:,.2f} {info.currency}")
    return info


def resolve_symbol(conn: MT5Connector, base: str = "XAUUSD") -> str:
    mt5 = conn._mt5
    info = conn.symbol_info(base)
    if info is not None and getattr(info, "trade_mode", 0) == 4:
        mt5.symbol_select(base, True)
        return base
    cands = [s for s in (mt5.symbols_get() or [])
             if s.name[:6].upper() == base and getattr(s, "trade_mode", 0) == 4]
    cands.sort(key=lambda s: (not getattr(s, "visible", False), len(s.name)))
    if not cands:
        raise SystemExit(f"ABORT: no tradable variant of {base} on this account")
    name = cands[0].name
    mt5.symbol_select(name, True)
    print(f"  symbol resolved: {base} -> {name}")
    return name


def fresh_h4(conn: MT5Connector, symbol: str, save: bool) -> pd.DataFrame:
    hist = pd.read_csv(CSV, parse_dates=["time"], index_col="time").sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    live = conn.get_rates(symbol, "H4", count=5000)
    live = live.rename(columns={"tick_volume": "tick_volume"})
    live["spread"] = live["spread"] / 10.0        # broker points -> pip units
    live = live[["open", "high", "low", "close", "tick_volume", "spread"]]
    # drop the still-forming bar: its close is not final
    now = pd.Timestamp.utcnow().tz_localize(None)
    live = live[live.index + pd.Timedelta(hours=4) <= now + pd.Timedelta(hours=3)]
    merged = pd.concat([hist, live])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    n_new = len(merged) - len(hist)
    print(f"  bars: csv {len(hist)} + terminal {len(live)} -> merged "
          f"{len(merged)} ({n_new:+d} new, last {merged.index[-1]})")
    if save and n_new > 0:
        merged.to_csv(CSV)
        print(f"  refreshed {CSV.name}")
    return merged


def desired_state(df: pd.DataFrame, cfg: dict) -> dict:
    res = run_trades(df, exit_mode=cfg["exit_mode"], flip_mode=cfg["flip_mode"],
                     params=cfg.get("params"))
    pos, pending = res["open_position"], res["pending"]
    atr = float(wilder_atr(df, PARAMS["atr_period"]).iloc[-1])
    sig = float(res["signal"].iloc[-1])
    if pending is not None:          # decided at last close; fills now
        d = pending["dir"]
        return dict(dir=d, sl_dist=PARAMS["sl_atr"] * atr,
                    sl=None, conf=confidence_bucket(pending["strength"]),
                    src="pending", forecast=sig)
    if pos is not None:
        return dict(dir=pos["dir"], sl_dist=None, sl=float(pos["sl"]),
                    conf=pos["conf"], src="position", forecast=sig)
    return dict(dir=0, sl=None, sl_dist=None, conf=None, src="flat", forecast=sig)


def size_lots(conn: MT5Connector, symbol: str, direction: int, price: float,
              sl: float, equity_acct_ccy: float, conf: str, vol_min: float,
              vol_step: float, max_lot: float, cfg: dict,
              force_min: bool) -> float:
    """Loss-at-SL computed by the broker in ACCOUNT currency (KES-safe)."""
    mt5 = conn._mt5
    order = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    loss_1lot = mt5.order_calc_profit(order, symbol, 1.0, price, sl)
    if loss_1lot is None or loss_1lot >= 0:
        if force_min:
            print("  ! order_calc_profit unavailable (transient?) — "
                  "falling back to broker min lot (demo gate)")
            return vol_min
        print("  ! order_calc_profit unavailable — refusing to size")
        return 0.0
    loss_1lot = abs(loss_1lot)
    risk = PARAMS["risk_frac"] * cfg["params"]["conf_risk_scale"][conf]
    ideal = (risk * equity_acct_ccy) / loss_1lot
    lots = round(round(ideal / vol_step) * vol_step, 2)
    if lots < vol_min:
        if force_min:
            actual_risk = vol_min * loss_1lot / equity_acct_ccy
            print(f"  ! forced to min lot {vol_min}: actual risk "
                  f"{actual_risk:.1%} of equity (target was {risk:.1%}) — "
                  "demo-gate only")
            return vol_min
        return 0.0
    return min(lots, max_lot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="actually send orders (demo account only)")
    ap.add_argument("--max-lot", type=float, default=0.05)
    ap.add_argument("--force-min-lot", action="store_true",
                    help="if correct sizing rounds to zero, trade the broker "
                         "minimum anyway (demo gate; prints the inflated risk)")
    ap.add_argument("--save-data", action="store_true",
                    help="write merged fresh bars back to the CSV")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    magic = cfg["magic"]
    journal = TradeJournal(args.journal)

    conn = MT5Connector()
    conn.connect()
    try:
        acct = require_demo(conn)
        symbol = resolve_symbol(conn)
        df = fresh_h4(conn, symbol, save=args.save_data)
        want = desired_state(df, cfg)
        print(f"  engine: {want['src']}  forecast {want['forecast']:+.2f}  "
              f"dir {want['dir']:+d}  conf {want['conf']}")

        mine = [p for p in (conn.get_positions(magic=magic) or [])
                if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        if held is not None:
            print(f"  broker: {'LONG' if held_dir > 0 else 'SHORT'} "
                  f"{held.volume} lots @ {held.price_open}  SL {held.sl}  "
                  f"ticket {held.ticket}")
        else:
            print("  broker: flat (magic-filtered)")

        si = conn.symbol_info(symbol)
        tick = conn.get_tick(symbol)
        actions = []

        if want["dir"] == 0 and held is not None:
            actions.append(("close", held, None, None))
        elif want["dir"] != 0:
            if held is not None and held_dir != want["dir"]:
                actions.append(("close", held, None, None))
                held = None
            if held is None:
                price = tick.ask if want["dir"] > 0 else tick.bid
                sl = (want["sl"] if want["sl"] is not None
                      else price - want["dir"] * want["sl_dist"])
                lots = size_lots(conn, symbol, want["dir"], price, sl,
                                 acct.equity, want["conf"],
                                 getattr(si, "volume_min", 0.01),
                                 getattr(si, "volume_step", 0.01),
                                 args.max_lot, cfg, args.force_min_lot)
                if lots > 0:
                    actions.append(("open", None,
                                    "buy" if want["dir"] > 0 else "sell",
                                    dict(lots=lots, sl=round(sl, 2))))
                else:
                    print("  ! sized to zero lots at this equity — no open")
            elif want["sl"] is not None and \
                    abs(float(held.sl or 0.0) - want["sl"]) > SL_TOLERANCE:
                actions.append(("modify_sl", held, None,
                                dict(sl=round(want["sl"], 2))))

        if not actions:
            print("  PLAN: nothing to do (in sync)")
        for act, position, side, extra in actions:
            desc = (f"{act} {side or ''} {extra or ''}"
                    if act != "close" else f"close ticket {position.ticket}")
            print(f"  PLAN: {desc}")
            if not args.execute:
                continue
            try:
                r = None
                if act == "close":
                    r = conn.close_position(position)
                elif act == "open":
                    r = conn.open_position(symbol, side, extra["lots"],
                                           sl=extra["sl"], magic=magic,
                                           comment=RUN_ID)
                else:
                    r = conn.modify_position(position.ticket, sl=extra["sl"],
                                             tp=float(position.tp or 0.0))
            except Exception as exc:  # market closed, requote, etc.
                print(f"    ORDER REJECTED: {exc}")
                print("    (if 10018/market closed: rerun after market open)")
                continue
            journal.record(dict(
                bot="v5_xau_demo", symbol=symbol,
                direction=side or act, entry_time=str(df.index[-1]),
                entry_reason=f"{want['src']} forecast={want['forecast']:+.2f}",
                volume=extra.get("lots", getattr(position, "volume", 0.0))
                if extra else getattr(position, "volume", 0.0),
                sl_pips=extra.get("sl") if extra else None,
                confidence={"low": 0.5, "med": 1.0, "high": 1.5}.get(want["conf"]),
                magic=magic, run_id=RUN_ID, dry_run=0))
            print(f"    EXECUTED: {r.get('retcode', r) if isinstance(r, dict) else r}")
        if not args.execute and actions:
            print("  (dry plan — rerun with --execute to send on the demo account)")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
