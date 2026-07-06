"""v5_xau_demo.py — DEMO-ONLY MT5 executor for the promoted XAUUSD engine.

Executes the E2b-promoted configuration (`xau-m15-e2b`): the validated H4
EWMAC signal with M15 PULLBACK-LIMIT entries — after a signal, a limit order
is placed 0.5 x ATR_H4 better than market; if unfilled for 24 M15 bars the
engine converts it to a market entry. Broker limits are GTC; TTL expiry is
enforced by these reconcile passes (cancel + market), not by server-side
expiration.

Each invocation (once per completed 4H bar; more often is harmless):
  1. HARD-ABORTS unless the logged-in account is a DEMO account;
  2. resolves the broker's tradable gold symbol;
  3. pulls fresh M15 bars into `data/XAUUSD_M15_spliced.csv` (same HFM
     feed, gate-checked) and refreshes the H4 CSV;
  4. replays the deterministic M15 engine for the desired state: position
     (with trailing SL), working pullback limit, or flat;
  5. reconciles the broker to that state (market / limit / cancel /
     modify-SL), sized from ACTUAL account equity via order_calc_profit,
     capped at --max-lot; every action journaled.

Default prints the plan; orders are sent only with --execute, and only on
a demo account — no override exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.mt5_connector import MT5Connector
from src.core.trade_journal import TradeJournal
from src.v5.xau_m15_exec import run_trades_m15
from src.v5.xau_trend import PARAMS

CONFIG_FILE = ROOT / "configs" / "v5_xau_trader.json"
M15_CSV = ROOT / "data" / "XAUUSD_M15_spliced.csv"
H4_CSV = ROOT / "data" / "XAUUSD_H4_long.csv"
RUN_ID = "v5-xau-demo"
SL_TOLERANCE = 0.5     # USD
LIMIT_TOLERANCE = 0.5  # USD — replace pending if engine limit moved more


def require_demo(conn: MT5Connector):
    info = conn.account_info()
    if info is None:
        raise SystemExit("ABORT: no account logged in on the MT5 terminal")
    is_demo = getattr(info, "trade_mode", None) == 0 or \
        "demo" in str(info.server).lower()
    if not is_demo:
        raise SystemExit(
            f"ABORT: account {info.login} on '{info.server}' is NOT a demo "
            "account. This executor refuses to trade non-demo accounts.")
    print(f"  demo account OK: {info.login} on {info.server}  "
          f"equity {info.equity:,.2f} {info.currency}")
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


def fresh_data(conn: MT5Connector, symbol: str, save: bool) -> pd.DataFrame:
    """Merge fresh terminal M15 bars into the spliced CSV; refresh H4 CSV."""
    hist = pd.read_csv(M15_CSV, parse_dates=["time"], index_col="time").sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    live = conn.get_rates(symbol, "M15", count=65_000)
    live["spread"] = live["spread"] / 10.0
    live = live[["open", "high", "low", "close", "tick_volume", "spread"]]
    now = pd.Timestamp.utcnow().tz_localize(None)
    live = live[live.index + pd.Timedelta(minutes=15) <= now + pd.Timedelta(hours=3)]
    merged = pd.concat([hist, live])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    print(f"  M15 bars: {len(hist)} -> {len(merged)} "
          f"({len(merged) - len(hist):+d} new, last {merged.index[-1]})")
    if save:
        merged.to_csv(M15_CSV)
        try:
            h4l = conn.get_rates(symbol, "H4", count=5000)
            h4l["spread"] = h4l["spread"] / 10.0
            h4l = h4l[["open", "high", "low", "close",
                       "tick_volume", "spread"]].iloc[:-1]
            h4h = pd.read_csv(H4_CSV, parse_dates=["time"],
                              index_col="time").sort_index()
            h4m = pd.concat([h4h, h4l])
            h4m[~h4m.index.duplicated(keep="last")].sort_index().to_csv(H4_CSV)
        except Exception as e:  # noqa: BLE001 — H4 refresh is best-effort
            print(f"  ! H4 CSV refresh skipped: {e}")
    return merged


def size_lots(conn, symbol, direction, price, sl, equity, conf, si, max_lot,
              cfg, force_min):
    mt5 = conn._mt5
    order = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(order, symbol, 1.0, price, sl)
    vol_min = getattr(si, "volume_min", 0.01)
    vol_step = getattr(si, "volume_step", 0.01)
    if loss is None or loss >= 0:
        if force_min:
            print("  ! order_calc_profit unavailable — broker min lot (demo gate)")
            return vol_min
        return 0.0
    loss = abs(loss)
    risk = PARAMS["risk_frac"] * cfg["params"]["conf_risk_scale"][conf]
    lots = round(round((risk * equity) / loss / vol_step) * vol_step, 2)
    if lots < vol_min:
        if force_min:
            print(f"  ! forced to min lot {vol_min}: actual risk "
                  f"{vol_min * loss / equity:.1%} (target {risk:.1%}) — demo gate")
            return vol_min
        return 0.0
    return min(lots, max_lot)


def my_pendings(conn, symbol, magic):
    orders = conn._mt5.orders_get(symbol=symbol) or []
    return [od for od in orders if getattr(od, "magic", 0) == magic]


def place_limit(conn, symbol, direction, lots, price, sl, magic):
    mt5 = conn._mt5
    req = {"action": mt5.TRADE_ACTION_PENDING, "symbol": symbol,
           "volume": lots,
           "type": (mt5.ORDER_TYPE_BUY_LIMIT if direction > 0
                    else mt5.ORDER_TYPE_SELL_LIMIT),
           "price": round(price, 2), "sl": round(sl, 2),
           "deviation": 20, "magic": magic, "comment": RUN_ID,
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_RETURN}
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"limit order failed: retcode="
                           f"{getattr(r, 'retcode', None)} {mt5.last_error()}")
    return r._asdict()


def cancel_pending(conn, ticket):
    mt5 = conn._mt5
    r = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"cancel failed: {getattr(r, 'retcode', None)}")
    return r._asdict()


def build_plan(res, held, held_dir, pendings, tick, si, acct, cfg, args, conn,
               symbol):
    """Reconciliation actions to move the broker to the engine's state."""
    pos, wo = res["open_position"], res["working_order"]
    actions = []
    want_dir = pos["dir"] if pos else (wo["dir"] if wo else 0)

    if held is not None and (want_dir == 0 or
                             (pos is not None and held_dir != pos["dir"])):
        actions.append(("close", dict(position=held, ticket=held.ticket)))
        held, held_dir = None, 0

    if pos is not None and held is None:
        price = tick.ask if pos["dir"] > 0 else tick.bid
        lots = size_lots(conn, symbol, pos["dir"], price, pos["sl"],
                         acct.equity, pos["conf"], si, args.max_lot,
                         cfg, args.force_min_lot)
        if lots > 0:
            actions.append(("open_market", dict(
                dir=pos["dir"], lots=lots, sl=round(pos["sl"], 2),
                why="engine holds position (catch-up)")))
    elif pos is not None and held is not None and \
            abs(float(held.sl or 0) - pos["sl"]) > SL_TOLERANCE:
        actions.append(("modify_sl", dict(position=held, ticket=held.ticket,
                                          sl=round(pos["sl"], 2))))

    want_limit = wo if (wo and wo.get("kind") == "limit" and pos is None
                        and held is None) else None
    if want_limit is not None:
        lim = want_limit["limit"]
        sl = lim - want_limit["dir"] * PARAMS["sl_atr"] * want_limit["atr_h4"]
        match = [od for od in pendings
                 if abs(od.price_open - lim) <= LIMIT_TOLERANCE]
        for od in [od for od in pendings if od not in match]:
            actions.append(("cancel", dict(ticket=od.ticket)))
        if not match:
            lots = size_lots(conn, symbol, want_limit["dir"], lim, sl,
                             acct.equity, "med", si, args.max_lot,
                             cfg, args.force_min_lot)
            if lots > 0:
                actions.append(("place_limit", dict(
                    dir=want_limit["dir"], lots=lots, price=round(lim, 2),
                    sl=round(sl, 2), ttl_left=want_limit.get("ttl"))))
    else:
        for od in pendings:
            actions.append(("cancel", dict(ticket=od.ticket)))

    if wo is not None and wo.get("kind") in ("market", "arm") and \
            pos is None and held is None:
        price = tick.ask if wo["dir"] > 0 else tick.bid
        sl = price - wo["dir"] * PARAMS["sl_atr"] * res["atr_h4_last"]
        lots = size_lots(conn, symbol, wo["dir"], price, sl, acct.equity,
                         "med", si, args.max_lot, cfg, args.force_min_lot)
        if lots > 0:
            actions.append(("open_market", dict(
                dir=wo["dir"], lots=lots, sl=round(sl, 2),
                why="limit TTL expired -> market")))
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-lot", type=float, default=0.05)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    exe = cfg.get("execution", {})
    magic = cfg["magic"]
    journal = TradeJournal(args.journal)

    conn = MT5Connector()
    conn.connect()
    try:
        acct = require_demo(conn)
        symbol = resolve_symbol(conn)
        m15 = fresh_data(conn, symbol, save=args.save_data)
        res = run_trades_m15(
            m15, limit_k=exe.get("limit_k", 0.5), trail_source="h4",
            params={"conf_risk_scale": cfg["params"]["conf_risk_scale"]})
        pos, wo = res["open_position"], res["working_order"]
        state = ("POSITION " + ("LONG" if pos["dir"] > 0 else "SHORT")
                 if pos else
                 "LIMIT working" if wo and wo.get("kind") == "limit" else
                 "MARKET pending" if wo else "flat")
        print(f"  engine: forecast {res['forecast']:+.2f}  {state}")
        if pos:
            print(f"          entry ~{pos['entry']:.2f}  SL {pos['sl']:.2f}"
                  f"{' (trailing)' if pos['trail_on'] else ''}  conf {pos['conf']}")
        if wo and wo.get("kind") == "limit":
            print(f"          limit {wo['limit']:.2f}  ttl {wo.get('ttl')} bars")

        mine = [p for p in (conn.get_positions(magic=magic) or [])
                if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        pendings = my_pendings(conn, symbol, magic)
        print(f"  broker: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        actions = build_plan(res, held, held_dir, pendings,
                             conn.get_tick(symbol), conn.symbol_info(symbol),
                             acct, cfg, args, conn, symbol)
        if not actions:
            print("  PLAN: in sync — nothing to do")
        for act, a in actions:
            printable = {k: v for k, v in a.items() if k != "position"}
            print(f"  PLAN: {act} {printable}")
            if not args.execute:
                continue
            try:
                if act == "close":
                    r = conn.close_position(a["position"])
                elif act == "open_market":
                    r = conn.open_position(
                        symbol, "buy" if a["dir"] > 0 else "sell",
                        a["lots"], sl=a["sl"], magic=magic, comment=RUN_ID)
                elif act == "modify_sl":
                    r = conn.modify_position(a["position"].ticket, sl=a["sl"],
                                             tp=float(a["position"].tp or 0.0))
                elif act == "place_limit":
                    r = place_limit(conn, symbol, a["dir"], a["lots"],
                                    a["price"], a["sl"], magic)
                elif act == "cancel":
                    r = cancel_pending(conn, a["ticket"])
                journal.record(dict(
                    bot="v5_xau_demo", symbol=symbol, direction=act,
                    entry_time=str(m15.index[-1]),
                    entry_reason=json.dumps(printable, default=str)[:180],
                    volume=a.get("lots", 0.0), sl_pips=a.get("sl"),
                    magic=magic, run_id=RUN_ID, dry_run=0))
                print(f"    EXECUTED: "
                      f"{r.get('retcode', r) if isinstance(r, dict) else r}")
            except Exception as exc:  # noqa: BLE001
                print(f"    ORDER REJECTED: {exc}")
        if not args.execute and actions:
            print("  (dry plan — rerun with --execute)")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
