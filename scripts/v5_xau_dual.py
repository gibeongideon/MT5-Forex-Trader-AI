"""v5_xau_dual.py — MT5 executor for the two 2026-07-14 H4 XAUUSD bots.

    --bot ls     long/short trend+breakout ensemble   (magic 360541)
    --bot champ  LONG-ONLY concentrated blend champion (magic 360542)

Thin sibling of v5_xau_live.py: reuses v5_xau_demo.py's validated helpers
(symbol resolution, H4 CSV refresh, order plumbing) and the SAME double
safety lock — a real account trades only with BOTH --live and --execute.

Engine: src.v5.xau_trend.run_trades (trade tickets, ATR trail) replayed on
the H4 CSV with the bot's signal from src.v5.xau_dual_signals patched in;
the broker is then reconciled to the engine's end state (market open /
close / modify-SL; this executor never places limit orders). Sizing is
risk_frac x conf_scale of ACTUAL equity over the 3xATR stop via
order_calc_profit, so it is correct on the cent account (XAUUSDc).

Run both bots in one pass (the xau-dual systemd timer does):
    python scripts/v5_xau_dual.py --bot ls    --live --execute --save-data
    python scripts/v5_xau_dual.py --bot champ --live --execute
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "v5_xau_demo", ROOT / "scripts" / "v5_xau_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

import src.v5.xau_trend as xt  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402
from src.core.trade_journal import TradeJournal  # noqa: E402
from src.v5.xau_dual_signals import SIGNALS  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_xau_dual.json"
SL_TOLERANCE = 0.5  # USD


def require_live(conn: MT5Connector, allow_live: bool):
    """Demo accounts run freely; real accounts require --live (mirror of
    v5_xau_live.require_live)."""
    info = conn.account_info()
    if info is None:
        raise SystemExit("ABORT: no account logged in on the MT5 terminal")
    is_demo = getattr(info, "trade_mode", None) == 0 or \
        "demo" in str(info.server).lower()
    if not is_demo and not allow_live:
        raise SystemExit(
            f"ABORT: account {info.login} on '{info.server}' is a REAL account. "
            "Pass --live to acknowledge real-money trading (and --execute to send).")
    print(f"  {'DEMO' if is_demo else 'REAL'} account: {info.login} on "
          f"{info.server}  equity {info.equity:,.2f} {info.currency}")
    return info, is_demo


def refresh_h4(conn: MT5Connector, symbol: str, save: bool) -> pd.DataFrame:
    """Merge fresh completed terminal H4 bars into the long H4 CSV."""
    hist = pd.read_csv(demo.H4_CSV, parse_dates=["time"],
                       index_col="time").sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    live = None
    for count in (5000, 2000, 500):
        try:
            live = conn.get_rates(symbol, "H4", count=count)
            break
        except RuntimeError:
            continue
    if live is None:
        print("  ! H4 fetch failed — replaying from CSV alone")
        return hist
    live["spread"] = live["spread"] / 10.0
    live = live[["open", "high", "low", "close",
                 "tick_volume", "spread"]].iloc[:-1]  # drop the forming bar
    merged = pd.concat([hist, live])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    print(f"  H4 bars: {len(hist)} -> {len(merged)} "
          f"({len(merged) - len(hist):+d} new, last {merged.index[-1]})")
    if save:
        merged.to_csv(demo.H4_CSV)
    return merged


def size_lots(conn, symbol, direction, price, sl, equity, conf, si, max_lot,
              bot_cfg, force_min):
    mt5 = conn._mt5
    order = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(order, symbol, 1.0, price, sl)
    vol_min = getattr(si, "volume_min", 0.01)
    vol_step = getattr(si, "volume_step", 0.01)
    if loss is None or loss >= 0:
        if force_min:
            print("  ! order_calc_profit unavailable — broker min lot")
            return vol_min
        return 0.0
    loss = abs(loss)
    risk = xt.PARAMS["risk_frac"] * bot_cfg["params"]["conf_risk_scale"][conf]
    lots = round(round((risk * equity) / loss / vol_step) * vol_step, 2)
    if lots < vol_min:
        if force_min:
            print(f"  ! forced to min lot {vol_min}: actual risk "
                  f"{vol_min * loss / equity:.1%} (target {risk:.1%})")
            return vol_min
        return 0.0
    return min(lots, max_lot)


def build_plan(res, held, held_dir, pendings, tick, si, acct, bot_cfg, args,
               conn, symbol, atr_last):
    """Reconcile broker to the engine's end state. Market orders only."""
    pos, pending = res["open_position"], res["pending"]
    actions = []
    want_dir = pos["dir"] if pos else (pending["dir"] if pending else 0)

    # anything pending at the broker is stale: this executor never places limits
    for od in pendings:
        actions.append(("cancel", dict(ticket=od.ticket)))

    if held is not None and want_dir != held_dir:
        actions.append(("close", dict(position=held, ticket=held.ticket)))
        held, held_dir = None, 0

    if pos is not None and held is None:
        price = tick.ask if pos["dir"] > 0 else tick.bid
        sl = pos["sl"] if pos["sl"] is not None else \
            price - pos["dir"] * bot_cfg["params"]["sl_atr"] * pos["atr_at_entry"]
        lots = size_lots(conn, symbol, pos["dir"], price, sl, acct.equity,
                         pos["conf"], si, args.max_lot, bot_cfg,
                         args.force_min_lot)
        if lots > 0:
            actions.append(("open_market", dict(
                dir=pos["dir"], lots=lots, sl=round(sl, 2),
                why="engine holds position (catch-up)")))
    elif pos is not None and held is not None and pos["sl"] is not None and \
            abs(float(held.sl or 0) - pos["sl"]) > SL_TOLERANCE:
        actions.append(("modify_sl", dict(position=held, ticket=held.ticket,
                                          sl=round(pos["sl"], 2))))

    if pending is not None and pos is None and held is None and \
            pending.get("wait", 0) <= 0:
        d = pending["dir"]
        price = tick.ask if d > 0 else tick.bid
        sl = price - d * bot_cfg["params"]["sl_atr"] * atr_last
        conf = xt.confidence_bucket(pending["strength"])
        lots = size_lots(conn, symbol, d, price, sl, acct.equity, conf, si,
                         args.max_lot, bot_cfg, args.force_min_lot)
        if lots > 0:
            actions.append(("open_market", dict(
                dir=d, lots=lots, sl=round(sl, 2),
                why=f"engine signal fill (conf {conf})")))
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot", choices=["ls", "champ"], required=True)
    ap.add_argument("--live", action="store_true",
                    help="acknowledge trading a REAL account (required for real)")
    ap.add_argument("--execute", action="store_true",
                    help="actually send orders (needs --live on a real account)")
    ap.add_argument("--max-lot", type=float, default=0.02)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    bot_cfg = cfg["bots"][args.bot]
    magic, run_id = bot_cfg["magic"], bot_cfg["run_id"]
    demo.RUN_ID = run_id  # tag order comments
    journal = TradeJournal(args.journal)

    conn = MT5Connector()
    conn.connect()
    try:
        acct, is_demo = require_live(conn, args.live)
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live")

        symbol = demo.resolve_symbol(conn)
        h4 = refresh_h4(conn, symbol, save=args.save_data)

        xt.xau_signal = lambda close: SIGNALS[args.bot](close)
        res = xt.run_trades(h4, equity0=float(acct.equity) or 3000.0,
                            exit_mode=cfg["exit_mode"],
                            flip_mode=cfg["flip_mode"],
                            params=bot_cfg["params"])
        atr_last = float(xt.wilder_atr(h4, xt.PARAMS["atr_period"]).iloc[-1])
        pos, pending = res["open_position"], res["pending"]
        fc = float(res["signal"].iloc[-1])
        state = ("POSITION " + ("LONG" if pos["dir"] > 0 else "SHORT") if pos
                 else "ENTRY pending" if pending else "flat")
        print(f"  engine[{args.bot}]: forecast {fc:+.2f}  {state}")
        if pos:
            print(f"          entry ~{pos['entry']:.2f}  SL {pos['sl']:.2f}"
                  f"{' (trailing)' if pos['trail_on'] else ''}  conf {pos['conf']}")

        mine = [p for p in (conn.get_positions(magic=magic) or [])
                if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        pendings = demo.my_pendings(conn, symbol, magic)
        print(f"  broker[{magic}]: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        actions = build_plan(res, held, held_dir, pendings,
                             conn.get_tick(symbol), conn.symbol_info(symbol),
                             acct, bot_cfg, args, conn, symbol, atr_last)
        if not actions:
            print("  PLAN: in sync — nothing to do")
        for act, a in actions:
            printable = {k: v for k, v in a.items() if k != "position"}
            print(f"  PLAN: {act} {printable}")
            if not send:
                continue
            try:
                if act == "close":
                    r = conn.close_position(a["position"])
                elif act == "open_market":
                    r = conn.open_position(
                        symbol, "buy" if a["dir"] > 0 else "sell",
                        a["lots"], sl=a["sl"], magic=magic, comment=run_id)
                elif act == "modify_sl":
                    r = conn.modify_position(a["position"].ticket, sl=a["sl"],
                                             tp=float(a["position"].tp or 0.0))
                elif act == "cancel":
                    r = demo.cancel_pending(conn, a["ticket"])
                journal.record(dict(
                    bot=f"v5_xau_dual_{args.bot}", symbol=symbol, direction=act,
                    entry_time=str(h4.index[-1]),
                    entry_reason=json.dumps(printable, default=str)[:180],
                    volume=a.get("lots", 0.0), sl_pips=a.get("sl"),
                    magic=magic, run_id=run_id, dry_run=0))
                print(f"    EXECUTED: "
                      f"{r.get('retcode', r) if isinstance(r, dict) else r}")
            except Exception as exc:  # noqa: BLE001
                print(f"    ORDER REJECTED: {exc}")
        if not send and actions:
            print("  (dry plan — rerun with --live --execute to send)")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
