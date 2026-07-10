"""v5_xau_live.py — LIVE MT5 executor for the promoted XAUUSD engine.

Thin fork of v5_xau_demo.py. It reuses that script's *validated* reconciliation
and sizing logic verbatim (imported, not copied) and changes only the account
gate: this script is allowed to trade a REAL account, but only behind a double
safety lock —

    * --live      : explicit acknowledgement you intend real-money trading;
                    without it, a real account HARD-ABORTS (demo still runs).
    * --execute   : actually send orders (same as the demo script).

So a live real-money order requires BOTH flags. `--live` alone still dry-plans.

Sizing is unchanged: risk_frac x conf_scale of ACTUAL account equity over the
2xATR stop, computed via broker order_calc_profit on the resolved symbol, so it
is automatically correct for the cent account (XAUUSDc, contract_size=1, USC).
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

# Reuse the validated demo executor's helpers verbatim (single source of truth).
_spec = importlib.util.spec_from_file_location(
    "v5_xau_demo", ROOT / "scripts" / "v5_xau_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

from src.core.mt5_connector import MT5Connector
from src.core.trade_journal import TradeJournal
from src.v5.xau_m15_exec import run_trades_m15

RUN_ID = "v5-xau-live"
demo.RUN_ID = RUN_ID  # so place_limit / order comments are tagged live


def fresh_data(conn: MT5Connector, symbol: str, save: bool) -> pd.DataFrame:
    """Like demo.fresh_data, but robust to brokers with shallow history.

    The cent symbol (XAUUSDc) only carries a few months of M15 bars and this
    broker returns nothing when the requested count exceeds what it holds, so
    we step the request down until it succeeds. The deep history still comes
    from the spliced CSV we merge into; the terminal fetch only appends recent
    bars, so a smaller window loses nothing.
    """
    hist = pd.read_csv(demo.M15_CSV, parse_dates=["time"],
                       index_col="time").sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    live = None
    for count in (60_000, 30_000, 20_000, 10_000, 5_000):
        try:
            live = conn.get_rates(symbol, "M15", count=count)
            break
        except RuntimeError:
            continue
    if live is None:
        raise SystemExit(f"ABORT: no M15 history available for {symbol}")
    live["spread"] = live["spread"] / 10.0
    live = live[["open", "high", "low", "close", "tick_volume", "spread"]]
    now = pd.Timestamp.utcnow().tz_localize(None)
    live = live[live.index + pd.Timedelta(minutes=15) <= now + pd.Timedelta(hours=3)]
    merged = pd.concat([hist, live])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    print(f"  M15 bars: {len(hist)} -> {len(merged)} "
          f"({len(merged) - len(hist):+d} new, last {merged.index[-1]})")
    if save:
        merged.to_csv(demo.M15_CSV)
        try:
            h4l = conn.get_rates(symbol, "H4", count=5000)
            h4l["spread"] = h4l["spread"] / 10.0
            h4l = h4l[["open", "high", "low", "close",
                       "tick_volume", "spread"]].iloc[:-1]
            h4h = pd.read_csv(demo.H4_CSV, parse_dates=["time"],
                              index_col="time").sort_index()
            h4m = pd.concat([h4h, h4l])
            h4m[~h4m.index.duplicated(keep="last")].sort_index().to_csv(demo.H4_CSV)
        except Exception as e:  # noqa: BLE001 — H4 refresh is best-effort
            print(f"  ! H4 CSV refresh skipped: {e}")
    return merged


def require_live(conn: MT5Connector, allow_live: bool):
    """Allow demo accounts freely; allow real accounts ONLY with --live."""
    info = conn.account_info()
    if info is None:
        raise SystemExit("ABORT: no account logged in on the MT5 terminal")
    is_demo = getattr(info, "trade_mode", None) == 0 or \
        "demo" in str(info.server).lower()
    kind = "DEMO" if is_demo else "REAL"
    if not is_demo and not allow_live:
        raise SystemExit(
            f"ABORT: account {info.login} on '{info.server}' is a REAL account. "
            "Pass --live to acknowledge real-money trading (and --execute to send).")
    print(f"  {kind} account: {info.login} on {info.server}  "
          f"equity {info.equity:,.2f} {info.currency}  leverage 1:{info.leverage}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="acknowledge trading a REAL account (required for real)")
    ap.add_argument("--execute", action="store_true",
                    help="actually send orders (needs --live on a real account)")
    ap.add_argument("--max-lot", type=float, default=0.02,
                    help="hard ceiling on lot size (tiny by default for cent acct)")
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(demo.CONFIG_FILE.read_text())
    exe = cfg.get("execution", {})
    magic = cfg["magic"]
    journal = TradeJournal(args.journal)

    conn = MT5Connector()
    conn.connect()
    try:
        acct = require_live(conn, args.live)
        # Second lock: never SEND on a real account without an explicit --live.
        is_demo = getattr(acct, "trade_mode", None) == 0 or \
            "demo" in str(acct.server).lower()
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live "
                  "(dry plan only)")

        symbol = demo.resolve_symbol(conn)
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
        pendings = demo.my_pendings(conn, symbol, magic)
        print(f"  broker: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        actions = demo.build_plan(res, held, held_dir, pendings,
                                  conn.get_tick(symbol),
                                  conn.symbol_info(symbol),
                                  acct, cfg, args, conn, symbol)
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
                        a["lots"], sl=a["sl"], magic=magic, comment=RUN_ID)
                elif act == "modify_sl":
                    r = conn.modify_position(a["position"].ticket, sl=a["sl"],
                                             tp=float(a["position"].tp or 0.0))
                elif act == "place_limit":
                    r = demo.place_limit(conn, symbol, a["dir"], a["lots"],
                                         a["price"], a["sl"], magic)
                elif act == "cancel":
                    r = demo.cancel_pending(conn, a["ticket"])
                journal.record(dict(
                    bot="v5_xau_live", symbol=symbol, direction=act,
                    entry_time=str(m15.index[-1]),
                    entry_reason=json.dumps(printable, default=str)[:180],
                    volume=a.get("lots", 0.0), sl_pips=a.get("sl"),
                    magic=magic, run_id=RUN_ID, dry_run=0))
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
