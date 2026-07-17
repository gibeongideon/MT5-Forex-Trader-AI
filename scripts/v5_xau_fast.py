"""v5_xau_fast.py — MT5 executor for the FAST intraday XAUUSD trend runner.

    --bot fast   LONG-ONLY concentrated intraday trend, M30 cadence (magic 360543)

Sibling of v5_xau_dual.py but on M30 bars with the fast signal
(src.v5.xau_fast_signals) and a hard SPREAD-GUARD preflight: this strategy is
only profitable on a raw/ECN gold account (spread <= ~$0.24; see
data/v5_runs/fast-trend/). If the live spread exceeds the configured ceiling
the bot ABORTS before planning — it physically cannot bleed on the wide-spread
cent account. Same double safety lock as the dual bot (real account needs BOTH
--live and --execute).

    python scripts/v5_xau_fast.py --bot fast                 # dry plan
    python scripts/v5_xau_fast.py --bot fast --live --execute # live (raw acct)
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
from src.v5.xau_fast_signals import SIGNALS  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_xau_fast.json"
M30_CSV = ROOT / "data" / "XAUUSD_M30_long.csv"
SL_TOLERANCE = 0.5  # USD


def require_live(conn: MT5Connector, allow_live: bool):
    info = conn.account_info()
    if info is None:
        raise SystemExit("ABORT: no account logged in on the MT5 terminal")
    is_demo = getattr(info, "trade_mode", None) == 0 or \
        "demo" in str(info.server).lower()
    if not is_demo and not allow_live:
        raise SystemExit(
            f"ABORT: account {info.login} on '{info.server}' is a REAL account. "
            "Pass --live to acknowledge real-money trading (and --execute).")
    print(f"  {'DEMO' if is_demo else 'REAL'} account: {info.login} on "
          f"{info.server}  equity {info.equity:,.2f} {info.currency}")
    return info, is_demo


def spread_guard(conn, symbol, si, max_spread_usd: float) -> float:
    """Measure the live $/oz spread; abort if it exceeds the ceiling.
    This encodes the core research finding: the fast bot is only viable on a
    raw/ECN gold account. Returns the measured spread in USD."""
    point = getattr(si, "point", 0.01)
    tick = conn.get_tick(symbol)
    live = None
    if tick and getattr(tick, "ask", 0) and getattr(tick, "bid", 0):
        live = float(tick.ask - tick.bid)
    pts = float(getattr(si, "spread", 0) or 0) * point
    spread_usd = max(live or 0.0, pts)
    print(f"  spread guard: {symbol} live spread ${spread_usd:.3f}/oz "
          f"(ceiling ${max_spread_usd:.2f})")
    if spread_usd > max_spread_usd:
        raise SystemExit(
            f"ABORT: spread ${spread_usd:.3f}/oz > ceiling ${max_spread_usd:.2f}. "
            f"The fast trend bot only works on a raw/ECN gold account "
            f"(<= ${max_spread_usd:.2f}). This account/symbol is too wide — "
            f"do not trade it here.")
    return spread_usd


def refresh_m30(conn, symbol, spread_model_usd: float, save: bool) -> pd.DataFrame:
    """Merge fresh completed M30 bars into the long CSV, then set a FIXED
    replay spread so the engine's historical trade state matches the
    backtest (which assumed the target account's spread)."""
    hist = pd.read_csv(M30_CSV, parse_dates=["time"], index_col="time").sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    live = None
    for count in (5000, 2000, 500):
        try:
            live = conn.get_rates(symbol, "M30", count=count)
            break
        except RuntimeError:
            continue
    if live is not None:
        live = live[["open", "high", "low", "close", "tick_volume",
                     "spread"]].iloc[:-1]  # drop the forming bar
        merged = pd.concat([hist, live[["open", "high", "low", "close",
                                        "tick_volume", "spread"]]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        print(f"  M30 bars: {len(hist)} -> {len(merged)} "
              f"({len(merged) - len(hist):+d} new, last {merged.index[-1]})")
        if save:
            merged.to_csv(M30_CSV)
    else:
        print("  ! M30 fetch failed — replaying from CSV alone")
        merged = hist
    # engine cost parity: fixed spread in "pips" (engine multiplies by 0.1)
    merged = merged.copy()
    merged["spread"] = spread_model_usd / 0.1
    return merged


def size_lots(conn, symbol, direction, price, sl, equity, conf, si, max_lot,
              bot_cfg, force_min):
    mt5 = conn._mt5
    order = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(order, symbol, 1.0, price, sl)
    vol_min = getattr(si, "volume_min", 0.01)
    vol_step = getattr(si, "volume_step", 0.01)
    if loss is None or loss >= 0:
        return vol_min if force_min else 0.0
    loss = abs(loss)
    risk = bot_cfg["params"]["risk_frac"] * \
        bot_cfg["params"]["conf_risk_scale"][conf]
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
    pos, pending = res["open_position"], res["pending"]
    actions = []
    want_dir = pos["dir"] if pos else (pending["dir"] if pending else 0)
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
                         pos["conf"], si, args.max_lot, bot_cfg, args.force_min_lot)
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
    ap.add_argument("--bot", choices=["fast"], default="fast")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-lot", type=float, default=0.10)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    bot_cfg = cfg["bots"][args.bot]
    magic, run_id = bot_cfg["magic"], bot_cfg["run_id"]
    demo.RUN_ID = run_id
    journal = TradeJournal(args.journal)

    conn = MT5Connector()
    conn.connect()
    try:
        acct, is_demo = require_live(conn, args.live)
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live")

        symbol = bot_cfg.get("symbol_override") or demo.resolve_symbol(conn)
        si = conn.symbol_info(symbol)
        # HARD GATE: refuse to trade a too-wide spread account.
        spread_guard(conn, symbol, si, bot_cfg.get("max_spread_usd", 0.24))

        m30 = refresh_m30(conn, symbol, bot_cfg.get("spread_model_usd", 0.12),
                          save=args.save_data)

        xt.xau_signal = lambda close: SIGNALS[args.bot](close)
        res = xt.run_trades(m30, equity0=float(acct.equity) or 3000.0,
                            exit_mode=cfg["exit_mode"], flip_mode=cfg["flip_mode"],
                            params=bot_cfg["params"])
        atr_last = float(xt.wilder_atr(m30, bot_cfg["params"].get(
            "atr_period", 14)).iloc[-1])
        pos, pending = res["open_position"], res["pending"]
        fc = float(res["signal"].iloc[-1])
        state = ("POSITION " + ("LONG" if pos["dir"] > 0 else "SHORT") if pos
                 else "ENTRY pending" if pending else "flat")
        print(f"  engine[fast]: forecast {fc:+.2f}  {state}")
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

        actions = build_plan(res, held, held_dir, pendings, conn.get_tick(symbol),
                             si, acct, bot_cfg, args, conn, symbol, atr_last)
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
                    bot="v5_xau_fast", symbol=symbol, direction=act,
                    entry_time=str(m30.index[-1]),
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
