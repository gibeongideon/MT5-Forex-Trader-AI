"""v5_xau_challenge.py — FundingPips challenge executor (XAUUSD champion).

Guard-first sibling of v5_xau_dual.py (see CHALLENGEBOT.MD). Every pass
(15-min timer cadence):

  1. evaluate guards on LIVE equity (src.v5.challenge_guards.decide):
       halt / day_lock / locked / complete  -> ensure FLAT, exit
       realize_target                        -> close position (bank it), exit
       trade                                 -> normal H4 reconcile
  2. state persisted to --state JSON (day anchor at 00:00 UTC+3, phase,
     locks); initialized from the account on the first pass.

Safety locks identical to the dual bot: real accounts need --live, orders
need --execute. Phase promotion (new credentials / Phase 2) is manual:
    python scripts/v5_xau_challenge.py --advance-phase
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "v5_xau_dual", ROOT / "scripts" / "v5_xau_dual.py")
dual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dual)
demo = dual.demo

import src.v5.challenge_guards as cg  # noqa: E402
import src.v5.xau_trend as xt  # noqa: E402
from src.v5.news_filter import NewsFilter, apply_to_plan  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402
from src.core.trade_journal import TradeJournal  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_xau_challenge.json"
STATE_DEFAULT = ROOT / "data" / "v5_runs" / "challenge_state.json"


def log_paper(path: str, row: dict) -> None:
    """Append one per-pass row to the dry-run CSV (creates with header)."""
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_state(path: Path, acct) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    state = cg.init_state(float(acct.balance), float(acct.equity))
    print(f"  state INIT: initial_balance {state['initial_balance']:,.2f}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1))


def flatten(conn, symbol, magic, send, journal, run_id, why) -> None:
    """Close our position + cancel our pendings (idempotent)."""
    mine = [p for p in (conn.get_positions(magic=magic) or [])
            if p.symbol == symbol]
    for od in demo.my_pendings(conn, symbol, magic):
        print(f"  GUARD[{why}]: cancel pending {od.ticket}")
        if send:
            demo.cancel_pending(conn, od.ticket)
    for p in mine:
        print(f"  GUARD[{why}]: close {p.volume} lots ticket {p.ticket}")
        if send:
            r = conn.close_position(p)
            journal.record(dict(bot="v5_xau_challenge", symbol=symbol,
                                direction="guard_close", entry_reason=why,
                                volume=p.volume, magic=magic,
                                run_id=run_id, dry_run=0))
            print(f"    EXECUTED: "
                  f"{r.get('retcode', r) if isinstance(r, dict) else r}")
    if not mine:
        print(f"  GUARD[{why}]: already flat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-lot", type=float, default=5.0)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--state", default=str(STATE_DEFAULT))
    ap.add_argument("--paper-csv", default=None,
                    help="append one row per pass to this CSV (dry-run log)")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    ap.add_argument("--advance-phase", action="store_true",
                    help="promote state to Phase 2 (run once, no trading)")
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    magic, run_id = cfg["magic"], cfg["run_id"]
    g = cfg["guards"]
    demo.RUN_ID = run_id
    journal = TradeJournal(args.journal)
    state_path = Path(args.state)

    conn = MT5Connector()
    conn.connect()
    try:
        acct, is_demo = dual.require_live(conn, args.live)
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live")

        state = load_state(state_path, acct)

        if args.advance_phase:
            state = cg.advance_phase(state, float(acct.balance))
            save_state(state_path, state)
            print(f"  state: advanced to PHASE 2, start "
                  f"{state['phase_start']:,.2f}, target +5%")
            return

        state, action = cg.decide(state, float(acct.balance),
                                  float(acct.equity))
        save_state(state_path, state)
        anchor_dd = (float(acct.equity) / state["day_anchor"] - 1) * 100
        total_dd = (float(acct.equity) / state["initial_balance"] - 1) * 100
        prog = (float(acct.equity) / state["phase_start"] - 1) * 100
        print(f"  guards: action={action}  phase {state['phase']} "
              f"progress {prog:+.2f}% (target +{state['phase_target_frac']*100:.0f}%)  "
              f"day {anchor_dd:+.2f}% (lock -3.5%)  total {total_dd:+.2f}% (halt -8%)")

        symbol = demo.resolve_symbol(conn)

        from datetime import datetime, timezone
        row = dict(time_utc=datetime.now(timezone.utc).strftime("%F %T"),
                   account=acct.login, balance=round(float(acct.balance), 2),
                   equity=round(float(acct.equity), 2), action=action,
                   phase=state["phase"], progress_pct=round(prog, 3),
                   day_dd_pct=round(anchor_dd, 3),
                   total_dd_pct=round(total_dd, 3), forecast=None,
                   engine_state=None, engine_entry=None, engine_sl=None,
                   engine_conf=None, news_blocked=0, news_event=None,
                   plan="", sent=int(send))

        if action in ("halt", "day_lock", "locked", "complete",
                      "realize_target"):
            flatten(conn, symbol, magic, send, journal, run_id, action)
            if action == "halt":
                print("  *** PERMANENT HALT — challenge risk line hit ***")
            if action == "complete":
                print("  *** PHASE TARGET REALIZED — await promotion, then "
                      "run --advance-phase ***")
            if args.paper_csv:
                log_paper(args.paper_csv, row)
            return

        # ---- normal reconcile pass (same flow as the dual bot) ----
        h4 = dual.refresh_h4(conn, symbol, save=args.save_data)
        xt.xau_signal = lambda close: champion_signal(close)
        res = xt.run_trades(h4, equity0=float(acct.equity) or 100_000.0,
                            exit_mode=cfg["exit_mode"],
                            flip_mode=cfg["flip_mode"], params=cfg["params"])
        atr_last = float(xt.wilder_atr(h4, xt.PARAMS["atr_period"]).iloc[-1])
        pos, pending = res["open_position"], res["pending"]
        fc = float(res["signal"].iloc[-1])
        state_s = ("POSITION LONG" if pos and pos["dir"] > 0 else
                   "POSITION SHORT" if pos else
                   "ENTRY pending" if pending else "flat")
        print(f"  engine[challenge]: forecast {fc:+.2f}  {state_s}")

        mine = [p for p in (conn.get_positions(magic=magic) or [])
                if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        pendings = demo.my_pendings(conn, symbol, magic)
        print(f"  broker[{magic}]: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        bot_cfg = {"params": cfg["params"]}
        actions = dual.build_plan(res, held, held_dir, pendings,
                                  conn.get_tick(symbol),
                                  conn.symbol_info(symbol), acct, bot_cfg,
                                  args, conn, symbol, atr_last)

        nf_cfg = cfg.get("news_filter", {})
        verdict = NewsFilter(nf_cfg, root=ROOT).check()
        if verdict["blocked"]:
            print(f"  NEWS WINDOW: {verdict['event']} @ "
                  f"{verdict['event_time']} ({verdict['window']}) — "
                  f"entries paused"
                  + (" [STALE CALENDAR]" if verdict.get("stale") else ""))
        actions, n_blocked, profit_close = apply_to_plan(
            actions, held, verdict, nf_cfg.get("close_in_profit", True))
        if n_blocked:
            print(f"  NEWS: {n_blocked} new-entry action(s) blocked")
        if profit_close:
            print("  NEWS: closing in-profit position ahead of the event")
        row.update(news_blocked=int(bool(verdict["blocked"])),
                   news_event=verdict.get("event"))

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
                    bot="v5_xau_challenge", symbol=symbol, direction=act,
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

        if args.paper_csv:
            row.update(
                forecast=round(fc, 3),
                engine_state=state_s,
                engine_entry=round(pos["entry"], 2) if pos else None,
                engine_sl=round(pos["sl"], 2) if pos and pos["sl"] else None,
                engine_conf=pos["conf"] if pos else None,
                plan="; ".join(
                    f"{act}:{json.dumps({k: v for k, v in a.items() if k != 'position'}, default=str)}"
                    for act, a in actions) or "in_sync")
            log_paper(args.paper_csv, row)
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
