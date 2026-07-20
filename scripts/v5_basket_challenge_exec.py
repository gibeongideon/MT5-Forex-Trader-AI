"""v5_basket_challenge_exec.py — FundingPips DIVERSIFIED BASKET challenge executor.

Guard-first, multi-symbol sibling of v5_xau_challenge.py. Trades the champion
long-only recipe across the tradeable drift classes (indices + crypto + XAU +
silver), 2-Step Standard @ 7% vol (configs/v5_basket_challenge.json). Every pass:

  1. GUARD on live account equity (src.v5.challenge_guards.decide):
       halt / day_lock / locked / complete  -> ensure FLAT (all symbols), exit
       realize_target                        -> close everything (bank it), exit
       trade                                 -> reconcile each symbol to target
  2. TARGET per symbol = scripts.v5_basket_challenge.target_leverage(model)
     -> account-leverage; converted to lots at execute time from live
     symbol_info/tick; buffered no-trade band cuts churn.
  3. State persisted (day anchor 00:00 UTC+3, phase, locks). Phase promotion
     is manual: --advance-phase.

Safety: real accounts need --live; orders need --execute; a symbol only trades
if its broker name is mapped in config (fp_symbol) AND resolves on the terminal.
Until the FundingPips account exists, every fp_symbol is null -> DRY-RUN only:
it prints the guard status + per-symbol target plan and logs a CSV row, never
sending an order. This is the pre-live verification mode.

    # dry-run one pass (read-only, safe on any account):
    conda run -n envmt5 python scripts/v5_basket_challenge_exec.py \
        --state data/v5_runs/basket_challenge_dry_state.json \
        --paper-csv data/v5_runs/basket_challenge_dry_log.csv
    # live (after account purchase + fp_symbol mapping):
    ... --live --execute
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.v5.challenge_guards as cg  # noqa: E402
from scripts.v5_basket_challenge import target_leverage, MODELS  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_basket_challenge.json"
STATE_DEFAULT = ROOT / "data" / "v5_runs" / "basket_challenge_state.json"


def apply_model_to_guards(cfg) -> None:
    """Override challenge_guards module constants from config (model-correct)."""
    g = cfg["guards"]
    cg.DAILY_GUARD_FRAC = float(g["daily_guard_frac"])
    cg.OVERALL_HALT_FRAC = float(g["overall_halt_frac"])
    cg.PHASE_TARGETS = {int(k): float(v) for k, v in g["phase_targets"].items()}
    cg.set_reset_tz(cfg.get("reset_tz"))   # FundingPips UTC+3 / FTMO CE(S)T


def log_row(path, row) -> None:
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
    st = cg.init_state(float(acct.balance), float(acct.equity))
    print(f"  state INIT: initial_balance {st['initial_balance']:,.2f}  "
          f"phase 1 target +{cg.PHASE_TARGETS[1]*100:.0f}%")
    return st


def save_state(path: Path, st) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=1))


def held_lots(conn, symbol, magic) -> float:
    """Signed net lots we hold on `symbol` (our magic only)."""
    net = 0.0
    for p in (conn.get_positions(magic=magic) or []):
        if p.symbol == symbol:
            net += p.volume if p.type == 0 else -p.volume
    return net


def close_volume(conn, symbol, magic, lots, run_id) -> list:
    """Reduce our exposure on `symbol` by `lots` by CLOSING our own tickets
    (fully, then partially for the remainder), oldest first.

    Required on HEDGING accounts (FTMO is margin_mode=2): sending an opposing
    market order there does NOT net the position down — it opens a second,
    opposite position, doubling margin and cost. Closing by ticket is correct
    on both hedging and netting accounts.
    """
    mt5 = conn._mt5
    remaining = round(float(lots), 2)
    done = []
    ps = [p for p in (conn.get_positions(magic=magic) or []) if p.symbol == symbol]
    ps.sort(key=lambda p: getattr(p, "time", 0))          # FIFO
    for p in ps:
        if remaining <= 1e-9:
            break
        vol = round(min(float(p.volume), remaining), 2)
        if vol <= 0:
            continue
        tick = conn.get_tick(symbol)
        is_long = (p.type == 0)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                   type=mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
                   position=p.ticket,
                   price=(tick.bid if is_long else tick.ask),
                   deviation=50, magic=magic, comment=run_id,
                   type_time=mt5.ORDER_TIME_GTC,
                   type_filling=conn._fill_type(symbol))
        r = mt5.order_send(req)
        rc = getattr(r, "retcode", None)
        done.append((p.ticket, vol, rc))
        if rc == mt5.TRADE_RETCODE_DONE:
            remaining = round(remaining - vol, 2)
    return done


def target_lots(conn, symbol, lev, equity) -> float | None:
    """lots = lev * equity / (contract_size * price). None if symbol unavailable."""
    info = conn.symbol_info(symbol)
    tick = conn.get_tick(symbol)
    if info is None or tick is None:
        return None
    price = float(getattr(tick, "ask", 0) or getattr(tick, "last", 0) or 0)
    csize = float(getattr(info, "trade_contract_size", 0) or 0)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    if price <= 0 or csize <= 0:
        return None
    raw = lev * equity / (csize * price)
    return round(raw / step) * step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--state", default=str(STATE_DEFAULT))
    ap.add_argument("--paper-csv", default=None)
    ap.add_argument("--advance-phase", action="store_true")
    ap.add_argument("--config", default=str(CONFIG_FILE),
                    help="challenge config JSON (default: FundingPips basket)")
    ap.add_argument("--guard-only", action="store_true",
                    help="fast real-time protector: evaluate guards + flatten on "
                         "breach, but DO NOT reconcile/open (for a 1-2min timer)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    apply_model_to_guards(cfg)
    model, magic, run_id = cfg["model"], cfg["magic"], cfg["run_id"]
    buf = float(cfg.get("reconcile_buffer", 0.15))
    symmap = {k: v for k, v in cfg["symbols"].items() if not k.startswith("_")}
    state_path = Path(args.state)

    conn = MT5Connector()
    conn.connect()
    try:
        acct = conn.account_info()
        equity, balance = float(acct.equity), float(acct.balance)
        # real-account order lock (mirror of dual/challenge bots)
        is_demo = "demo" in str(getattr(acct, "server", "")).lower()
        send = args.execute and (is_demo or args.live)

        state = load_state(state_path, acct)
        if args.advance_phase:
            state = cg.advance_phase(state, balance)
            save_state(state_path, state)
            print(f"  advanced to PHASE 2, start {state['phase_start']:,.2f}, "
                  f"target +{cg.PHASE_TARGETS[2]*100:.0f}%")
            return

        state, action = cg.decide(state, balance, equity)
        save_state(state_path, state)
        day_dd = (equity / state["day_anchor"] - 1) * 100
        tot_dd = (equity / state["initial_balance"] - 1) * 100
        prog = (equity / state["phase_start"] - 1) * 100
        print(f"[basket challenge] model={model.upper()} acct={acct.login} "
              f"bal={balance:,.2f} eq={equity:,.2f} {acct.currency}")
        print(f"  guards: action={action}  phase {state['phase']} "
              f"progress {prog:+.2f}% (target +{state['phase_target_frac']*100:.0f}%)  "
              f"day {day_dd:+.2f}% (lock -{cg.DAILY_GUARD_FRAC*100:.1f}%)  "
              f"total {tot_dd:+.2f}% (halt -{cg.OVERALL_HALT_FRAC*100:.0f}%)")

        row = dict(time_utc=datetime.now(timezone.utc).strftime("%F %T"),
                   account=acct.login, balance=round(balance, 2),
                   equity=round(equity, 2), action=action, phase=state["phase"],
                   progress_pct=round(prog, 3), day_dd_pct=round(day_dd, 3),
                   total_dd_pct=round(tot_dd, 3), n_symbols=len(symmap),
                   n_mapped=0, plan="", sent=int(send))

        # ---- guard actions: ensure flat, no reconcile ----
        if action in ("halt", "day_lock", "locked", "complete", "realize_target"):
            flats = []
            for esym, meta in symmap.items():
                bsym = meta.get("fp_symbol")
                if bsym and held_lots(conn, bsym, magic) != 0.0:
                    flats.append(bsym)
                    print(f"  GUARD[{action}]: flatten {bsym}")
                    if send:
                        for p in (conn.get_positions(magic=magic) or []):
                            if p.symbol == bsym:
                                conn.close_position(p)
            if action == "halt":
                print("  *** PERMANENT HALT — risk line hit ***")
            elif action == "complete":
                print("  *** PHASE TARGET REALIZED — await promotion, run --advance-phase ***")
            elif not flats:
                print(f"  GUARD[{action}]: already flat")
            row["plan"] = f"{action}:flatten({len(flats)})"
            if args.paper_csv:
                log_row(args.paper_csv, row)
            return

        # ---- guard-only mode: no reconcile, just confirm we're safe ----
        if args.guard_only:
            print("  guard-only: safe (action=trade) — no reconcile")
            row["plan"] = "guard_only:safe"
            if args.paper_csv:
                log_row(args.paper_csv, row)
            return

        # ---- normal reconcile: move each symbol toward its target ----
        # cfg["classes"] lets one engine serve several books (100K vs 10K).
        targets = target_leverage(model, cfg.get("classes"))
        plans, n_mapped = [], 0
        print(f"  {'symbol':9s} {'tgt lev':>8s} {'held lot':>9s} {'tgt lot':>9s}  action")
        for esym, lev in sorted(targets.items()):
            meta = symmap.get(esym, {})
            bsym = meta.get("fp_symbol")
            if not bsym:                            # unmapped -> can't trade yet
                print(f"  {esym:9s} {lev:8.3f} {'—':>9s} {'—':>9s}  UNMAPPED (dry)")
                continue
            n_mapped += 1
            tl = target_lots(conn, bsym, lev, equity)
            hl = held_lots(conn, bsym, magic)
            if tl is None:
                print(f"  {esym:9s} {lev:8.3f} {hl:9.2f} {'n/a':>9s}  SYMBOL UNAVAILABLE")
                continue
            band = buf * max(abs(tl), 1e-9)
            act = "hold"
            if abs(tl - hl) > band:
                act = "adjust" if hl != 0 else "open"
                plans.append((bsym, hl, tl))
            print(f"  {esym:9s} {lev:8.3f} {hl:9.2f} {tl:9.2f}  {act}")
            if send and act != "hold":
                delta = round(tl - hl, 2)
                try:
                    if delta > 0:                      # increase -> open/add
                        r = conn.open_position(bsym, "buy", abs(delta),
                                               magic=magic, comment=run_id)
                        print(f"    EXECUTED buy {abs(delta):.2f}: "
                              f"{r.get('retcode', r) if isinstance(r, dict) else r}")
                    else:                              # decrease -> CLOSE tickets
                        res = close_volume(conn, bsym, magic, abs(delta), run_id)
                        for tk, vol, rc in res:
                            print(f"    CLOSED {vol:.2f} of #{tk}: retcode={rc}")
                        if not res:
                            print("    ! nothing to close (no tickets found)")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ORDER REJECTED: {exc}")

        if n_mapped == 0:
            print("  PLAN: DRY-RUN — no fp_symbol mapped yet (targets shown above; "
                  "fill configs/v5_basket_challenge.json symbols after account purchase)")
        elif not plans:
            print("  PLAN: in sync — nothing to do")
        row.update(n_mapped=n_mapped,
                   plan="; ".join(f"{s}:{h:.2f}->{t:.2f}" for s, h, t in plans) or "in_sync")
        if not send and plans:
            print("  (dry plan — rerun with --live --execute to send)")
        if args.paper_csv:
            log_row(args.paper_csv, row)
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
