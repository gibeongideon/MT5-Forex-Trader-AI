"""v5_basket_runner.py — daily target-position runner for the promoted V5 basket.

Computes TODAY's target positions with the exact validated V5 pipeline
(`src/v5/levers.py::lever_positions` — the same code path as the promoted
backtest run in `configs/v5_basket_champion.json`), reports per-symbol actions
vs the last run, persists state under the run directory, appends an audit CSV,
and journals every non-hold action as a DRY-RUN order intent in TradeJournal.

ADVISORY / PAPER ONLY — no order-send code path exists in this script
(standing rule: never auto-run live). A separate executor or the operator acts
on the printout; dry-run journal rows feed the V5 reconciliation gate.

Usage:
    python scripts/v5_basket_runner.py                      # today's targets
    python scripts/v5_basket_runner.py --equity 10000       # + lot sizing
    python scripts/v5_basket_runner.py --overlay            # + AI agent risk overlay (Phase C)
    python scripts/v5_basket_runner.py --validate           # re-check Sharpe vs promoted run
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

from src.core.trade_journal import TradeJournal
from src.cta.panel import asset_classes, build_panels
from src.cta.sizing import (DEFAULT_CONTRACT, DEFAULT_VOL, gross_exposure,
                            min_viable_equity, target_lots)
from src.v5.h4_cta import mtm_pnl_price_units
from src.v5.levers import lever_positions

CONFIG_FILE = ROOT / "configs" / "v5_basket_champion.json"
STALE_DAYS = 4

# alias → HFM symbol, verified live on HFMarketsKE-Demo2 (symbol_select + symbol_info).
# HFM suffixes rates/US indices with `.F` and prefixes crypto with `#`. There is NO
# 30-year bond on this server (`US30.F` is the Dow index, not a bond).
MT5_MAP = {
    "GOLD":   "XAUUSD",
    "SILVER": "XAGUSD",
    "UST10Y": "US10YR.F",
    "UST30Y": None,       # not offered by HFM — excluded from any tradable basket
    "SPX":    "US500.F",
    "DAX":    "GER40",
    "WTI":    "USOIL",
    "BRENT":  "UKOIL",
    "EURUSD": "EURUSD",
    "USDJPY": "USDJPY",
    "BTC":    "#BTCUSD",
}

# USD notional of 1.0 lot depends on the quote convention. For a USD-BASE pair
# (USDJPY) one lot is `contract_size` USD outright; multiplying by the price would
# yield JPY and oversize the leg ~162x. Everything else (CFD/metal/USD-quote FX)
# is contract_size * price.
FX_BASE_USD = {"USDJPY"}


def _usd_per_lot(alias: str, contract_size: float, price: float) -> float:
    return contract_size if alias in FX_BASE_USD else contract_size * price


def load_champion_config(path: Path = CONFIG_FILE) -> dict:
    return json.loads(path.read_text())


def _action(prev: float, tgt: float, eps: float = 1e-6) -> str:
    if abs(tgt) < eps and abs(prev) < eps:
        return "flat"
    if abs(prev) < eps:
        return "OPEN"
    if abs(tgt) < eps:
        return "CLOSE"
    if np.sign(tgt) != np.sign(prev):
        return "FLIP"
    if abs(tgt) > abs(prev) + eps:
        return "ADD"
    if abs(tgt) < abs(prev) - eps:
        return "TRIM"
    return "hold"


def _build_specs(kept, close, live):
    """Per-alias broker specs. live=query MT5 (exact), else offline defaults.

    `contract_size` is stored as an EFFECTIVE size such that contract_size * price
    equals the true USD notional of one lot, so `sizing.target_lots` (which assumes
    that product) stays correct for USD-base FX. Same convention the feasibility
    study used.
    """
    specs, conn = {}, None
    if live:
        from src.core.connector import get_connector
        conn = get_connector("mt5")
        conn.connect()
    for a in kept:
        sym = MT5_MAP.get(a, a)
        if sym is None:
            raise RuntimeError(f"{a} has no tradable HFM symbol — remove it from the basket")
        if live:
            try:
                if not conn.symbol_select(sym, True):
                    raise RuntimeError(f"symbol_select({sym}) rejected — not tradable")
                si = conn.symbol_info(sym)
                tk = conn.get_tick(sym)
                price = float((tk.ask + tk.bid) / 2) or float(close[a].dropna().iloc[-1])
                raw_contract = float(getattr(si, "trade_contract_size",
                                             DEFAULT_CONTRACT.get(sym, 1.0)))
                per_lot = _usd_per_lot(a, raw_contract, price)
                specs[a] = dict(
                    symbol=sym,
                    contract_size=per_lot / price,
                    price=price,
                    vol_min=float(getattr(si, "volume_min", 0.01)),
                    vol_step=float(getattr(si, "volume_step", 0.01)),
                    vol_max=float(getattr(si, "volume_max", 1e6)))
                continue
            except Exception as e:  # noqa: BLE001 — fall back per-symbol
                print(f"  ! live spec for {sym} failed ({e}) — offline fallback")
        price = float(close[a].dropna().iloc[-1])
        per_lot = _usd_per_lot(a, DEFAULT_CONTRACT.get(sym, 1.0), price)
        specs[a] = dict(symbol=sym, contract_size=per_lot / price,
                        price=price, **DEFAULT_VOL)
    if conn:
        conn.disconnect()
    return specs


def _print_sizing(units, kept, close, equity, live):
    specs = _build_specs(kept, close, live)
    res = target_lots(units, equity, specs)
    src = "LIVE broker specs" if live else "OFFLINE defaults — VERIFY in terminal"
    print(f"\n  -- LOTS for equity ${equity:,.0f} USD  [{src}] --")
    print("     (equity is USD; this account settles in KES — do not pass the KES balance)")
    for a in kept:
        r = res[a]
        flag = "ROUND->0" if r["rounded_zero"] else ("CAPPED" if r["capped"] else "")
        print(f"  {a:8} {r['symbol']:8} px {specs[a]['price']:>10.2f} "
              f"ideal {r['ideal_lots']:>+8.3f} LOTS {r['lots']:>+7.2f} "
              f"notional {r['actual_notional']:>+11,.0f}  {flag}")
    g = gross_exposure(res)
    mve = min_viable_equity(units, specs)
    print(f"  gross ${g['gross_notional']:,.0f} ({g['gross_notional']/equity:.2f}x)"
          f"  net ${g['net_notional']:+,.0f}")
    zeros = [a for a in kept if res[a]["rounded_zero"]]
    if zeros:
        print(f"  ! legs rounding to ZERO at this equity: {zeros}")
    print(f"  min viable equity (every leg >= 1 min-lot): ${mve:,.0f}"
          + ("  (current OK)" if equity >= mve else "  (UNDER — vol target distorted)"))
    return res


def journal_intents(journal: TradeJournal, date: str, actions: dict,
                    targets: dict, run_id: str, magic: int) -> int:
    """Write one dry-run order-intent row per non-hold action. Returns count."""
    n = 0
    for alias, act in actions.items():
        if act in ("hold", "flat"):
            continue
        tgt = targets[alias]
        journal.record(dict(
            bot="v5_basket_runner",
            symbol=MT5_MAP.get(alias, alias),
            direction="buy" if tgt > 0 else ("sell" if tgt < 0 else "close"),
            entry_time=date,
            entry_price=None,
            exit_reason="dry_run_intent",
            entry_reason=act,
            confidence=None,
            volume=abs(round(tgt, 4)),
            magic=magic,
            run_id=run_id,
            dry_run=1,
        ))
        n += 1
    return n


def run_book(label: str, targets: dict, prev: dict, kept: list) -> dict:
    """Print one book (pure or overlay) and return per-alias actions."""
    gross = sum(abs(v) for v in targets.values())
    print(f"\n  [{label}]  {'alias':8} {'broker':8} {'target':>8} {'dir':>6} "
          f"{'action':>7} {'prev':>8}")
    actions = {}
    for a in kept:
        tgt = float(targets[a])
        pv = float(prev.get(a, 0.0))
        act = _action(pv, tgt)
        actions[a] = act
        d = "LONG" if tgt > 1e-6 else "SHORT" if tgt < -1e-6 else "flat"
        print(f"  {'':10}{a:8} {MT5_MAP.get(a, a):8} {tgt:>+8.3f} {d:>6} "
              f"{act:>7} {pv:>+8.3f}")
    print(f"  [{label}]  gross={gross:.2f} net={sum(targets.values()):+.2f}")
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    ap.add_argument("--equity", type=float, default=None,
                    help="book equity in USD. NOT account currency — this HFM account "
                         "is denominated in KES; passing the raw KES balance oversizes "
                         "every leg by the USDKES rate (~129x).")
    ap.add_argument("--live", action="store_true",
                    help="with --equity: query MT5 for exact specs")
    ap.add_argument("--overlay", action="store_true",
                    help="apply the AI agent risk overlay (scales in [0,1])")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-journal even if state date == today's bar")
    args = ap.parse_args()

    champ = load_champion_config(Path(args.config))
    cfg = champ["lever_cfg"]
    run_dir = ROOT / "data" / "v5_runs" / champ["run_id_pure"]
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"
    history_file = run_dir / "positions.csv"

    close, spread, kept = build_panels(champ["instruments"], "D1")
    classes = asset_classes(kept)
    pos = lever_positions(close, kept, classes, cfg)

    last_date = pos.index[-1]
    today = pos.iloc[-1]
    date_str = str(last_date.date())
    age = (pd.Timestamp.utcnow().tz_localize(None) - last_date).days

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    prev = state.get("positions", {})
    already_ran = state.get("date") == date_str

    print(f"\n{'=' * 78}\n  V5 BASKET — targets as of {date_str}"
          f"  (tier {champ['tier']}, promoted {champ['promoted_from_run_id']},"
          f" expected Sharpe {champ['expected_sharpe']})\n{'=' * 78}")
    if age > STALE_DAYS:
        print(f"  ! DATA IS {age}d STALE — refresh data/*_D1_long.csv before acting")
    if already_ran and not args.force:
        print(f"  (state already at {date_str} — reprint only, no journal/history append)")

    raw_targets = {a: round(float(today[a]), 4) for a in kept}
    actions = run_book("pure", raw_targets, prev, kept)

    scaled_targets, scales, overlay_meta = raw_targets, None, None
    if args.overlay:
        from src.v5.agents import build_evidence_pack, run_overlay
        evidence = build_evidence_pack(
            date=date_str, targets=raw_targets, prev_positions=prev,
            actions=actions, close=close, kept=kept)
        overlay_run_dir = ROOT / "data" / "v5_runs" / champ["run_id_overlay"]
        scales, overlay_meta = run_overlay(evidence, overlay_run_dir)
        scaled_targets = {a: round(raw_targets[a] * scales[a], 4) for a in kept}
        prev_overlay_state = overlay_run_dir / "state.json"
        prev_ov = (json.loads(prev_overlay_state.read_text()).get("positions", {})
                   if prev_overlay_state.exists() else {})
        overlay_actions = run_book("overlay", scaled_targets, prev_ov, kept)
        print(f"  [overlay] scales={scales}  reason={overlay_meta.get('reason')}")

    if args.equity:
        _print_sizing({a: float(scaled_targets[a]) for a in kept},
                      kept, close, args.equity, args.live)

    if args.validate:
        pnl = mtm_pnl_price_units(pos, close, spread).loc["2010-01-01":]
        net = pnl["net"].dropna()
        sh = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
        print(f"\n  [validate] full net Sharpe={sh:+.3f} "
              f"(promoted run: {champ['expected_sharpe']:+.3f})")

    if not already_ran or args.force:
        journal = TradeJournal(args.journal)
        n = journal_intents(journal, date_str, actions, raw_targets,
                            champ["run_id_pure"], champ["magic"])
        row = {"date": date_str, **{f"raw_{a}": raw_targets[a] for a in kept}}
        if args.overlay:
            overlay_run_dir.mkdir(parents=True, exist_ok=True)
            n += journal_intents(journal, date_str, overlay_actions,
                                 scaled_targets, champ["run_id_overlay"],
                                 champ["magic"] + 1)
            row.update({f"scaled_{a}": scaled_targets[a] for a in kept})
            (overlay_run_dir / "state.json").write_text(json.dumps(
                {"date": date_str, "positions": scaled_targets}, indent=2))
        hist_row = pd.DataFrame([row])
        hist_row.to_csv(history_file, mode="a" if history_file.exists() else "w",
                        header=not history_file.exists(), index=False)
        state_file.write_text(json.dumps(
            {"date": date_str, "positions": raw_targets}, indent=2))
        print(f"\n  journaled {n} dry-run intents  state -> {state_file}"
              f"\n  (advisory only — no live orders placed)")


if __name__ == "__main__":
    main()
