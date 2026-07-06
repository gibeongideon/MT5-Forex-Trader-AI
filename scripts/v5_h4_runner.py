"""v5_h4_runner.py — 4H-bar advisory runner for the minimal-capital H4 book.

Every completed 4H bar: rebuild target weights with the validated H4 engine
(`src/v5/h4_cta.py`, run ids `h4-cta-v5*`), quantize them to broker lots at
the account equity (`src/v5/h4_discrete.py`, validated by `h4-capital-sweep`),
diff against held lots, journal every lot change as a DRY-RUN intent, and
persist state. ADVISORY / PAPER ONLY — no order-send code path.

Intended cadence: invoke shortly after each 4H bar close (cron or /loop),
with fresh `data/<SYM>_H4_long.csv` files.

    python scripts/v5_h4_runner.py --equity 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.trade_journal import TradeJournal
from src.v5.h4_cta import CONFIG, h4_positions, load_h4_panel
from src.v5.h4_discrete import target_lots_today

CONFIG_FILE = ROOT / "configs" / "v5_h4_day_trader.json"
STALE_HOURS = 12


def load_day_config(path: Path = CONFIG_FILE) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    ap.add_argument("--equity", type=float, default=None,
                    help="account equity USD (default from config)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_day_config(Path(args.config))
    equity = args.equity if args.equity is not None else cfg["default_equity"]
    run_dir = ROOT / "data" / "v5_runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"

    close, spread = load_h4_panel(args.data_dir, symbols=tuple(cfg["symbols"]))
    positions = h4_positions(close, {**CONFIG, **cfg["lever_cfg"]})
    last_bar = positions.index[-1]
    bar_str = str(last_bar)
    weights = {s: round(float(positions[s].iloc[-1]), 4) for s in cfg["symbols"]}
    prices = {s: float(close[s].iloc[-1]) for s in cfg["symbols"]}
    lots_info = target_lots_today(weights, equity, prices)

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    held = state.get("lots", {})
    already_ran = state.get("bar") == bar_str
    age_h = (pd.Timestamp.utcnow().tz_localize(None) - last_bar).total_seconds() / 3600

    print(f"\n{'=' * 74}\n  V5 H4 DAY BOOK — bar {bar_str}  equity ${equity:,.0f}"
          f"  (vol target {cfg['lever_cfg']['target_vol']:.0%})\n{'=' * 74}")
    if equity < cfg["min_recommended_equity"]:
        print(f"  ! equity below validated floor ${cfg['min_recommended_equity']:,.0f} "
              f"— 0.01-lot grid dominates outcomes at this size")
    if age_h > STALE_HOURS:
        print(f"  ! DATA IS {age_h:.0f}h STALE — refresh data/*_H4_long.csv before acting")
    if already_ran and not args.force:
        print(f"  (state already at {bar_str} — reprint only)")

    intents = []
    print(f"\n  {'symbol':8} {'weight':>8} {'ideal':>8} {'LOTS':>7} {'held':>7} "
          f"{'notional$':>10}  flag")
    for s in cfg["symbols"]:
        r = lots_info[s]
        hv = float(held.get(s, 0.0))
        delta = r["lots"] - hv
        flag = "ROUND->0" if r["rounded_zero"] else ""
        act = "hold" if abs(delta) < 1e-9 else ("buy" if delta > 0 else "sell")
        print(f"  {s:8} {weights[s]:>+8.3f} {r['ideal_lots']:>+8.3f} "
              f"{r['lots']:>+7.2f} {hv:>+7.2f} {r['notional']:>10,.0f}  {flag}")
        if act != "hold":
            intents.append(dict(
                bot="v5_h4_runner", symbol=s, direction=act,
                entry_time=bar_str, entry_reason=f"delta_lots={round(delta, 2)}",
                volume=abs(round(delta, 2)), magic=cfg["magic"],
                run_id=cfg["run_id"], dry_run=1))

    gross = sum(abs(r["notional"]) for r in lots_info.values())
    print(f"\n  gross notional ${gross:,.0f} ({gross / equity:.2f}x equity)")
    zeroed = [s for s in cfg["symbols"] if lots_info[s]["rounded_zero"]]
    if zeroed:
        print(f"  ! legs rounded to ZERO at this equity: {zeroed}")

    if (not already_ran) or args.force:
        journal = TradeJournal(args.journal)
        for it in intents:
            journal.record(it)
        row = {"bar": bar_str, "equity": equity,
               **{f"w_{s}": weights[s] for s in cfg["symbols"]},
               **{f"lots_{s}": lots_info[s]["lots"] for s in cfg["symbols"]}}
        hist = run_dir / "positions.csv"
        pd.DataFrame([row]).to_csv(hist, mode="a" if hist.exists() else "w",
                                   header=not hist.exists(), index=False)
        state_file.write_text(json.dumps(
            {"bar": bar_str, "equity": equity,
             "lots": {s: lots_info[s]["lots"] for s in cfg["symbols"]}}, indent=2))
        print(f"  journaled {len(intents)} dry-run intents  state -> {state_file}")
    print("  (advisory only — no live orders placed)")


if __name__ == "__main__":
    main()
