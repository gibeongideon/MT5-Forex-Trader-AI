"""v5_xau_runner.py — 4H advisory runner for the standalone XAUUSD trade engine.

Each completed H4 bar it replays the deterministic engine
(`src/v5/xau_trend.py`) over the full H4 history, so its live state (open
position, trail stop, pending flip) is byte-identical to the validated
backtest (`xau-trend-trail-conf-riskscaled`). It prints the current ticket —
direction, lots, stop — plus any action for the NEXT bar open, and journals
dry-run intents. ADVISORY / PAPER ONLY: no order-send path.

    python scripts/v5_xau_runner.py --equity 3000
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
from src.v5.xau_trend import run_trades

CONFIG_FILE = ROOT / "configs" / "v5_xau_trader.json"
STALE_HOURS = 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    ap.add_argument("--data", default=str(ROOT / "data" / "XAUUSD_H4_long.csv"))
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    equity = args.equity if args.equity is not None else cfg["default_equity"]
    run_dir = ROOT / "data" / "v5_runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"

    df = pd.read_csv(args.data, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    res = run_trades(df, equity0=equity, exit_mode=cfg["exit_mode"],
                     flip_mode=cfg["flip_mode"], params=cfg.get("params"))
    bar = df.index[-1]
    bar_str = str(bar)
    sig = float(res["signal"].iloc[-1])
    pos = res["open_position"]
    pending = res["pending"]

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    already_ran = state.get("bar") == bar_str
    age_h = (pd.Timestamp.utcnow().tz_localize(None) - bar).total_seconds() / 3600

    print(f"\n{'=' * 72}\n  V5 XAUUSD TREND — bar {bar_str}  equity ${equity:,.0f}"
          f"\n{'=' * 72}")
    if age_h > STALE_HOURS:
        print(f"  ! DATA IS {age_h:.0f}h STALE — refresh XAUUSD_H4_long.csv before acting")
    if already_ran and not args.force:
        print(f"  (state already at {bar_str} — reprint only)")

    print(f"  forecast {sig:+.2f}  "
          f"({'buy' if sig > 0 else 'sell'} bias, "
          f"{'actionable' if abs(sig) >= 0.5 else 'below entry threshold'})")

    intents = []
    if pos is not None:
        print(f"  POSITION: {'LONG' if pos['dir'] > 0 else 'SHORT'} "
              f"{pos['lots']:.2f} lots @ {pos['entry']:.2f}  "
              f"SL {pos['sl']:.2f}{' (trailing)' if pos['trail_on'] else ''}  "
              f"conf {pos['conf']}")
        intents.append(dict(direction="modify_sl", volume=pos["lots"],
                            reason=f"sl={round(pos['sl'], 2)}"))
    else:
        print("  POSITION: flat")
    if pending is not None:
        d = "buy" if pending["dir"] > 0 else "sell"
        print(f"  NEXT OPEN: {'flip to ' if pos else 'open '}{d.upper()} "
              f"(signal {pending['strength']:+.2f})")
        intents.append(dict(direction=d, volume=0.0,
                            reason=f"signal={round(pending['strength'], 2)}"))
    else:
        print("  NEXT OPEN: no action")

    if (not already_ran) or args.force:
        journal = TradeJournal(args.journal)
        for it in intents:
            journal.record(dict(
                bot="v5_xau_runner", symbol=cfg["symbol"],
                direction=it["direction"], entry_time=bar_str,
                entry_reason=it["reason"], volume=it["volume"],
                magic=cfg["magic"], run_id=cfg["run_id"], dry_run=1))
        state_file.write_text(json.dumps(
            {"bar": bar_str, "equity": equity, "forecast": round(sig, 3),
             "position": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in pos.items() if k != "opened_t"}
             | {"opened_t": str(pos["opened_t"])} if pos else None},
            indent=2, default=str))
        print(f"  journaled {len(intents)} dry-run intents  state -> {state_file}")
    print("  (advisory only — no live orders placed)")


if __name__ == "__main__":
    main()
