"""READ-ONLY broker spread probe (run on the VPS against a live bridge).

Measures the REAL bid/ask spread on gold (and any requested symbols) from the
broker itself, so backtest cost assumptions can be verified against reality.
Does NOT place, modify, or close any order — only *_info / tick / rates reads.

    python scripts/vps_spread_probe.py --port 18813 --samples 40 --interval 2
"""
from __future__ import annotations

import argparse
import statistics
import time

from mt5linux import MetaTrader5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=18813)
    ap.add_argument("--symbols", default="",
                    help="comma list; default = auto-detect all XAU* symbols")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    mt5 = MetaTrader5(host=args.host, port=args.port)
    if not mt5.initialize():
        raise SystemExit(f"initialize() failed: {mt5.last_error()}")

    ai = mt5.account_info()
    print(f"ACCOUNT login={ai.login} server={ai.server} "
          f"currency={ai.currency} balance={ai.balance} "
          f"leverage=1:{ai.leverage} type={'DEMO' if ai.trade_mode==0 else 'REAL/other'}")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        allsy = mt5.symbols_get()
        symbols = [s.name for s in allsy if s.name[:3].upper() == "XAU"]
    print(f"gold symbols found: {symbols}\n")

    for sym in symbols:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        if info is None:
            print(f"{sym}: no symbol_info"); continue
        point = info.point
        contract = getattr(info, "trade_contract_size", None)
        digits = info.digits
        spreads_usd, spreads_pts = [], []
        for _ in range(args.samples):
            t = mt5.symbol_info_tick(sym)
            if t and t.ask > 0 and t.bid > 0:
                spreads_usd.append(t.ask - t.bid)
            si = mt5.symbol_info(sym)
            if si:
                spreads_pts.append(si.spread)      # in points
            time.sleep(args.interval)
        if not spreads_usd:
            print(f"{sym}: no ticks (market closed?)  spread_points sample="
                  f"{spreads_pts[:5]}  point={point}")
            continue
        med = statistics.median(spreads_usd)
        print(f"{sym:12s} contract={contract} digits={digits} point={point}")
        print(f"   spread $/oz  min={min(spreads_usd):.3f} "
              f"median={med:.3f} max={max(spreads_usd):.3f}  "
              f"(n={len(spreads_usd)})")
        if spreads_pts:
            print(f"   spread points median={statistics.median(spreads_pts):.1f} "
                  f"-> ${statistics.median(spreads_pts)*point:.3f}")
        print(f"   >>> vs backtest assumptions: cent $0.34 | raw/ECN $0.12")
    mt5.shutdown()


if __name__ == "__main__":
    main()
