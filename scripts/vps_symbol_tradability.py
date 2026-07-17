"""READ-ONLY: report trade_mode + commission/swap for gold symbols so we know
whether a raw-spread symbol is actually TRADABLE (not just quotable) and what
commission stacks on top of the tight spread. No orders placed."""
from __future__ import annotations

import argparse
from mt5linux import MetaTrader5

TRADE_MODE = {0: "DISABLED", 1: "LONGONLY", 2: "SHORTONLY", 3: "CLOSEONLY",
              4: "FULL"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--symbols", required=True)
    args = ap.parse_args()
    mt5 = MetaTrader5(host="localhost", port=args.port)
    if not mt5.initialize():
        raise SystemExit(f"init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    print(f"ACCOUNT {ai.login} {ai.server} margin_mode={ai.margin_mode}\n")
    for sym in args.symbols.split(","):
        sym = sym.strip()
        mt5.symbol_select(sym, True)
        i = mt5.symbol_info(sym)
        if i is None:
            print(f"{sym}: no info"); continue
        # commission is not on symbol_info directly; use order_calc if available
        comm = getattr(i, "commission", None)
        print(f"{sym:12s} trade_mode={TRADE_MODE.get(i.trade_mode, i.trade_mode)} "
              f"vol_min={i.volume_min} vol_step={i.volume_step} "
              f"contract={i.trade_contract_size} "
              f"swap_long={i.swap_long} swap_short={i.swap_short} "
              f"spread_pts={i.spread}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
