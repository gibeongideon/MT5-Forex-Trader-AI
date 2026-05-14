"""
Verify the full connection stack: Linux Python → rpyc bridge → Wine MT5 terminal.

Run (MT5 + bridge must already be running via ./start_mt5.sh):
    conda activate envmt5
    python tests/test_connection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_bridge():
    """Connect to the rpyc bridge server running inside Wine."""
    print("1. Connecting to mt5linux rpyc bridge (localhost:18812)...")
    try:
        from mt5linux import MetaTrader5
        mt5 = MetaTrader5(host="localhost", port=18812)
        print("   OK  Bridge connection established")
        return mt5
    except Exception as e:
        print(f"   FAIL  Cannot connect to bridge: {e}")
        print("   Run ./start_mt5.sh to start MT5 and the bridge server.")
        sys.exit(1)


def test_initialize(mt5, terminal_path: str):
    print(f"\n2. Calling mt5.initialize({terminal_path!r})...")
    ok = mt5.initialize(path=terminal_path)
    if not ok:
        err = mt5.last_error()
        print(f"   FAIL  error: {err}")
        print("   Troubleshooting:")
        print("     • Is terminal64.exe at the configured path?")
        print("     • Is algo trading enabled? Tools > Options > Expert Advisors")
        print("     • Are you logged in to a broker account?")
        return False
    print("   OK  mt5.initialize() succeeded")
    return True


def test_terminal_info(mt5):
    print("\n3. Terminal info...")
    info = mt5.terminal_info()
    if info:
        print(f"   OK  connected={info.connected}  trade_allowed={info.trade_allowed}")
    else:
        print(f"   WARN  terminal_info() returned None: {mt5.last_error()}")


def test_account(mt5):
    print("\n4. Account info...")
    acc = mt5.account_info()
    if acc:
        print(f"   OK  login={acc.login}  balance={acc.balance:.2f} {acc.currency}  server={acc.server}")
    else:
        print("   WARN  Not logged in (account_info is None). Log in to a broker in MT5 first.")


def test_rates(mt5, symbol: str = "EURUSD"):
    print(f"\n5. Fetching 10 M15 candles for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 10)
    if rates is not None and len(rates) > 0:
        tick = mt5.symbol_info_tick(symbol)
        bid = f"{tick.bid:.5f}" if tick else "N/A"
        ask = f"{tick.ask:.5f}" if tick else "N/A"
        print(f"   OK  {len(rates)} candles   current bid={bid}  ask={ask}")
    else:
        print(f"   WARN  No rates for {symbol}: {mt5.last_error()}")
        print(f"   Try a different symbol name (e.g. EURUSDm, EURUSD.pro)")


def main():
    cfg = _cfg()
    terminal_path = cfg["mt5"]["terminal_path"]

    print("=== MT5 Connection Test (Ubuntu / Wine / mt5linux) ===\n")
    mt5 = test_bridge()
    ok = test_initialize(mt5, terminal_path)

    if ok:
        test_terminal_info(mt5)
        test_account(mt5)
        test_rates(mt5)
        mt5.shutdown()
        print("\nAll checks done — MT5 is reachable from Python on Ubuntu!")
    else:
        print("\nFix the issues above and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
