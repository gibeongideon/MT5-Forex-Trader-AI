"""
MT5 connection manager for Ubuntu/Wine using the mt5linux bridge.

Architecture:
  Linux Python (this code)
      ↕ rpyc socket (localhost:18812)
  Wine Python (bridge server - start with: ./start_mt5.sh)
      ↕ Windows IPC (named pipe)
  terminal64.exe running under Wine (~/.mt5)
"""

import os
import subprocess
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from mt5linux import MetaTrader5

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

# Module-level singleton — shared across all connector instances
_mt5: MetaTrader5 | None = None


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_mt5(host: str = "localhost", port: int = 18812) -> MetaTrader5:
    """Return the shared mt5linux instance, creating it if needed."""
    global _mt5
    if _mt5 is None:
        try:
            _mt5 = MetaTrader5(host=host, port=port)
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to mt5linux bridge on {host}:{port}\n"
                "Start the bridge first: ./start_mt5.sh\n"
                f"Details: {e}"
            )
    return _mt5


class MT5Connector:
    """
    High-level wrapper around mt5linux for Ubuntu/Wine.

    Usage (context manager):
        with MT5Connector() as conn:
            df = conn.get_rates("EURUSD", "M15")

    Usage (manual):
        conn = MT5Connector()
        conn.connect()
        ...
        conn.disconnect()
    """

    def __init__(self, host: str = "localhost", port: int = 18812):
        cfg = _load_config()
        mt5_cfg = cfg["mt5"]

        self.terminal_path: str = mt5_cfg["terminal_path"]
        self.wine_prefix: str = mt5_cfg["wine_prefix"]
        self.login: int = int(os.getenv("MT5_LOGIN") or mt5_cfg.get("login") or 0)
        self.password: str = os.getenv("MT5_PASSWORD") or mt5_cfg.get("password") or ""
        self.server: str = os.getenv("MT5_SERVER") or mt5_cfg.get("server") or ""
        self.timeout: int = mt5_cfg.get("timeout", 60000)
        self.host = host
        self.port = port
        self._connected = False

    def connect(self) -> None:
        self._mt5 = get_mt5(self.host, self.port)

        kwargs: dict = {"path": self.terminal_path, "timeout": self.timeout}
        if self.login:
            kwargs["login"] = self.login
        if self.password:
            kwargs["password"] = self.password
        if self.server:
            kwargs["server"] = self.server

        if not self._mt5.initialize(**kwargs):
            err = self._mt5.last_error()
            raise ConnectionError(
                f"mt5.initialize() failed: {err}\n"
                "Make sure:\n"
                "  1. MT5 terminal is running  (./start_mt5.sh)\n"
                "  2. Algorithmic trading is enabled: Tools > Options > Expert Advisors\n"
                "  3. You are logged in to a broker account in MT5"
            )

        self._connected = True
        info = self._mt5.account_info()
        if info:
            print(f"Connected: account={info.login}  balance={info.balance:.2f} {info.currency}  server={info.server}")
        else:
            print("Connected to MT5 terminal (not yet logged in to a trading account)")

    def disconnect(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
            print("Disconnected from MT5")

    def reset_singleton(self) -> None:
        """Drop the cached rpyc bridge so the next connect() builds a fresh one."""
        global _mt5
        _mt5 = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #

    def account_info(self):
        return self._mt5.account_info()

    def account_balance(self) -> float:
        info = self._mt5.account_info()
        return info.balance if info else 0.0

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    def get_rates(self, symbol: str, timeframe: str, count: int = 500):
        import pandas as pd

        tf = _parse_timeframe(timeframe, self._mt5)
        rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rate data for {symbol} {timeframe}: {self._mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def get_tick(self, symbol: str):
        return self._mt5.symbol_info_tick(symbol)

    def symbol_info(self, symbol: str):
        return self._mt5.symbol_info(symbol)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Add the symbol to MarketWatch. symbol_info/ticks return None until this
        is done for symbols the terminal is not already watching."""
        return bool(self._mt5.symbol_select(symbol, enable))

    # ------------------------------------------------------------------ #
    # Orders & positions
    # ------------------------------------------------------------------ #

    def _fill_type(self, symbol: str):
        """Pick the order-filling mode the symbol/broker actually supports.
        Brokers differ (IC Markets=IOC, HFM may require FOK) — sending an
        unsupported mode returns retcode 10030 (invalid fill). The symbol's
        `filling_mode` bitmask: 1=FOK allowed, 2=IOC allowed."""
        mt5 = self._mt5
        try:
            mode = int(getattr(self.symbol_info(symbol), "filling_mode", 0))
        except Exception:
            mode = 0
        if mode & 2:   # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        if mode & 1:   # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def open_position(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        magic: int = 0,
    ) -> dict:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for {symbol}")

        price = tick.ask if order_type == "buy" else tick.bid
        action = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._fill_type(symbol),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"order_send failed: retcode={code}  {mt5.last_error()}")

        print(f"Opened {order_type.upper()} {volume} {symbol} @ {result.price}  ticket={result.order}")
        return result._asdict()

    def close_position(self, position) -> dict:
        mt5 = self._mt5
        symbol = position.symbol
        volume = position.volume
        ticket = position.ticket

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for {symbol}")

        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._fill_type(symbol),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"close_position failed: retcode={code}")

        print(f"Closed ticket={ticket} {symbol}")
        return result._asdict()

    def close_position_partial(self, position, volume: float) -> dict:
        """Close only `volume` lots of an open position (deal in the opposite direction).
        Falls back to a full close when volume >= the position's size."""
        mt5 = self._mt5
        if volume is None or volume >= position.volume:
            return self.close_position(position)

        symbol = position.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for {symbol}")

        if position.type == mt5.POSITION_TYPE_BUY:
            order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            order_type, price = mt5.ORDER_TYPE_BUY, tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "partial_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._fill_type(symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"close_position_partial failed: retcode={code}")
        print(f"Partial-closed {volume} of ticket={position.ticket} {symbol}")
        return result._asdict()

    def get_positions(self, symbol: str = None, magic: int = None):
        positions = self._mt5.positions_get(symbol=symbol) if symbol else self._mt5.positions_get()
        if positions is None:
            return []
        if magic is not None:
            positions = [p for p in positions if p.magic == magic]
        return list(positions)

    def modify_position(self, ticket: int, sl: float, tp: float) -> dict:
        """Modify the SL and/or TP of an open position."""
        mt5 = self._mt5
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       sl,
            "tp":       tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"modify_position failed: retcode={code}  ticket={ticket}")
        return result._asdict()

    def calc_lot_size(self, symbol: str, sl_pips: float, risk_pct: float = 0.01) -> float:
        info = self._mt5.symbol_info(symbol)
        account = self._mt5.account_info()
        if info is None or account is None:
            return 0.01

        pip_value = info.trade_tick_value * (info.point / info.trade_tick_size)
        sl_value = sl_pips * pip_value * 10
        risk_amount = account.balance * risk_pct

        lot = risk_amount / sl_value if sl_value > 0 else info.volume_min
        lot = max(info.volume_min, min(info.volume_max, lot))
        step = info.volume_step
        lot = round(round(lot / step) * step, 2)
        return lot


def _parse_timeframe(tf: str, mt5: MetaTrader5) -> int:
    mapping = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    tf = tf.upper()
    if tf not in mapping:
        raise ValueError(f"Unknown timeframe '{tf}'. Valid: {list(mapping)}")
    return mapping[tf]
