"""
Base class for all MT5 trading bots.
Subclass this and implement on_tick() with your strategy logic.
"""

import signal
import time
from abc import ABC, abstractmethod
from datetime import datetime, date, time as dtime

import yaml
from pathlib import Path

from .mt5_connector import MT5Connector

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class BotBase(ABC):
    """
    Lifecycle:
        bot = MyBot()
        bot.run()           # blocks; Ctrl+C to stop
    """

    def __init__(self, name: str = "Bot", tick_interval: float = 5.0):
        with open(_CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)

        trading = self.config["trading"]
        self.name = name
        self.magic = trading["magic_number"]
        self.max_open_trades: int = trading["max_open_trades"]
        self.max_daily_loss: float = trading["max_daily_loss"]
        self.risk_per_trade: float = trading["risk_per_trade"]
        self.tick_interval = tick_interval  # seconds between on_tick() calls

        self.conn: MT5Connector = MT5Connector()
        self._running = False
        self._daily_loss = 0.0
        self._day_start_balance = 0.0
        self._last_day: date = date.today()

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def on_tick(self) -> None:
        """Called every tick_interval seconds. Put your strategy logic here."""

    def on_start(self) -> None:
        """Called once after successful connection. Override for init logic."""

    def on_stop(self) -> None:
        """Called once before disconnecting. Override for cleanup."""

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print(f"[{self.name}] Starting...")
        self._connect_with_retry()
        self._day_start_balance = self.conn.account_balance()
        self._running = True

        try:
            self.on_start()
            print(f"[{self.name}] Running. Press Ctrl+C to stop.")
            while self._running:
                try:
                    self._check_daily_loss()
                    self.on_tick()
                except Exception as e:
                    print(f"[{self.name}] ERROR in on_tick: {e}")
                time.sleep(self.tick_interval)
        finally:
            self.on_stop()
            self.conn.disconnect()
            print(f"[{self.name}] Stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Helpers available to subclasses
    # ------------------------------------------------------------------ #

    def open_positions(self, symbol: str = None):
        return self.conn.get_positions(symbol=symbol, magic=self.magic)

    def open_count(self, symbol: str = None) -> int:
        return len(self.open_positions(symbol))

    def buy(self, symbol: str, volume: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> dict:
        return self.conn.open_position(symbol, "buy", volume, sl, tp, comment, self.magic)

    def sell(self, symbol: str, volume: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> dict:
        return self.conn.open_position(symbol, "sell", volume, sl, tp, comment, self.magic)

    def close_all(self, symbol: str = None) -> None:
        for pos in self.open_positions(symbol):
            self.conn.close_position(pos)

    def calc_lot(self, symbol: str, sl_pips: float) -> float:
        return self.conn.calc_lot_size(symbol, sl_pips, self.risk_per_trade)

    def risk_sized_lot(
        self,
        symbol:       str,
        confidence:   float,
        sl_pips:      float,
        tp_pips:      float,
        atr_value:    float = 0.0,
        drawdown_pct: float = 0.0,
        risk_config=None,
    ) -> tuple[float, float]:
        """
        Phase 8 helper: returns (lot_size, effective_sl_pips) using
        confidence-scaled risk sizing.

        Parameters
        ----------
        symbol       : trading symbol (e.g. "EURUSD")
        confidence   : model's P_buy or P_sell for this signal
        sl_pips      : baseline stop-loss pips (overridden if ATR stop enabled)
        tp_pips      : baseline take-profit pips
        atr_value    : raw ATR in price units (0 = use fixed SL)
        drawdown_pct : current peak-to-trough drawdown (0–1)
        risk_config  : RiskConfig instance; uses defaults if None

        Returns
        -------
        (lot_size, effective_sl_pips)
        lot_size is 0.0 when confidence is below the minimum threshold.
        """
        from src.core.risk_manager import RiskManager, RiskConfig
        rm = RiskManager(risk_config or RiskConfig())
        pip_size = 0.0001  # EURUSD default

        sizing = rm.size(
            confidence=confidence,
            balance=self.conn.account_balance(),
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            drawdown_pct=drawdown_pct,
            atr_value=atr_value,
            pip_size=pip_size,
        )
        if sizing.skip:
            return 0.0, sl_pips

        lot = self.conn.calc_lot_size(symbol, sizing.sl_pips, sizing.risk_pct)
        return lot, sizing.sl_pips

    def in_session(self) -> bool:
        """
        Return True if the current UTC time is within the configured trading session.

        Reads trading.session_filter from config.yaml:
            enabled   : bool   — if false, always return True
            start_utc : "HH:MM" — session open (inclusive)
            end_utc   : "HH:MM" — session close (exclusive)

        Designed to be called at the top of on_tick() so the bot only enters
        new trades during the active London/NY trading hours.
        """
        sf = self.config.get("trading", {}).get("session_filter", {})
        if not sf.get("enabled", False):
            return True
        now_utc = datetime.utcnow().time()
        start = dtime.fromisoformat(sf.get("start_utc", "00:00"))
        end   = dtime.fromisoformat(sf.get("end_utc",   "23:59"))
        return start <= now_utc < end

    def rates(self, symbol: str, timeframe: str, count: int = 200):
        return self.conn.get_rates(symbol, timeframe, count)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{self.name}] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _check_daily_loss(self) -> None:
        today = date.today()
        if today != self._last_day:
            self._day_start_balance = self.conn.account_balance()
            self._daily_loss = 0.0
            self._last_day = today
            return

        balance = self.conn.account_balance()
        if balance <= 0:
            return  # connection error — skip check rather than triggering false loss limit
        loss_pct = (self._day_start_balance - balance) / self._day_start_balance if self._day_start_balance > 0 else 0
        if loss_pct >= self.max_daily_loss:
            self.log(f"Daily loss limit reached ({loss_pct:.1%}). Closing all positions and stopping.")
            self.close_all()
            self.stop()

    def _connect_with_retry(self, max_wait: int = 60) -> None:
        """Retry bridge connection with backoff — waits for MT5 to start."""
        delay = 5
        attempt = 0
        while True:
            try:
                self.conn.connect()
                return
            except Exception as e:
                attempt += 1
                print(f"[{self.name}] MT5 bridge not ready (attempt {attempt}): {e}")
                print(f"[{self.name}] Retrying in {delay}s — start MT5 with ./start_mt5.sh")
                time.sleep(delay)
                delay = min(delay * 2, max_wait)
                # Reset cached singleton so next attempt creates a fresh connection
                import src.core.mt5_connector as _mc
                _mc._mt5 = None

    def _handle_signal(self, signum, frame) -> None:
        print(f"\n[{self.name}] Signal {signum} received, shutting down...")
        self.stop()
