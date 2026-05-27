"""
Example bot: Moving Average Crossover on EURUSD M15.

Strategy:
  - Fast MA (9) crosses above Slow MA (21) → BUY
  - Fast MA (9) crosses below Slow MA (21) → SELL
  - One trade at a time per direction
  - SL = 30 pips, TP = 60 pips (2:1 RR)
  - Position size from 1% account risk

Run:
    conda activate envmt5
    python src/bots/example_bot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.core.bot_base import BotBase


class MACrossBot(BotBase):

    def __init__(self):
        super().__init__(name="MACross", tick_interval=15.0)

        cfg = self.config.get("example_bot", {})
        self.symbol: str = cfg.get("symbol", "EURUSD")
        self.timeframe: str = cfg.get("timeframe", "M15")
        self.fast: int = cfg.get("fast_ma", 9)
        self.slow: int = cfg.get("slow_ma", 21)
        self.sl_pips: float = cfg.get("sl_pips", 30)
        self.tp_pips: float = cfg.get("tp_pips", 60)

        self._prev_signal: str = "none"

    def on_start(self) -> None:
        info = self.conn.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol {self.symbol} not found on this broker.")
        self.log(f"Symbol: {self.symbol}  point={info.point}  digits={info.digits}")
        self.log(f"Strategy: MA({self.fast}) x MA({self.slow})  SL={self.sl_pips}p  TP={self.tp_pips}p")

    def on_tick(self) -> None:
        df = self.rates(self.symbol, self.timeframe, count=self.slow + 5)
        closes = df["close"].values

        fast_ma = _sma(closes, self.fast)
        slow_ma = _sma(closes, self.slow)

        signal = "none"
        if fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]:
            signal = "buy"
        elif fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]:
            signal = "sell"

        if signal == self._prev_signal or signal == "none":
            return

        self._prev_signal = signal
        self.log(f"Signal: {signal.upper()}  fast_ma={fast_ma[-1]:.5f}  slow_ma={slow_ma[-1]:.5f}")

        # Close opposing positions
        opposing = "sell" if signal == "buy" else "buy"
        for pos in self.open_positions(self.symbol):
            if _position_direction(pos) == opposing:
                self.conn.close_position(pos)

        if self.open_count(self.symbol) >= self.max_open_trades:
            self.log("Max open trades reached, skipping.")
            return

        info = self.conn.symbol_info(self.symbol)
        pip = info.point * 10
        tick = self.conn.get_tick(self.symbol)

        if signal == "buy":
            entry = tick.ask
            sl = entry - self.sl_pips * pip
            tp = entry + self.tp_pips * pip
        else:
            entry = tick.bid
            sl = entry + self.sl_pips * pip
            tp = entry - self.tp_pips * pip

        volume = self.calc_lot(self.symbol, self.sl_pips)
        self.log(f"Opening {signal.upper()}  vol={volume}  sl={sl:.5f}  tp={tp:.5f}")

        if signal == "buy":
            self.buy(self.symbol, volume, sl=sl, tp=tp, comment="MACross")
        else:
            self.sell(self.symbol, volume, sl=sl, tp=tp, comment="MACross")


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        result[i] = values[i - period + 1:i + 1].mean()
    return result


def _position_direction(position) -> str:
    # POSITION_TYPE_BUY = 0, POSITION_TYPE_SELL = 1
    return "buy" if position.type == 0 else "sell"


if __name__ == "__main__":
    bot = MACrossBot()
    bot.run()
