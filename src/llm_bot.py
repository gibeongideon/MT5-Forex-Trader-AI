"""
LLM Signal Bot — live M15 trading using Claude as signal predictor.

Runs on bar close (every 15 min). Fetches the last 50 OHLCV bars, calls
the Predictor (which uses LLMSignalModel → Claude API), and places a
buy/sell order when confidence is above threshold.

Usage:
    conda activate envmt5
    ./start_mt5.sh          # if not already running
    python src/llm_bot.py

Config:
    config.yaml → ai_bot section (symbol, sl_pips, tp_pips, min_confidence)
    config.yaml → llm_signal section (model_id, provider, cache_bars)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot_base import BotBase
from src.predictor import Predictor

# confidence label → minimum P value
_CONFIDENCE_MIN = {
    "high":   0.60,
    "medium": 0.45,
    "low":    0.35,
}


class LLMBot(BotBase):

    def __init__(self):
        super().__init__(name="LLMBot", tick_interval=10.0)

        cfg = self.config.get("ai_bot", {})
        self.symbol     = cfg.get("symbol",    "EURUSD")
        self.timeframe  = cfg.get("timeframe", "M15")
        self.sl_pips    = float(cfg.get("sl_pips", 30))
        self.tp_pips    = float(cfg.get("tp_pips", 60))
        self.candles    = int(cfg.get("candles", 50))

        min_conf_label  = cfg.get("min_confidence", "medium")
        self.threshold  = _CONFIDENCE_MIN.get(min_conf_label, 0.45)

        self._predictor     = Predictor(threshold=self.threshold)
        self._last_bar_time = None   # timestamp of last processed bar

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        info = self.conn.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol {self.symbol} not found.")
        self.log(f"Symbol: {self.symbol}  digits={info.digits}")
        self.log(f"Timeframe: {self.timeframe}  SL={self.sl_pips}p  TP={self.tp_pips}p")
        self.log(f"Confidence threshold: {self.threshold} ({self.config['ai_bot'].get('min_confidence','medium')})")
        self.log(f"LLM model: {self.config['llm_signal'].get('model_id','?')}  provider={self.config['llm_signal'].get('provider','?')}")
        self.log("Waiting for first M15 bar close...")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def on_tick(self) -> None:
        df = self.rates(self.symbol, self.timeframe, count=self.candles)
        if df is None or len(df) < 10:
            return

        # Act only on bar close — skip if we already processed this bar
        latest_bar_time = df.index[-1]
        if latest_bar_time == self._last_bar_time:
            return
        self._last_bar_time = latest_bar_time

        # ── Get prediction ─────────────────────────────────────────────────────
        result = self._predictor.predict(df)
        signal = result["signal"]
        conf   = result["confidence"]

        self.log(
            f"Bar {latest_bar_time}  "
            f"P_buy={result['P_buy']:.2f}  P_hold={result['P_hold']:.2f}  P_sell={result['P_sell']:.2f}  "
            f"→ {signal.upper()} (conf={conf:.2f})"
        )

        if signal == "hold":
            return

        # ── Risk check ─────────────────────────────────────────────────────────
        if self.open_count(self.symbol) >= self.max_open_trades:
            self.log("Max open trades reached — skipping.")
            return

        # Close any opposite position before opening new one
        opposing = "sell" if signal == "buy" else "buy"
        for pos in self.open_positions(self.symbol):
            if _direction(pos) == opposing:
                self.log(f"Closing opposing {opposing.upper()} position {pos.ticket}")
                self.conn.close_position(pos)

        # ── Place order ────────────────────────────────────────────────────────
        info = self.conn.symbol_info(self.symbol)
        pip  = info.point * 10
        tick = self.conn.get_tick(self.symbol)

        volume, effective_sl = self.risk_sized_lot(
            symbol     = self.symbol,
            confidence = conf,
            sl_pips    = self.sl_pips,
            tp_pips    = self.tp_pips,
        )

        if signal == "buy":
            entry = tick.ask
            sl    = entry - effective_sl * pip
            tp    = entry + self.tp_pips * pip
            res   = self.buy(self.symbol, volume, sl=sl, tp=tp, comment="LLMBot")
        else:
            entry = tick.bid
            sl    = entry + effective_sl * pip
            tp    = entry - self.tp_pips * pip
            res   = self.sell(self.symbol, volume, sl=sl, tp=tp, comment="LLMBot")

        if res:
            self.log(
                f"ORDER PLACED: {signal.upper()}  vol={volume}  "
                f"entry={entry:.5f}  sl={sl:.5f}  tp={tp:.5f}"
            )
        else:
            self.log(f"ORDER FAILED: {signal.upper()}")


def _direction(pos) -> str:
    return "buy" if pos.type == 0 else "sell"


if __name__ == "__main__":
    bot = LLMBot()
    bot.run()
