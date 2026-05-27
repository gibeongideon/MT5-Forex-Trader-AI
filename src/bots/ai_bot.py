"""
AI trading bot: uses Claude API to analyze EURUSD M15 candles and decide BUY/SELL/HOLD.
Trades once per newly closed candle. Risk management and position sizing from BotBase.

Run:
    conda activate envmt5
    python src/bots/ai_bot.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import anthropic

from src.core.bot_base import BotBase


class AIBot(BotBase):

    def __init__(self):
        super().__init__(name="AIBot", tick_interval=60.0)

        cfg = self.config.get("ai_bot", {})
        self.symbol: str       = cfg.get("symbol", "EURUSD")
        self.timeframe: str    = cfg.get("timeframe", "M15")
        self.sl_pips: float    = cfg.get("sl_pips", 30)
        self.tp_pips: float    = cfg.get("tp_pips", 60)
        self.candles: int      = cfg.get("candles", 50)
        self.min_confidence: str = cfg.get("min_confidence", "medium")

        self._ai = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self._last_candle_time = None
        self._last_action: str = "hold"

    # ------------------------------------------------------------------ #
    # BotBase interface
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:
        info = self.conn.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol {self.symbol} not found on this broker.")
        self.log(f"Symbol: {self.symbol}  TF: {self.timeframe}  SL={self.sl_pips}p  TP={self.tp_pips}p")
        self.log("AI analyst: claude-opus-4-7 (adaptive thinking)")

    def on_tick(self) -> None:
        df = self.rates(self.symbol, self.timeframe, count=self.candles)
        candle_time = df.index[-1]

        # Only act when a new candle has closed
        if candle_time == self._last_candle_time:
            return
        self._last_candle_time = candle_time

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values

        rsi   = _rsi(closes, 14)
        sma20 = _sma(closes, 20)
        sma50 = _sma(closes, 50)

        context = _build_context(
            self.symbol, self.timeframe, df,
            closes, highs, lows, rsi, sma20, sma50,
        )

        self.log(f"Candle {candle_time} closed — querying Claude...")
        decision   = self._query_claude(context)
        action     = decision.get("action", "hold").lower()
        confidence = decision.get("confidence", "low").lower()
        reason     = decision.get("reason", "")

        self.log(f"Signal: {action.upper()} | {confidence} confidence | {reason}")

        confidence_ok = (
            confidence == "high"
            or (confidence == "medium" and self.min_confidence == "medium")
        )
        if action == "hold" or not confidence_ok:
            return
        if action == self._last_action:
            self.log(f"Signal unchanged ({action}) — skipping duplicate.")
            return

        self._last_action = action
        self._execute(action)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _query_claude(self, context: str) -> dict:
        prompt = (
            f"{context}\n\n"
            f"Analyze this {self.symbol} market data and decide: BUY, SELL, or HOLD.\n\n"
            "Reply with a JSON object only — no other text:\n"
            '{"action": "buy"|"sell"|"hold", "confidence": "high"|"medium"|"low", "reason": "<30 words"}'
        )
        try:
            resp = self._ai.messages.create(
                model="claude-opus-4-7",
                max_tokens=2048,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            return _parse_json(text)
        except anthropic.APIError as e:
            self.log(f"Claude API error: {e}")
            return {"action": "hold", "confidence": "low", "reason": "API error"}

    def _execute(self, action: str) -> None:
        # Close opposing positions first
        opposing = "sell" if action == "buy" else "buy"
        for pos in self.open_positions(self.symbol):
            if _pos_dir(pos) == opposing:
                self.conn.close_position(pos)

        if self.open_count(self.symbol) >= self.max_open_trades:
            self.log("Max open trades reached — skipping.")
            return

        info   = self.conn.symbol_info(self.symbol)
        pip    = info.point * 10
        tick   = self.conn.get_tick(self.symbol)
        volume = self.calc_lot(self.symbol, self.sl_pips)

        if action == "buy":
            entry = tick.ask
            sl = entry - self.sl_pips * pip
            tp = entry + self.tp_pips * pip
            self.log(f"BUY  vol={volume}  sl={sl:.5f}  tp={tp:.5f}")
            self.buy(self.symbol, volume, sl=sl, tp=tp, comment="AIBot")
        else:
            entry = tick.bid
            sl = entry + self.sl_pips * pip
            tp = entry - self.tp_pips * pip
            self.log(f"SELL vol={volume}  sl={sl:.5f}  tp={tp:.5f}")
            self.sell(self.symbol, volume, sl=sl, tp=tp, comment="AIBot")


# ─── market context ──────────────────────────────────────────────────────────

def _build_context(symbol, timeframe, df, closes, highs, lows, rsi, sma20, sma50) -> str:
    trend  = "BULLISH" if sma20[-1] > sma50[-1] else "BEARISH"
    rsi_v  = rsi[-1]
    rsi_lbl = " (overbought)" if rsi_v > 70 else " (oversold)" if rsi_v < 30 else ""
    recent = df.tail(10)

    lines = [
        f"Symbol: {symbol}  Timeframe: {timeframe}",
        f"Price:     {closes[-1]:.5f}",
        f"RSI(14):   {rsi_v:.1f}{rsi_lbl}",
        f"SMA20:     {sma20[-1]:.5f}  SMA50: {sma50[-1]:.5f}  MA trend: {trend}",
        f"10-bar range: {lows[-10:].min():.5f} — {highs[-10:].max():.5f}",
        "",
        "Last 10 closed candles (time  open  high  low  close):",
    ]
    for ts, row in recent.iterrows():
        lines.append(
            f"  {ts.strftime('%m-%d %H:%M')}  "
            f"{row['open']:.5f}  {row['high']:.5f}  "
            f"{row['low']:.5f}  {row['close']:.5f}"
        )
    return "\n".join(lines)


# ─── technical indicators ────────────────────────────────────────────────────

def _sma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        result[i] = values[i - period + 1:i + 1].mean()
    return result


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    delta  = np.diff(values, prepend=values[0])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    result = np.full(len(values), np.nan)
    if len(values) <= period:
        return result

    avg_gain = gains[1:period + 1].mean()
    avg_loss = losses[1:period + 1].mean()

    for i in range(period, len(values)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


# ─── utils ───────────────────────────────────────────────────────────────────

def _pos_dir(position) -> str:
    return "buy" if position.type == 0 else "sell"


def _parse_json(text: str) -> dict:
    """Extract JSON from Claude's response, tolerating markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"action": "hold", "confidence": "low", "reason": "JSON parse error"}


if __name__ == "__main__":
    bot = AIBot()
    bot.run()
