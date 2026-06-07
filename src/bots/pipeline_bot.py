"""
PipelineBot — live MT5 bot powered by the end-to-end PredictorPipeline.

Connects to the ICMarketsKE-Demo account, fetches live M15 EURUSD bars each
minute, runs the trained pipeline predictor, and executes trades on signal.

Safety limits:
  - max 1 position at a time (hard-coded — overrides config max_open_trades)
  - 5% daily loss limit → bot stops and closes all (from BotBase)
  - min 40% confidence threshold (from pipeline config)
  - magic=20260101 — only manages positions opened by this bot
  - session_filter in config.yaml — no new entries outside London/NY hours

Position management (per-tick, before signal check):
  - Breakeven move: once profit >= 1× SL distance, SL moves to entry + 2 pips
  - Protects profits without premature exit

Trade journal:
  - Every signal logged to data/live_trades.db (SQLite, survives crashes)
  - Every order logged on entry, every closed trade logged on exit

Usage:
  # 1. Train / retrain first
  conda run -n envmt5 python scripts/retrain_champion.py

  # 2. Run the bot (Ctrl+C to stop cleanly)
  conda run -n envmt5 python src/bots/pipeline_bot.py

  # 3. Dry run — print signals without trading
  conda run -n envmt5 python src/bots/pipeline_bot.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.core.bot_base import BotBase
from src.core.trade_journal import TradeJournal
from src.pipeline import PredictorPipeline


SYMBOL    = "EURUSD"
TIMEFRAME = "M15"
PIP_SIZE  = 0.0001
BARS      = 200       # bars fetched per tick — enough for all indicators + encoder
BREAKEVEN_BUFFER_PIPS = 2  # SL moved to entry + this many pips at breakeven


class PipelineBot(BotBase):
    """
    Live bot that trades EURUSD M15 using PredictorPipeline signals.

    Tick logic (every 60 s):
      0. _manage_positions() — breakeven SL moves on open positions
      1. Session filter — skip new entries outside configured hours
      2. Fetch 200 M15 bars from MT5
      3. Skip if this bar was already processed
      4. Predict: {signal, confidence, P_buy, P_hold, P_sell, sizing}
      5. HOLD or sizing.skip → do nothing
      6. Close opposite-direction positions opened by this bot
      7. BUY/SELL signal → open 1 position (up to MAX_POSITIONS cap)
    """

    MAX_POSITIONS = 1   # hard cap — one position at a time

    def __init__(self, dry_run: bool = False):
        super().__init__(name="PipelineBot", tick_interval=60.0)
        self.dry_run = dry_run

        # Load trained pipeline from config artifacts directory
        self.pipe = PredictorPipeline.from_config()
        art_dir = (
            self.config
            .get("pipeline", {})
            .get("artifacts", {})
            .get("directory", "data/models/pipeline")
        )
        self.pipe.load(art_dir)

        pl_cfg  = self.config.get("pipeline", {})
        bt_cfg  = pl_cfg.get("backtest", {})
        self.sl_pips = float(bt_cfg.get("sl_pips", 30.0))
        self.tp_pips = float(bt_cfg.get("tp_pips", 60.0))

        # Trade journal — logs every signal and order to SQLite
        self.journal = TradeJournal(db_path=ROOT / "data" / "live_trades.db")

        # Track last processed bar to avoid acting on the same bar twice
        self._last_bar: pd.Timestamp | None = None
        # Track positions we've already moved to breakeven {ticket: True}
        self._breakeven_done: set[int] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        mode = "DRY RUN" if self.dry_run else "LIVE"
        self.log(f"Starting [{mode}]")
        self.log(
            f"Pipeline: {len(self.pipe.feature_names())} features  "
            f"model={self.pipe.cfg.model_type}  "
            f"encoder={'yes (' + self.pipe.cfg.encoder_mode + ')' if self.pipe._enc else 'no'}"
        )
        sf = self.config.get("trading", {}).get("session_filter", {})
        session_str = (
            f"{sf.get('start_utc')}–{sf.get('end_utc')} UTC"
            if sf.get("enabled") else "24/5 (no filter)"
        )
        self.log(
            f"Symbol={SYMBOL}  TF={TIMEFRAME}  "
            f"SL={self.sl_pips:.0f}p  TP={self.tp_pips:.0f}p  "
            f"threshold={self.pipe.cfg.bt_threshold:.0%}  "
            f"max_positions={self.MAX_POSITIONS}  session={session_str}"
        )
        positions = self.open_positions(SYMBOL)
        if positions:
            self.log(f"Existing {SYMBOL} positions: {len(positions)}")
            for p in positions:
                d = "BUY" if p.type == 0 else "SELL"
                self.log(
                    f"  ticket={p.ticket}  {d}  vol={p.volume}  "
                    f"profit={p.profit:+.2f} USD  magic={p.magic}"
                )

    def on_stop(self) -> None:
        self.log("Bot stopped. Positions left open for manual review.")

    # ── Main tick ─────────────────────────────────────────────────────────────

    def on_tick(self) -> None:
        # ── 0. Manage open positions (breakeven) ─────────────────────────────
        if not self.dry_run:
            self._manage_positions()

        # ── 1. Session filter ─────────────────────────────────────────────────
        if not self.in_session():
            return   # outside London/NY hours — no new entries

        # ── 2. Fetch bars ─────────────────────────────────────────────────────
        try:
            ohlcv = self.rates(SYMBOL, TIMEFRAME, BARS)
        except Exception as e:
            self.log(f"get_rates error: {e}")
            if "IPC" in str(e) or "connection" in str(e).lower():
                try:
                    self.conn.connect()
                    self.log("Reconnected to MT5")
                except Exception as re:
                    self.log(f"Reconnect failed: {re}")
            return

        if ohlcv is None or len(ohlcv) < 100:
            self.log("Insufficient bars — waiting...")
            return

        # ── 3. New bar gate ───────────────────────────────────────────────────
        bar_time = ohlcv.index[-1]
        if bar_time == self._last_bar:
            return   # same bar — silent skip
        self._last_bar = bar_time

        # ── 4. Predict ────────────────────────────────────────────────────────
        try:
            sig = self.pipe.predict(ohlcv)
        except Exception as e:
            self.log(f"predict() error: {e}")
            return

        direction  = sig["signal"]
        confidence = sig["confidence"]
        sizing     = sig["sizing"]

        self.log(
            f"BAR {bar_time}  "
            f"{direction.upper():4s}  conf={confidence:.1%}  "
            f"P_buy={sig['P_buy']:.3f}  "
            f"P_hold={sig['P_hold']:.3f}  "
            f"P_sell={sig['P_sell']:.3f}"
        )

        # Log signal to journal (every bar regardless of action)
        try:
            self.journal.record({
                "bot":          self.name,
                "symbol":       SYMBOL,
                "direction":    direction,
                "entry_time":   str(bar_time),
                "entry_price":  0.0,
                "exit_time":    None,
                "exit_price":   None,
                "pnl_pips":     0.0,
                "pnl_dollars":  0.0,
                "model":        self.pipe.cfg.model_type,
                "confidence":   confidence,
                "entry_reason": f"signal:{direction}",
                "exit_reason":  "pending",
                "volume":       0.0,
                "sl_pips":      self.sl_pips,
                "tp_pips":      self.tp_pips,
            })
        except Exception:
            pass  # journal errors must never interrupt trading

        # ── 5. Skip non-actionable signals ────────────────────────────────────
        if direction == "hold" or sizing["skip"]:
            return

        # ── 6. Position management ────────────────────────────────────────────
        our_positions = [p for p in self.open_positions(SYMBOL)
                         if p.magic == self.magic]

        for pos in our_positions:
            pos_dir = "buy" if pos.type == 0 else "sell"
            if pos_dir == direction:
                self.log(f"Already have {direction.upper()} (ticket={pos.ticket}) — skipping")
                return

        # Close opposite-direction positions opened by this bot
        for pos in our_positions:
            pos_dir = "buy" if pos.type == 0 else "sell"
            self.log(
                f"Closing opposite {pos_dir.upper()} "
                f"ticket={pos.ticket}  profit={pos.profit:+.2f} USD"
            )
            if not self.dry_run:
                try:
                    self.conn.close_position(pos)
                    self._breakeven_done.discard(pos.ticket)
                except Exception as e:
                    self.log(f"Close error: {e}")
                    return

        # Re-check count after closes
        if not self.dry_run:
            remaining = [p for p in self.open_positions(SYMBOL) if p.magic == self.magic]
            if len(remaining) >= self.MAX_POSITIONS:
                self.log(f"Still at cap ({self.MAX_POSITIONS}) after closing — skip")
                return

        # ── 7. Size position ──────────────────────────────────────────────────
        lot, eff_sl = self.risk_sized_lot(
            symbol       = SYMBOL,
            confidence   = confidence,
            sl_pips      = self.sl_pips,
            tp_pips      = self.tp_pips,
            drawdown_pct = self._drawdown_pct(),
        )

        if lot <= 0:
            self.log("Lot size 0 (below min) — skipping")
            return

        # ── 8. Open position ──────────────────────────────────────────────────
        tick = self.conn.get_tick(SYMBOL)
        if tick is None:
            self.log("Cannot get tick — skipping")
            return

        if direction == "buy":
            price    = tick.ask
            sl_price = round(price - eff_sl * PIP_SIZE, 5)
            tp_price = round(price + self.tp_pips * PIP_SIZE, 5)
        else:
            price    = tick.bid
            sl_price = round(price + eff_sl * PIP_SIZE, 5)
            tp_price = round(price - self.tp_pips * PIP_SIZE, 5)

        self.log(
            f"{'[DRY] ' if self.dry_run else ''}"
            f"Opening {direction.upper()}  "
            f"lot={lot}  price={price:.5f}  "
            f"SL={eff_sl:.0f}p ({sl_price:.5f})  "
            f"TP={self.tp_pips:.0f}p ({tp_price:.5f})  "
            f"risk={sizing['risk_pct']:.2%}"
        )

        if self.dry_run:
            return

        try:
            if direction == "buy":
                result = self.buy(
                    SYMBOL, lot,
                    sl=sl_price, tp=tp_price,
                    comment=f"pipe {confidence:.0%}",
                )
            else:
                result = self.sell(
                    SYMBOL, lot,
                    sl=sl_price, tp=tp_price,
                    comment=f"pipe {confidence:.0%}",
                )
            ticket = result.get("order")
            self.log(f"Order done — ticket={ticket}")

            # Log filled order to journal
            try:
                self.journal.record({
                    "bot":          self.name,
                    "symbol":       SYMBOL,
                    "direction":    direction,
                    "entry_time":   str(bar_time),
                    "entry_price":  price,
                    "exit_time":    None,
                    "exit_price":   None,
                    "pnl_pips":     0.0,
                    "pnl_dollars":  0.0,
                    "model":        self.pipe.cfg.model_type,
                    "confidence":   confidence,
                    "entry_reason": f"signal:{direction}",
                    "exit_reason":  "open",
                    "volume":       lot,
                    "sl_pips":      eff_sl,
                    "tp_pips":      self.tp_pips,
                })
            except Exception:
                pass

        except Exception as e:
            self.log(f"Order error: {e}")

    # ── Position management ───────────────────────────────────────────────────

    def _manage_positions(self) -> None:
        """
        Run before signal check each tick.
        Moves SL to breakeven once profit >= 1× SL distance (breakeven protection).
        Only fires once per position (tracked in self._breakeven_done).
        """
        try:
            positions = self.open_positions(SYMBOL)
        except Exception:
            return

        for pos in positions:
            if pos.magic != self.magic:
                continue
            if pos.ticket in self._breakeven_done:
                continue
            if pos.sl == 0.0:
                continue  # no SL set — skip

            entry   = pos.price_open
            sl      = pos.sl
            sl_dist = abs(entry - sl)

            if pos.type == 0:  # BUY
                current = pos.price_current
                profit_dist = current - entry
                if profit_dist >= sl_dist:
                    new_sl = round(entry + BREAKEVEN_BUFFER_PIPS * PIP_SIZE, 5)
                    if new_sl > sl:
                        try:
                            self.conn.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                            self._breakeven_done.add(pos.ticket)
                            self.log(
                                f"Breakeven: ticket={pos.ticket}  "
                                f"SL moved {sl:.5f} → {new_sl:.5f}"
                            )
                        except Exception as e:
                            self.log(f"Breakeven modify error: {e}")
            else:  # SELL
                current = pos.price_current
                profit_dist = entry - current
                if profit_dist >= sl_dist:
                    new_sl = round(entry - BREAKEVEN_BUFFER_PIPS * PIP_SIZE, 5)
                    if new_sl < sl:
                        try:
                            self.conn.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                            self._breakeven_done.add(pos.ticket)
                            self.log(
                                f"Breakeven: ticket={pos.ticket}  "
                                f"SL moved {sl:.5f} → {new_sl:.5f}"
                            )
                        except Exception as e:
                            self.log(f"Breakeven modify error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _drawdown_pct(self) -> float:
        try:
            info = self.conn.account_info()
            if info and self._day_start_balance > 0:
                return max(0.0, (self._day_start_balance - info.equity) / self._day_start_balance)
        except Exception:
            pass
        return 0.0


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="PipelineBot — live MT5 predictor")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print signals and sizing without sending any orders",
    )
    args = p.parse_args()
    PipelineBot(dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
