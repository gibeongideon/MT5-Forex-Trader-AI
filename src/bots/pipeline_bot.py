"""
PipelineBot — live MT5 bot powered by the end-to-end PredictorPipeline.

Connects to the ICMarketsKE-Demo account, fetches live M15 EURUSD bars each
minute, runs the trained pipeline predictor, and executes trades on signal.

Safety limits:
  - max 1 position per direction (hedge_loss mode may hold 2 simultaneously)
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

──────────────────────────────────────────────────────────────────────────────
Flip Modes  (--flip-mode)
──────────────────────────────────────────────────────────────────────────────
Controls what happens when a new signal arrives in the OPPOSITE direction to
an existing open position.  Same-direction signals are always ignored (one
position per direction at a time).

  always (default)
    Opposite signal → close existing trade unconditionally → open new trade.
    Simple and aggressive.  Every counter-signal forces a flip regardless of
    whether the existing trade is winning or losing.
    Backtest win rate: ~61%

  hedge_loss  ★ recommended
    Opposite signal + existing trade IN PROFIT  → close it + open new trade.
                                                  (Same as "always".)
    Opposite signal + existing trade IN LOSS    → leave the losing trade open
                                                  (runs to its natural SL/TP)
                                                  AND open the new trade too.
    Both trades run simultaneously as a hedge until the losing one exits.
    Prevents locking in avoidable losses on noisy counter-signals.
    Backtest win rate: ~71%  MaxDD: ~5%

  hedge_exit
    Same as hedge_loss, but adds one extra rule for the losing trade that was
    kept open: close it the moment its profit first crosses above zero instead
    of waiting for the full TP.  Recovers capital faster at break-even.
    Backtest win rate: ~61%  (more trades dilute the rate vs hedge_loss)

  trailing_hedge  (--trail-pips N, default 10)
    Same opening logic as hedge_loss — keep losing trade open, open hedge.
    Once the losing trade turns profitable, a trailing stop (N pips behind
    the peak profit) is activated.  The position stays open while the recovery
    continues; it closes only when profit pulls back N pips from its peak.
    Captures more of the recovery move than hedge_exit (which exits too early)
    while still locking in profit before a reversal erases gains.
    Parameter: --trail-pips (default 10 pips behind peak)

──────────────────────────────────────────────────────────────────────────────
Planned modes (not yet implemented — see scripts/backtest_flip_modes.py)
──────────────────────────────────────────────────────────────────────────────

  lock  (Mode 4 — Classic Net-Zero Hedge)
    Open equal-size opposite trade to freeze the current floating loss.
    Monitor combined P&L of both legs each tick.  When their sum >= 0,
    close both simultaneously for a near-zero net result.
    State: self._pair_map: dict[int, int]  (original_ticket → hedge_ticket)

  ratio_hedge  (Mode 5 — Asymmetric Lot Hedge)
    Open a larger opposite position (default 2× lot) so the hedge earns profit
    faster and can cover the original loss in fewer pips.  When combined P&L
    of the pair >= 0, close both.
    State: self._pair_map: dict[int, int]
    Param: --hedge-ratio FLOAT  (default 2.0)

  partial_close  (Mode 7 — Split-Risk Hedge)
    Close 50% of the losing position immediately at market (halving the loss
    exposure), then open a new full-size position in the new direction.
    The remaining 50% continues as a normal position to its natural SL/TP.
    State: self._partial_done: set[int]  (tickets already half-closed)

  zone_recovery  (Mode 8 — ZRA / Zone Recovery Algorithm)
    Multi-layer geometric hedge: original 1× stays open, hedge at 2× opens
    on opposite signal.  If the 2× hedge starts losing by zone_pips, a 4×
    counter-layer opens (price-triggered, no new signal needed).  4× → 8×
    follows the same rule.  All layers close together when combined P&L >= 0.
    Max 4 layers (blowup protection on trending markets).
    State: self._zone_tickets, self._zone_lots
    Param: --zone-pips FLOAT  (pip gap before new layer, default 30)

──────────────────────────────────────────────────────────────────────────────
Dedicated candle-model modes  (require --candle-model-dir)
──────────────────────────────────────────────────────────────────────────────
These two modes use a separate CatBoost model trained specifically for 1-bar
direction prediction.  They completely bypass the standard 8-step pipeline
logic and have their own independent trade lifecycle.

  candle_predictor  ★ current live champion
    Fires one trade per M15 bar when model confidence >= 0.60.
    Force-closes at the NEXT bar open regardless of P&L.
    SL/TP (10p / 30p) are purely intra-bar flash-crash protection.
    One position open at a time; no simultaneous hedges.
    Backtest (2.4 yr OOS):
      EURUSD  Sharpe +20.1  Win 87.1%  MaxDD 6.7%   ~2,212 trades
      USDJPY  Sharpe +25.6  Win 81.1%  MaxDD 10.9%  ~4,131 trades

  candle_trail  (--trail-activation-pips, --trail-pips-behind, --trail-max-bars-*)
    Same model and entry logic as candle_predictor, but lets winners run:
      • confidence < 0.70  → max 1 bar  (identical to candle_predictor)
      • confidence 0.70–0.80 → max 2 bars
      • confidence ≥ 0.80  → max 4 bars
    Once profit reaches trail_activation_pips (default 15), trailing SL
    activates at trail_pips_behind (default 10) from peak.  SL only moves
    in the profitable direction.  Primary exits: trailing SL hit, TP hit,
    or max bars elapsed.  No new trade while an existing one is open.
    Params: --trail-activation-pips 15  --trail-pips-behind 10
            --trail-max-bars-low 1  --trail-max-bars-med 2  --trail-max-bars-high 4
    Backtest (2.4 yr OOS):
      EURUSD  Sharpe +17.9  Win 90.0%  MaxDD 3.4%   ~1,922 trades  ← half the DD
      USDJPY  Sharpe +21.4  Win 88.5%  MaxDD 5.3%   ~3,778 trades  ← half the DD
    Trade-off: Sharpe slightly lower than candle_predictor; drawdown halved;
    win rate higher; average winning trade is larger.

──────────────────────────────────────────────────────────────────────────────
Full performance summary (backtested, 60k M15 EURUSD bars unless noted)
──────────────────────────────────────────────────────────────────────────────

  ── Standard pipeline modes (XGBoost + enc8 signals, 60k bars in-sample) ──
  Mode              Trades   Win%   Sharpe   MaxDD   Notes
  always             7,636   61%    +12.97   4.9%
  hedge_loss         4,013   71%     +7.47   4.9%    fewest trades, highest win rate
  hedge_exit         6,033   61%    +10.12   5.3%
  trailing_hedge     5,635   69%     +9.23   3.2%    lowest DD among standard modes
  lock               TBD — run scripts/backtest_flip_modes.py
  ratio_hedge        TBD
  partial_close      TBD
  zone_recovery      TBD

  ── Candle modes (CatBoost 1-bar model, 60k M15 bars, OOS 2.4 yr) ─────────
  Mode              Trades   Win%   Sharpe   MaxDD   Notes
  candle_predictor   2,212   87%    +20.1    6.7%    EURUSD — 1-bar force-close
  candle_predictor   4,131   81%    +25.6   10.9%    USDJPY
  candle_trail       1,922   90%    +17.9    3.4%    EURUSD — trailing SL, conf hold
  candle_trail       3,778   89%    +21.4    5.3%    USDJPY ← recommended for live

──────────────────────────────────────────────────────────────────────────────

Usage:
  # 1. Train / retrain first
  conda run -n envmt5 python scripts/retrain_champion.py

  # 2. Standard pipeline mode
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --model-dir data/models/pipeline_EURUSD --flip-mode trailing_hedge --trail-pips 10

  # 3. Candle predictor (1-bar force-close)
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --candle-model-dir data/models/candle_EURUSD --flip-mode candle_predictor

  # 4. Candle trail (trailing SL + confidence hold)
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --candle-model-dir data/models/candle_EURUSD --flip-mode candle_trail \\
      --trail-activation-pips 15 --trail-pips-behind 10

  # 5. Dry run
  conda run -n envmt5 python src/bots/pipeline_bot.py --dry-run \\
      --candle-model-dir data/models/candle_EURUSD --flip-mode candle_trail

  # 6. Backtest candle_trail vs candle_predictor
  conda run -n envmt5 python scripts/backtest_candle_trail.py

  # 7. Compare all standard modes
  conda run -n envmt5 python scripts/backtest_flip_modes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.core.bot_base import BotBase
from src.core.suffix_ae_sizer import SuffixAESizer
from src.core.trade_journal import TradeJournal
from src.pipeline import PredictorPipeline


_DEFAULT_SYMBOL    = "EURUSD"
TIMEFRAME = "M15"
_PIP_SIZES = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
    "USDJPY": 0.01,   "USDCHF": 0.0001, "USDCAD": 0.0001,
    "GBPJPY": 0.01,   "EURJPY": 0.01,   "XAUUSD": 0.01,
}
BARS      = 200       # bars fetched per tick — enough for all indicators + encoder
BREAKEVEN_BUFFER_PIPS = 2  # SL moved to entry + this many pips at breakeven


class PipelineBot(BotBase):
    """
    Live bot that trades any pair using PredictorPipeline signals.

    Tick logic (every 60 s):
      0. _manage_positions() — breakeven SL moves on open positions
      1. Session filter — skip new entries outside configured hours
      2. Fetch 200 M15 bars from MT5
      3. Skip if this bar was already processed
      4. Predict: {signal, confidence, P_buy, P_hold, P_sell, sizing}
      5. HOLD or sizing.skip → do nothing
      6. Handle opposite positions: close (always) or hedge (hedge_loss mode)
      7. BUY/SELL signal → open 1 position (up to MAX_POSITIONS cap)
    """

    MAX_POSITIONS = 1   # hard cap — one position at a time

    MAX_ZONE_LAYERS = 4

    def __init__(self, dry_run: bool = False, symbol: str = _DEFAULT_SYMBOL,
                 model_dir: str | None = None, flip_mode: str = "always",
                 trail_pips: float = 10.0, hedge_ratio: float = 2.0,
                 zone_pips: float = 30.0,
                 candle_model_dir: str | None = None,
                 magic: int | None = None,
                 trail_activation_pips: float = 15.0,
                 trail_pips_behind: float = 10.0,
                 trail_max_bars_low: int = 1,
                 trail_max_bars_med: int = 2,
                 trail_max_bars_high: int = 4):
        super().__init__(name=f"PipelineBot-{symbol}", tick_interval=60.0)
        if magic is not None:
            self.magic = magic  # override config.yaml magic_number
        self.dry_run     = dry_run
        self.symbol      = symbol
        self.flip_mode   = flip_mode
        self.trail_pips  = trail_pips   # trailing_hedge
        self.hedge_ratio = hedge_ratio  # ratio_hedge: hedge lot multiplier
        self.zone_pips   = zone_pips    # zone_recovery: pip gap before next layer

        # Load trained pipeline — prefer explicit --model-dir, then config default
        self.pipe = PredictorPipeline.from_config()
        art_dir = model_dir or (
            self.config
            .get("pipeline", {})
            .get("artifacts", {})
            .get("directory", "data/models/pipeline")
        )
        self.pipe.load(art_dir)

        # pip size: prefer pair_meta.json saved during retrain, then lookup table
        import json
        pair_meta_path = Path(art_dir) / "pair_meta.json"
        if pair_meta_path.exists():
            pm = json.loads(pair_meta_path.read_text())
            self._pip_size = float(pm.get("pip_size", _PIP_SIZES.get(symbol, 0.0001)))
            self.sl_pips   = float(pm.get("sl_pips", 30.0))
            self.tp_pips   = float(pm.get("tp_pips", 60.0))
        else:
            self._pip_size = _PIP_SIZES.get(symbol, 0.0001)

        if not pair_meta_path.exists():
            pl_cfg  = self.config.get("pipeline", {})
            bt_cfg  = pl_cfg.get("backtest", {})
            self.sl_pips = float(bt_cfg.get("sl_pips", 30.0))
            self.tp_pips = float(bt_cfg.get("tp_pips", 60.0))

        # Trade journal — logs every signal and order to SQLite
        self.journal = TradeJournal(db_path=ROOT / "data" / "live_trades.db")

        # Suffix Automaton + Autoencoder proactive lot multiplier (history 150 bars
        # fits inside the 200-bar fetch; algo_mode=1 linear; AE gate on)
        self._sa_sizer = SuffixAESizer(
            history_length=150, dna_window=16, algo_mode=1, use_ae=True
        )

        # candle_predictor / candle_trail: dedicated 1-bar model + ticket tracking
        self._candle_ticket: Optional[int] = None
        self.candle_pipe: Optional[PredictorPipeline] = None
        self._candle_sl_pips: float = 15.0
        self._candle_tp_pips: float = 20.0

        # candle_trail: per-trade state
        self._trail_ticket:       Optional[int] = None
        self._trail_peak_pips:    float         = 0.0
        self._trail_bars_held:    int           = 0
        self._trail_direction:    Optional[str] = None   # "buy" | "sell"
        self._trail_entry_price:  float         = 0.0
        self._trail_max_bars:     int           = 1
        self._trail_activation_pips: float      = trail_activation_pips
        self._trail_pips_behind:     float      = trail_pips_behind
        self._trail_max_bars_low:    int        = trail_max_bars_low
        self._trail_max_bars_med:    int        = trail_max_bars_med
        self._trail_max_bars_high:   int        = trail_max_bars_high

        if flip_mode in ("candle_predictor", "candle_trail"):
            if candle_model_dir is None:
                raise ValueError(
                    f"--candle-model-dir is required when --flip-mode {flip_mode}"
                )
            self.candle_pipe = PredictorPipeline.from_config()
            self.candle_pipe.load(candle_model_dir)
            candle_meta = Path(candle_model_dir) / "pair_meta.json"
            if candle_meta.exists():
                cm = json.loads(candle_meta.read_text())
                self._candle_sl_pips = float(cm.get("sl_pips", 15.0))
                self._candle_tp_pips = float(cm.get("tp_pips", 20.0))

        self._last_bar: pd.Timestamp | None = None
        self._breakeven_done: set[int] = set()
        # hedge_exit
        self._hedged_tickets: set[int] = set()
        # trailing_hedge: ticket → peak profit pips
        self._hedged_trail: dict[int, float] = {}
        # lock / ratio_hedge: orig_ticket → hedge_ticket (and reverse)
        self._pair_map: dict[int, int] = {}
        self._pair_rev: dict[int, int] = {}
        # partial_close: tickets where half-close already executed
        self._partial_done: set[int] = set()
        # zone_recovery: open layer tickets + lot size per ticket
        self._zone_tickets: list[int] = []
        self._zone_lots: dict[int, float] = {}    # ticket → lot size

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
        flip_detail = self.flip_mode
        if self.flip_mode == "trailing_hedge":
            flip_detail += f"(trail={self.trail_pips:.0f}p)"
        elif self.flip_mode == "ratio_hedge":
            flip_detail += f"(r={self.hedge_ratio:.1f}x)"
        elif self.flip_mode == "zone_recovery":
            flip_detail += f"(zone={self.zone_pips:.0f}p)"
        elif self.flip_mode == "candle_predictor":
            flip_detail += f"(SL={self._candle_sl_pips:.0f}p/TP={self._candle_tp_pips:.0f}p force-close=1bar)"
        self.log(
            f"Symbol={self.symbol}  TF={TIMEFRAME}  "
            f"SL={self.sl_pips:.0f}p  TP={self.tp_pips:.0f}p  "
            f"threshold={self.pipe.cfg.bt_threshold:.0%}  "
            f"max_positions={self.MAX_POSITIONS}  flip={flip_detail}  "
            f"session={session_str}"
        )
        positions = self.open_positions(self.symbol)
        if positions:
            self.log(f"Existing {self.symbol} positions: {len(positions)}")
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
        # ── candle modes: completely separate logic ────────────────────────────
        if self.flip_mode == "candle_predictor":
            self._on_tick_candle_predictor()
            return
        if self.flip_mode == "candle_trail":
            self._on_tick_candle_trail()
            return

        # ── 0. Manage open positions (breakeven) ─────────────────────────────
        if not self.dry_run:
            self._manage_positions()

        # ── 1. Session filter ─────────────────────────────────────────────────
        if not self.in_session():
            return   # outside London/NY hours — no new entries

        # ── 2. Fetch bars ─────────────────────────────────────────────────────
        try:
            ohlcv = self.rates(self.symbol, TIMEFRAME, BARS)
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
                "symbol":       self.symbol,
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
        our_positions = [p for p in self.open_positions(self.symbol)
                         if p.magic == self.magic]

        for pos in our_positions:
            pos_dir = "buy" if pos.type == 0 else "sell"
            if pos_dir == direction:
                self.log(f"Already have {direction.upper()} (ticket={pos.ticket}) — skipping")
                return

        # Handle opposite-direction positions
        for pos in our_positions:
            pos_dir = "buy" if pos.type == 0 else "sell"
            if pos_dir == direction:
                continue   # same direction — already guarded above

            in_loss = pos.profit <= 0

            if self.flip_mode in ("hedge_loss", "hedge_exit", "trailing_hedge") and in_loss:
                self.log(
                    f"Hedge mode: {pos_dir.upper()} ticket={pos.ticket} at "
                    f"{pos.profit:+.2f} USD — keeping, adding {direction.upper()} hedge"
                )
                if self.flip_mode == "hedge_exit":
                    self._hedged_tickets.add(pos.ticket)
                elif self.flip_mode == "trailing_hedge":
                    self._hedged_trail.setdefault(pos.ticket, float("-inf"))

            elif self.flip_mode in ("lock", "ratio_hedge") and in_loss:
                if pos.ticket not in self._pair_map and pos.ticket not in self._pair_rev:
                    self.log(
                        f"{'Lock' if self.flip_mode == 'lock' else 'Ratio'} hedge: "
                        f"{pos_dir.upper()} ticket={pos.ticket} {pos.profit:+.2f} USD — "
                        f"keeping, opening {direction.upper()} "
                        f"{'equal' if self.flip_mode == 'lock' else f'{self.hedge_ratio:.1f}×'} hedge"
                    )
                    # Hedge ticket will be stored after the position opens (step 8)
                    self._pair_map[pos.ticket] = -1   # placeholder until step 8

            elif self.flip_mode == "partial_close" and in_loss:
                if pos.ticket not in self._partial_done and not self.dry_run:
                    self.log(
                        f"Partial close: {pos_dir.upper()} ticket={pos.ticket} "
                        f"{pos.profit:+.2f} USD — closing 50%, keeping rest"
                    )
                    try:
                        self.conn.close_position_partial(pos, volume=pos.volume / 2)
                        self._partial_done.add(pos.ticket)
                    except Exception as e:
                        self.log(f"Partial close error (closing full): {e}")
                        self.conn.close_position(pos)
                        self._breakeven_done.discard(pos.ticket)

            elif self.flip_mode == "zone_recovery" and in_loss:
                if len(self._zone_tickets) < self.MAX_ZONE_LAYERS:
                    if pos.ticket not in self._zone_lots:
                        self._zone_tickets.append(pos.ticket)
                        self._zone_lots[pos.ticket] = pos.volume
                    self.log(
                        f"Zone layer {len(self._zone_tickets)}: keeping "
                        f"{pos_dir.upper()} ticket={pos.ticket}, opening 2× {direction.upper()} layer"
                    )
                else:
                    self.log(f"Zone max layers hit — closing all and flipping clean")
                    for zt in list(self._zone_tickets):
                        zp_list = [p for p in self.open_positions(self.symbol)
                                   if p.magic == self.magic and p.ticket == zt]
                        for zp in zp_list:
                            try:
                                self.conn.close_position(zp)
                            except Exception as e:
                                self.log(f"Zone close error: {e}")
                    self._zone_tickets.clear(); self._zone_lots.clear()
                    try:
                        self.conn.close_position(pos)
                        self._breakeven_done.discard(pos.ticket)
                    except Exception as e:
                        self.log(f"Close error: {e}")

            else:
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

        # Re-check count after closes (hedge modes keep opposite open intentionally)
        HEDGE_MODES = ("hedge_loss", "hedge_exit", "trailing_hedge",
                       "lock", "ratio_hedge", "partial_close", "zone_recovery")
        if not self.dry_run:
            if self.flip_mode in HEDGE_MODES:
                remaining = [p for p in self.open_positions(self.symbol)
                             if p.magic == self.magic
                             and ("buy" if p.type == 0 else "sell") == direction]
            else:
                remaining = [p for p in self.open_positions(self.symbol)
                             if p.magic == self.magic]
            if len(remaining) >= self.MAX_POSITIONS:
                self.log(f"Still at cap ({self.MAX_POSITIONS}) after closing — skip")
                return

        # ── 7. Size position ──────────────────────────────────────────────────
        lot, eff_sl = self.risk_sized_lot(
            symbol       = self.symbol,
            confidence   = confidence,
            sl_pips      = self.sl_pips,
            tp_pips      = self.tp_pips,
            drawdown_pct = self._drawdown_pct(),
        )

        # Determine lot multiplier for this open
        _pending_orig = next(
            (orig for orig, hedge in self._pair_map.items() if hedge == -1), None
        )
        if _pending_orig is not None:
            lot_mult = self.hedge_ratio if self.flip_mode == "ratio_hedge" else 1.0
        elif self.flip_mode == "zone_recovery" and self._zone_tickets:
            lot_mult = 2.0 ** len(self._zone_tickets)   # 2×, 4×, 8× for layers 2-4
        else:
            lot_mult = 1.0

        lot = round(lot * lot_mult, 2)

        if lot <= 0:
            self.log("Lot size 0 (below min) — skipping")
            return

        # Apply Suffix Automaton + Autoencoder structural multiplier.
        # closes must be most-recent-first so we reverse the DataFrame order.
        sa_mult = self._sa_sizer.compute(ohlcv["close"].values[::-1].tolist())
        if sa_mult != 1.0:
            lot_before = lot
            lot = round(lot * sa_mult, 2)
            lot = max(lot, self.conn.symbol_info(self.symbol).volume_min)
            self.log(f"SA+AE: mult={sa_mult:.4f}  lot {lot_before} → {lot}")

        # ── 8. Open position ──────────────────────────────────────────────────
        tick = self.conn.get_tick(self.symbol)
        if tick is None:
            self.log("Cannot get tick — skipping")
            return

        if direction == "buy":
            price    = tick.ask
            sl_price = round(price - eff_sl * self._pip_size, 5)
            tp_price = round(price + self.tp_pips * self._pip_size, 5)
        else:
            price    = tick.bid
            sl_price = round(price + eff_sl * self._pip_size, 5)
            tp_price = round(price - self.tp_pips * self._pip_size, 5)

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
                    self.symbol, lot,
                    sl=sl_price, tp=tp_price,
                    comment=f"pipe {confidence:.0%}",
                )
            else:
                result = self.sell(
                    self.symbol, lot,
                    sl=sl_price, tp=tp_price,
                    comment=f"pipe {confidence:.0%}",
                )
            ticket = result.get("order")
            self.log(f"Order done — ticket={ticket}")

            # Link pair (lock / ratio_hedge)
            if _pending_orig is not None and ticket:
                self._pair_map[_pending_orig] = ticket
                self._pair_rev[ticket] = _pending_orig

            # Register zone layer
            if self.flip_mode == "zone_recovery" and ticket:
                self._zone_tickets.append(ticket)
                self._zone_lots[ticket] = lot

            # Log filled order to journal
            try:
                self.journal.record({
                    "bot":          self.name,
                    "symbol":       self.symbol,
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

    # ── Candle predictor ──────────────────────────────────────────────────────

    def _on_tick_candle_predictor(self) -> None:
        """
        1-bar trade logic for candle_predictor mode.

        On every new M15 bar:
          1. Force-close the previous bar's trade (if still open)
          2. Predict current bar's direction using candle_pipe
          3. Open a new trade; store ticket in self._candle_ticket

        SL/TP are intra-bar protective stops only (flash crash).
        Primary exit is always the next bar's open (force-close here).
        """
        if not self.in_session():
            return

        try:
            ohlcv = self.rates(self.symbol, TIMEFRAME, BARS)
        except Exception as e:
            self.log(f"[candle] get_rates error: {e}")
            return

        if ohlcv is None or len(ohlcv) < 100:
            self.log("[candle] Insufficient bars — waiting...")
            return

        bar_time = ohlcv.index[-1]
        if bar_time == self._last_bar:
            return
        self._last_bar = bar_time

        # Step 1: Force-close previous bar's trade
        if self._candle_ticket is not None and not self.dry_run:
            our_pos = [p for p in self.open_positions(self.symbol)
                       if p.magic == self.magic and p.ticket == self._candle_ticket]
            if our_pos:
                self.log(
                    f"[candle] Force-closing bar trade  "
                    f"ticket={self._candle_ticket}  profit={our_pos[0].profit:+.2f} USD"
                )
                try:
                    self.conn.close_position(our_pos[0])
                    self._breakeven_done.discard(self._candle_ticket)
                except Exception as e:
                    self.log(f"[candle] Force-close error: {e}")
            else:
                self.log(
                    f"[candle] ticket={self._candle_ticket} already closed "
                    f"(SL/TP hit intra-bar)"
                )
            self._candle_ticket = None

        # Step 2: Predict with candle model
        try:
            sig = self.candle_pipe.predict(ohlcv)
        except Exception as e:
            self.log(f"[candle] predict() error: {e}")
            return

        direction  = sig["signal"]
        confidence = sig["confidence"]

        self.log(
            f"[candle] BAR {bar_time}  {direction.upper():4s}  "
            f"conf={confidence:.1%}  "
            f"P_buy={sig['P_buy']:.3f}  "
            f"P_hold={sig['P_hold']:.3f}  "
            f"P_sell={sig['P_sell']:.3f}"
        )

        try:
            self.journal.record({
                "bot": self.name, "symbol": self.symbol,
                "direction": direction, "entry_time": str(bar_time),
                "entry_price": 0.0, "exit_time": None, "exit_price": None,
                "pnl_pips": 0.0, "pnl_dollars": 0.0,
                "model": "candle_predictor", "confidence": confidence,
                "entry_reason": f"candle:{direction}", "exit_reason": "pending",
                "volume": 0.0, "sl_pips": self._candle_sl_pips,
                "tp_pips": self._candle_tp_pips,
            })
        except Exception:
            pass

        # Step 3: Skip hold
        if direction == "hold":
            return

        # Step 4: Size and open trade
        lot, eff_sl = self.risk_sized_lot(
            symbol       = self.symbol,
            confidence   = confidence,
            sl_pips      = self._candle_sl_pips,
            tp_pips      = self._candle_tp_pips,
            drawdown_pct = self._drawdown_pct(),
        )
        if lot <= 0:
            self.log("[candle] Lot size 0 — skipping")
            return

        tick = self.conn.get_tick(self.symbol)
        if tick is None:
            self.log("[candle] Cannot get tick — skipping")
            return

        if direction == "buy":
            price    = tick.ask
            sl_price = round(price - eff_sl * self._pip_size, 5)
            tp_price = round(price + self._candle_tp_pips * self._pip_size, 5)
        else:
            price    = tick.bid
            sl_price = round(price + eff_sl * self._pip_size, 5)
            tp_price = round(price - self._candle_tp_pips * self._pip_size, 5)

        self.log(
            f"{'[DRY] ' if self.dry_run else ''}"
            f"[candle] Opening {direction.upper()}  lot={lot}  price={price:.5f}  "
            f"SL={eff_sl:.0f}p  TP={self._candle_tp_pips:.0f}p  "
            f"(force-close next bar)"
        )

        if self.dry_run:
            return

        try:
            if direction == "buy":
                result = self.buy(self.symbol, lot, sl=sl_price, tp=tp_price,
                                  comment=f"candle {confidence:.0%}")
            else:
                result = self.sell(self.symbol, lot, sl=sl_price, tp=tp_price,
                                   comment=f"candle {confidence:.0%}")
            self._candle_ticket = result.get("order")
            self.log(f"[candle] Order done — ticket={self._candle_ticket}")
        except Exception as e:
            self.log(f"[candle] Order error: {e}")

    # ── Candle trail mode ────────────────────────────────────────────────────

    def _reset_trail(self) -> None:
        self._trail_ticket      = None
        self._trail_peak_pips   = 0.0
        self._trail_bars_held   = 0
        self._trail_direction   = None
        self._trail_entry_price = 0.0
        self._trail_max_bars    = 1

    def _on_tick_candle_trail(self) -> None:
        """
        candle_trail mode: same model as candle_predictor but winners can run.

        Every tick:
          A. If a trail trade is open:
               - Compute current profit in pips
               - Once peak >= trail_activation_pips, advance trailing SL
               - On each new bar, increment bar counter; force-close at max_bars
          B. If no trade open and a new bar arrived:
               - Predict direction; determine max_bars from confidence tier
               - Open trade; store ticket and parameters

        max_bars tiers (configurable via CLI):
          conf < 0.70  → max_bars_low  (default 1 — same as candle_predictor)
          conf 0.70–0.80 → max_bars_med  (default 2)
          conf ≥ 0.80  → max_bars_high (default 4)
        """
        if not self.in_session():
            return

        try:
            ohlcv = self.rates(self.symbol, TIMEFRAME, BARS)
        except Exception as e:
            self.log(f"[trail] get_rates error: {e}")
            return

        if ohlcv is None or len(ohlcv) < 100:
            return

        bar_time   = ohlcv.index[-1]
        is_new_bar = bar_time != self._last_bar
        if is_new_bar:
            self._last_bar = bar_time

        # ── A. Manage open trail trade ─────────────────────────────────────
        if self._trail_ticket is not None:
            if not self.dry_run:
                pos = next(
                    (p for p in self.open_positions(self.symbol)
                     if p.magic == self.magic and p.ticket == self._trail_ticket),
                    None,
                )
                if pos is None:
                    self.log(f"[trail] ticket={self._trail_ticket} closed by broker (SL/TP/margin)")
                    self._reset_trail()
                    # Fall through: open next trade if new bar below
                else:
                    # Profit in pips from entry
                    if self._trail_direction == "buy":
                        profit_pips = (pos.price_current - pos.price_open) / self._pip_size
                    else:
                        profit_pips = (pos.price_open - pos.price_current) / self._pip_size

                    if profit_pips > self._trail_peak_pips:
                        self._trail_peak_pips = profit_pips

                    # Trail SL once activation threshold reached
                    if self._trail_peak_pips >= self._trail_activation_pips:
                        trail_offset = self._trail_peak_pips - self._trail_pips_behind
                        if trail_offset > 0:
                            if self._trail_direction == "buy":
                                new_sl = round(pos.price_open + trail_offset * self._pip_size, 5)
                                if new_sl > pos.sl:
                                    try:
                                        self.conn.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                                        self.log(
                                            f"[trail] SL advanced → {new_sl:.5f}  "
                                            f"(+{trail_offset:.1f}p from entry)"
                                        )
                                    except Exception as e:
                                        self.log(f"[trail] modify_position error: {e}")
                            else:
                                new_sl = round(pos.price_open - trail_offset * self._pip_size, 5)
                                if new_sl < pos.sl:
                                    try:
                                        self.conn.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                                        self.log(
                                            f"[trail] SL advanced → {new_sl:.5f}  "
                                            f"(+{trail_offset:.1f}p from entry)"
                                        )
                                    except Exception as e:
                                        self.log(f"[trail] modify_position error: {e}")

                    # Count new bars; force-close at max_bars
                    if is_new_bar:
                        self._trail_bars_held += 1
                        self.log(
                            f"[trail] bar {self._trail_bars_held}/{self._trail_max_bars}  "
                            f"ticket={self._trail_ticket}  "
                            f"profit={profit_pips:+.1f}p  peak={self._trail_peak_pips:.1f}p"
                        )
                        if self._trail_bars_held >= self._trail_max_bars:
                            self.log(
                                f"[trail] Max bars reached — force-closing "
                                f"ticket={self._trail_ticket}  profit={pos.profit:+.2f} USD"
                            )
                            try:
                                self.conn.close_position(pos)
                            except Exception as e:
                                self.log(f"[trail] force-close error: {e}")
                            self._reset_trail()
                    return  # trade still open (or just force-closed) — no new entry

            else:
                # dry_run: simulate bar counting only
                if is_new_bar:
                    self._trail_bars_held += 1
                    self.log(
                        f"[DRY][trail] bar {self._trail_bars_held}/{self._trail_max_bars}  "
                        f"ticket={self._trail_ticket}"
                    )
                    if self._trail_bars_held >= self._trail_max_bars:
                        self.log(f"[DRY][trail] Max bars — would force-close ticket={self._trail_ticket}")
                        self._reset_trail()
                return

        # ── B. Open new trade on new bar ──────────────────────────────────
        if not is_new_bar:
            return

        try:
            sig = self.candle_pipe.predict(ohlcv)
        except Exception as e:
            self.log(f"[trail] predict() error: {e}")
            return

        direction  = sig["signal"]
        confidence = sig["confidence"]

        self.log(
            f"[trail] BAR {bar_time}  {direction.upper():4s}  "
            f"conf={confidence:.1%}  "
            f"P_buy={sig['P_buy']:.3f}  "
            f"P_hold={sig['P_hold']:.3f}  "
            f"P_sell={sig['P_sell']:.3f}"
        )

        if direction == "hold":
            return

        # Confidence tier → max bars
        if confidence < 0.70:
            max_bars = self._trail_max_bars_low
        elif confidence < 0.80:
            max_bars = self._trail_max_bars_med
        else:
            max_bars = self._trail_max_bars_high

        lot, eff_sl = self.risk_sized_lot(
            symbol       = self.symbol,
            confidence   = confidence,
            sl_pips      = self._candle_sl_pips,
            tp_pips      = self._candle_tp_pips,
            drawdown_pct = self._drawdown_pct(),
        )
        if lot <= 0:
            self.log("[trail] Lot size 0 — skipping")
            return

        tick = self.conn.get_tick(self.symbol)
        if tick is None:
            self.log("[trail] Cannot get tick — skipping")
            return

        if direction == "buy":
            price    = tick.ask
            sl_price = round(price - eff_sl * self._pip_size, 5)
            tp_price = round(price + self._candle_tp_pips * self._pip_size, 5)
        else:
            price    = tick.bid
            sl_price = round(price + eff_sl * self._pip_size, 5)
            tp_price = round(price - self._candle_tp_pips * self._pip_size, 5)

        self.log(
            f"{'[DRY] ' if self.dry_run else ''}"
            f"[trail] Opening {direction.upper()}  lot={lot}  price={price:.5f}  "
            f"SL={eff_sl:.0f}p  TP={self._candle_tp_pips:.0f}p  "
            f"max_bars={max_bars}  conf={confidence:.0%}  "
            f"trail_act={self._trail_activation_pips:.0f}p  "
            f"trail_behind={self._trail_pips_behind:.0f}p"
        )

        if self.dry_run:
            # Track in dry-run so bar counting works
            self._trail_ticket      = -1   # sentinel
            self._trail_direction   = direction
            self._trail_entry_price = price
            self._trail_max_bars    = max_bars
            self._trail_peak_pips   = 0.0
            self._trail_bars_held   = 0
            return

        try:
            if direction == "buy":
                result = self.buy(self.symbol, lot, sl=sl_price, tp=tp_price,
                                  comment=f"trail {confidence:.0%}")
            else:
                result = self.sell(self.symbol, lot, sl=sl_price, tp=tp_price,
                                   comment=f"trail {confidence:.0%}")
            self._trail_ticket      = result.get("order")
            self._trail_direction   = direction
            self._trail_entry_price = price
            self._trail_max_bars    = max_bars
            self._trail_peak_pips   = 0.0
            self._trail_bars_held   = 0
            self.log(
                f"[trail] Order done — ticket={self._trail_ticket}  max_bars={max_bars}"
            )
        except Exception as e:
            self.log(f"[trail] Order error: {e}")

    # ── Position management ───────────────────────────────────────────────────

    def _manage_positions(self) -> None:
        """
        Per-tick position management (runs before signal check):
          - trailing_hedge: advance trail stop, close on pull-back
          - hedge_exit: close tracked loser at first profit tick
          - lock / ratio_hedge: close pair when combined P&L >= 0
          - zone_recovery: open next layer when latest zone layer loses zone_pips;
                           close all when combined P&L >= 0
          - breakeven: move SL to entry+2p once profit >= 1× SL distance
        """
        try:
            positions = self.open_positions(self.symbol)
        except Exception:
            return

        # Purge stale tickets from all tracking dicts
        open_tickets = {p.ticket for p in positions}
        self._hedged_tickets  -= self._hedged_tickets - open_tickets
        self._partial_done    -= self._partial_done - open_tickets
        for t in list(self._hedged_trail):
            if t not in open_tickets:
                del self._hedged_trail[t]
        for t in list(self._pair_map):
            if t not in open_tickets:
                self._pair_rev.pop(self._pair_map.pop(t), None)
        for t in list(self._pair_rev):
            if t not in open_tickets:
                self._pair_map.pop(self._pair_rev.pop(t), None)
        for t in list(self._zone_lots):
            if t not in open_tickets:
                if t in self._zone_tickets:
                    self._zone_tickets.remove(t)
                del self._zone_lots[t]

        # lock / ratio_hedge: close pair when combined P&L >= 0
        if self.flip_mode in ("lock", "ratio_hedge"):
            for orig_t, hedge_t in list(self._pair_map.items()):
                if hedge_t == -1:
                    continue   # placeholder — hedge not opened yet
                orig_p  = next((p for p in positions if p.ticket == orig_t), None)
                hedge_p = next((p for p in positions if p.ticket == hedge_t), None)
                if orig_p is None or hedge_p is None:
                    continue
                if orig_p.profit + hedge_p.profit >= 0:
                    pd_str = "buy" if orig_p.type == 0 else "sell"
                    self.log(
                        f"Pair exit: {pd_str.upper()} t={orig_t} {orig_p.profit:+.2f} + "
                        f"hedge t={hedge_t} {hedge_p.profit:+.2f} = combined >= 0"
                    )
                    for cp in (orig_p, hedge_p):
                        try:
                            self.conn.close_position(cp)
                            self._breakeven_done.discard(cp.ticket)
                        except Exception as e:
                            self.log(f"Pair close error: {e}")
                    self._pair_rev.pop(hedge_t, None)
                    del self._pair_map[orig_t]

        # zone_recovery: check combined P&L + price-triggered new layers
        if self.flip_mode == "zone_recovery" and len(self._zone_tickets) > 1:
            zone_pos = [p for p in positions if p.ticket in set(self._zone_tickets)]
            combined = sum(p.profit for p in zone_pos)
            if combined >= 0:
                self.log(f"Zone exit: combined P&L {combined:+.2f} >= 0 — closing all layers")
                for zp in zone_pos:
                    try:
                        self.conn.close_position(zp)
                        self._breakeven_done.discard(zp.ticket)
                    except Exception as e:
                        self.log(f"Zone close error: {e}")
                self._zone_tickets.clear(); self._zone_lots.clear()

        if self.flip_mode == "zone_recovery" and self._zone_tickets:
            latest_t = self._zone_tickets[-1]
            latest_p = next((p for p in positions if p.ticket == latest_t), None)
            if latest_p and len(self._zone_tickets) < self.MAX_ZONE_LAYERS:
                if latest_p.type == 0:
                    latest_pips = (latest_p.price_current - latest_p.price_open) / self._pip_size
                else:
                    latest_pips = (latest_p.price_open - latest_p.price_current) / self._pip_size
                if latest_pips <= -self.zone_pips:
                    new_dir = "sell" if latest_p.type == 0 else "buy"
                    new_mult = 2.0 ** len(self._zone_tickets)
                    base_lot = list(self._zone_lots.values())[0] if self._zone_lots else 0.01
                    new_lot = round(base_lot * new_mult / self._zone_lots.get(self._zone_tickets[0], base_lot), 2)
                    # just use base * mult relative to layer-1 lot
                    layer1_lot = self._zone_lots.get(self._zone_tickets[0], base_lot)
                    new_lot = max(round(layer1_lot * new_mult, 2), 0.01)
                    self.log(
                        f"Zone layer {len(self._zone_tickets)+1}: {new_dir.upper()} "
                        f"lot={new_lot:.2f} (latest layer lost {latest_pips:+.1f}p)"
                    )
                    try:
                        tick = self.conn.get_tick(self.symbol)
                        if tick:
                            price = tick.ask if new_dir == "buy" else tick.bid
                            sl_p  = price - self.sl_pips * self._pip_size if new_dir == "buy" \
                                    else price + self.sl_pips * self._pip_size
                            tp_p  = price + self.tp_pips * self._pip_size if new_dir == "buy" \
                                    else price - self.tp_pips * self._pip_size
                            if new_dir == "buy":
                                res = self.buy(self.symbol, new_lot, sl=round(sl_p,5), tp=round(tp_p,5))
                            else:
                                res = self.sell(self.symbol, new_lot, sl=round(sl_p,5), tp=round(tp_p,5))
                            new_ticket = res.get("order")
                            if new_ticket:
                                self._zone_tickets.append(new_ticket)
                                self._zone_lots[new_ticket] = new_lot
                    except Exception as e:
                        self.log(f"Zone new layer error: {e}")

        # trailing_hedge: update peak profit and close when trail is hit
        for pos in positions:
            if pos.ticket not in self._hedged_trail:
                continue
            if pos.direction == 0:  # MT5: 0=buy, 1=sell
                current_pips = (pos.price_current - pos.price_open) / self._pip_size
            else:
                current_pips = (pos.price_open - pos.price_current) / self._pip_size
            peak = max(self._hedged_trail[pos.ticket], current_pips)
            self._hedged_trail[pos.ticket] = peak
            if peak > 0 and current_pips <= peak - self.trail_pips:
                pos_dir = "buy" if pos.direction == 0 else "sell"
                self.log(
                    f"Trail stop: {pos_dir.upper()} ticket={pos.ticket}  "
                    f"peak={peak:+.1f}p  now={current_pips:+.1f}p  "
                    f"trail={self.trail_pips}p — closing"
                )
                try:
                    self.conn.close_position(pos)
                    del self._hedged_trail[pos.ticket]
                    self._breakeven_done.discard(pos.ticket)
                except Exception as e:
                    self.log(f"Trail close error: {e}")

        # Close hedged losers at first profit
        for pos in positions:
            if pos.ticket not in self._hedged_tickets:
                continue
            if pos.profit > 0:
                pos_dir = "buy" if pos.type == 0 else "sell"
                self.log(
                    f"Hedged loser {pos_dir.upper()} ticket={pos.ticket} "
                    f"now at {pos.profit:+.2f} USD — closing at first profit"
                )
                try:
                    self.conn.close_position(pos)
                    self._hedged_tickets.discard(pos.ticket)
                    self._breakeven_done.discard(pos.ticket)
                except Exception as e:
                    self.log(f"Hedge close error: {e}")

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
                    new_sl = round(entry + BREAKEVEN_BUFFER_PIPS * self._pip_size, 5)
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
                    new_sl = round(entry - BREAKEVEN_BUFFER_PIPS * self._pip_size, 5)
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
    p.add_argument("--dry-run", action="store_true",
                   help="Print signals and sizing without sending any orders")
    p.add_argument("--symbol", default=_DEFAULT_SYMBOL,
                   help="MT5 symbol to trade (default: EURUSD)")
    p.add_argument("--model-dir", default=None,
                   help="Path to model artifacts dir (overrides config)")
    p.add_argument("--flip-mode", default="always",
                   choices=["always", "hedge_loss", "hedge_exit", "trailing_hedge",
                            "lock", "ratio_hedge", "partial_close", "zone_recovery",
                            "candle_predictor", "candle_trail"],
                   help="Flip mode when an opposite signal arrives on an open position.")
    p.add_argument("--trail-pips", type=float, default=10.0,
                   help="trailing_hedge: pips behind peak before close (default 10)")
    p.add_argument("--hedge-ratio", type=float, default=2.0,
                   help="ratio_hedge: hedge lot multiplier (default 2.0)")
    p.add_argument("--zone-pips", type=float, default=30.0,
                   help="zone_recovery: pip gap before new zone layer (default 30)")
    p.add_argument("--candle-model-dir", default=None,
                   help="Path to candle predictor model dir (required with --flip-mode candle_predictor / candle_trail)")
    p.add_argument("--magic", type=int, default=None,
                   help="Override magic number (default: config.yaml trading.magic_number)")
    p.add_argument("--trail-activation-pips", type=float, default=15.0,
                   help="candle_trail: pips in profit before trailing SL activates (default 15)")
    p.add_argument("--trail-pips-behind", type=float, default=10.0,
                   help="candle_trail: trailing SL distance behind peak (default 10)")
    p.add_argument("--trail-max-bars-low", type=int, default=1,
                   help="candle_trail: max bars for conf<0.70 (default 1)")
    p.add_argument("--trail-max-bars-med", type=int, default=2,
                   help="candle_trail: max bars for conf 0.70-0.80 (default 2)")
    p.add_argument("--trail-max-bars-high", type=int, default=4,
                   help="candle_trail: max bars for conf>=0.80 (default 4)")
    args = p.parse_args()
    PipelineBot(
        dry_run               = args.dry_run,
        symbol                = args.symbol,
        model_dir             = args.model_dir,
        flip_mode             = args.flip_mode,
        trail_pips            = args.trail_pips,
        hedge_ratio           = args.hedge_ratio,
        zone_pips             = args.zone_pips,
        candle_model_dir      = args.candle_model_dir,
        magic                 = args.magic,
        trail_activation_pips = args.trail_activation_pips,
        trail_pips_behind     = args.trail_pips_behind,
        trail_max_bars_low    = args.trail_max_bars_low,
        trail_max_bars_med    = args.trail_max_bars_med,
        trail_max_bars_high   = args.trail_max_bars_high,
    ).run()


if __name__ == "__main__":
    main()
