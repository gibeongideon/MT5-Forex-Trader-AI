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

  profit_retrace  (--profit-retrace-activation 120 --profit-retrace-ratio 0.5)
    Every open bot trade is monitored on the bot tick interval.  Once a trade's
    floating profit reaches the activation amount in account currency, its best
    profit is tracked.  If profit later falls back to ratio × peak, close it.
    Example: after +120, close at +60 on pullback; after +800, close at +400.
    Use --tick-interval 5 for close monitoring every 5 seconds.

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
Hybrid model  (--candle-feature-dir)  ★ NEW — v2 champion
──────────────────────────────────────────────────────────────────────────────
A distinct model architecture that stacks two independently-trained models:

  Layer 1 — CatBoost candle predictor (1-bar specialist)
    Predicts direction of bar i+1 using data from bars 0..i only.
    Output: P_buy, P_hold, P_sell probabilities.

  Layer 2 — XGBoost + enc8 pipeline (4-bar generalist)
    Standard champion model, but retrained with two extra input features:
      candle_p_buy   — Layer 1 P_buy for the current bar
      candle_p_sell  — Layer 1 P_sell for the current bar
    The XGBoost learns to weight the short-term candle signal alongside its
    own 32 technical indicators + 8 encoder latent dims.  Total: 42 features.

  Architecture:
    candle_pipe.predict(ohlcv)         → {P_buy, P_sell, ...}
    pipe.predict(ohlcv, candle_signal) → injects P_buy/P_sell → 42-feat XGBoost → signal

  Leakage-free training:
    Candle predictions used for retraining are generated from 13 walk-forward
    fold models (wf_cache_candle2_EURUSD/), each predicting ONLY its OOS
    window — no bar ever trained and predicted by the same fold model.
    Stored in data/features/candle_signal_{SYMBOL}.parquet.

  Artifacts:
    data/models/pipeline_EURUSD_v2/  (42-feature XGBoost + enc8)
    data/models/pipeline_USDJPY_v2/  (42-feature XGBoost + enc8)
    data/models/candle_EURUSD/       (CatBoost 1-bar, unchanged)
    data/models/candle_USDJPY/       (CatBoost 1-bar, unchanged)

  WF OOS results vs baseline (60k M15 bars, expanding 180d/30d):
    Symbol   Model         Feats  Sharpe   MaxDD   Return  Trades
    EURUSD   Baseline v1     40   +1.35    16.4%   +49%     648
    EURUSD   Hybrid v2  ★    42   +3.01    11.4%   +93%     540   (+1.66 Sharpe, -5pp DD)
    USDJPY   Baseline v1     40   +3.24    24.3%  +551%    1920
    USDJPY   Hybrid v2  ★    42   +4.27    19.6% +1419%    1395   (+1.03 Sharpe, -4.7pp DD)

  Fallback: if --candle-feature-dir is not set, predict() injects 0.5 neutral
  for candle_p_buy/sell so old v1 model files still work unchanged.

──────────────────────────────────────────────────────────────────────────────
Dedicated candle-model modes  (require --candle-model-dir)
──────────────────────────────────────────────────────────────────────────────
These two modes use the CatBoost 1-bar model standalone (no XGBoost pipeline).
They completely bypass the standard 8-step pipeline logic and have their own
independent trade lifecycle.

  candle_predictor
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
    Tuned params (grid search, 160 combos):
      EURUSD: --trail-activation-pips 12 --trail-pips-behind 5
              --trail-max-bars-low 1 --trail-max-bars-med 2 --trail-max-bars-high 4
      USDJPY: --trail-activation-pips 15 --trail-pips-behind 5
              --trail-max-bars-low 1 --trail-max-bars-med 3 --trail-max-bars-high 4
    Backtest (2.4 yr OOS, tuned):
      EURUSD  Sharpe +19.4  Win 91.3%  MaxDD 3.1%   ~1,922 trades  ← -53% DD
      USDJPY  Sharpe +24.5  Win 88.4%  MaxDD 4.8%   ~3,778 trades  ← -56% DD

──────────────────────────────────────────────────────────────────────────────
Full performance summary
──────────────────────────────────────────────────────────────────────────────

  ── Standard pipeline modes (XGBoost v1 + enc8, 60k bars in-sample) ────────
  Mode              Trades   Win%   Sharpe   MaxDD   Notes
  always             7,636   61%    +12.97   4.9%
  hedge_loss         4,013   71%     +7.47   4.9%    fewest trades, highest win rate
  hedge_exit         6,033   61%    +10.12   5.3%
  trailing_hedge     5,635   69%     +9.23   3.2%    lowest DD among standard modes
  lock               TBD — run scripts/backtest_flip_modes.py
  ratio_hedge        TBD
  partial_close      TBD
  zone_recovery      TBD

  ── Hybrid v2 model (XGBoost + enc8 + candle features, WF OOS) ─────────────
  Symbol   Flip mode       Trades   Sharpe   MaxDD   Notes
  EURUSD   any             540      +3.01    11.4%   42 features, candle injected
  USDJPY   any             1,395    +4.27    19.6%   42 features, candle injected

  ── Candle standalone modes (CatBoost 1-bar, 60k M15 bars, OOS 2.4 yr) ─────
  Mode              Trades   Win%   Sharpe   MaxDD   Notes
  candle_predictor   2,212   87%    +20.1    6.7%    EURUSD — 1-bar force-close
  candle_predictor   4,131   81%    +25.6   10.9%    USDJPY
  candle_trail       1,922   91%    +19.4    3.1%    EURUSD — tuned trailing SL
  candle_trail       3,778   88%    +24.5    4.8%    USDJPY ← recommended for live

──────────────────────────────────────────────────────────────────────────────

Usage:
  # 1. Generate OOS candle signal parquet (one-time, ~2 min)
  conda run -n envmt5 python scripts/build_candle_features.py

  # 2. Retrain hybrid v2 champion (includes candle features automatically)
  conda run -n envmt5 python scripts/retrain_champion.py \\
      --data data/EURUSD_M15.csv --out data/models/pipeline_EURUSD_v2

  # 3. Run hybrid v2 bot  ★ recommended
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --model-dir data/models/pipeline_EURUSD_v2 \\
      --candle-feature-dir data/models/candle_EURUSD \\
      --flip-mode trailing_hedge

  # 4. Candle standalone — 1-bar force-close
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --candle-model-dir data/models/candle_EURUSD --flip-mode candle_predictor

  # 5. Candle standalone — trailing SL (tuned params)
  conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \\
      --candle-model-dir data/models/candle_EURUSD --flip-mode candle_trail \\
      --trail-activation-pips 12 --trail-pips-behind 5 \\
      --trail-max-bars-low 1 --trail-max-bars-med 2 --trail-max-bars-high 4

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

    MAX_POSITIONS = 1   # hard cap per bot instance — one position at a time
    MAX_SAME_DIRECTION_POSITIONS = 1   # account-level cap per symbol/direction

    MAX_ZONE_LAYERS = 4

    def __init__(self, dry_run: bool = False, symbol: str = _DEFAULT_SYMBOL,
                 model_dir: str | None = None, flip_mode: str = "always",
                 trail_pips: float = 10.0, hedge_ratio: float = 2.0,
                 zone_pips: float = 30.0,
                 candle_model_dir: str | None = None,
                 candle_feature_dir: str | None = None,
                 magic: int | None = None,
                 trail_activation_pips: float = 15.0,
                 trail_pips_behind: float = 10.0,
                 trail_max_bars_low: int = 1,
                 trail_max_bars_med: int = 2,
                 trail_max_bars_high: int = 4,
                 max_lot: float | None = None,
                 tick_interval: float | None = None,
                 profit_retrace_activation: float = 120.0,
                 profit_retrace_ratio: float = 0.5,
                 platform: str | None = None,
                 journal_run_id: str = ""):
        super().__init__(
            name=f"PipelineBot-{symbol}",
            tick_interval=float(tick_interval or 60.0),
            platform=platform,
        )
        if magic is not None:
            self.magic = magic  # override config.yaml magic_number
        self.dry_run     = dry_run
        self.symbol      = symbol
        self.journal_run_id = journal_run_id
        self._symbol_resolved = False   # broker-suffix auto-resolution (e.g. EURUSD→EURUSD.Z)
        # realistic hard cap on lot size (risk-% sizing on a large balance can balloon);
        # from config trading.max_lot, default 0.50 lots. Override per-bot via --max-lot.
        self.max_lot = float(
            (max_lot if max_lot is not None
             else self.config.get("trading", {}).get("max_lot", 0.50))
        )
        self.flip_mode   = flip_mode
        self.trail_pips  = trail_pips   # trailing_hedge
        self.hedge_ratio = hedge_ratio  # ratio_hedge: hedge lot multiplier
        self.zone_pips   = zone_pips    # zone_recovery: pip gap before next layer
        if not 0.0 < profit_retrace_ratio < 1.0:
            raise ValueError("--profit-retrace-ratio must be between 0 and 1")
        self.profit_retrace_activation = profit_retrace_activation
        self.profit_retrace_ratio = profit_retrace_ratio

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

        # Load candle pipe: for flip modes (required) or for feature injection (optional)
        _candle_dir = candle_model_dir or candle_feature_dir
        if _candle_dir is not None:
            self.candle_pipe = PredictorPipeline.from_config()
            self.candle_pipe.load(_candle_dir)
            candle_meta = Path(_candle_dir) / "pair_meta.json"
            if candle_meta.exists():
                cm = json.loads(candle_meta.read_text())
                self._candle_sl_pips = float(cm.get("sl_pips", 15.0))
                self._candle_tp_pips = float(cm.get("tp_pips", 20.0))
        self._candle_feature_inject = (
            candle_feature_dir is not None
            and flip_mode not in ("candle_predictor", "candle_trail")
        )

        self._last_bar: pd.Timestamp | None = None
        self._breakeven_done: set[int] = set()
        # hedge_exit
        self._hedged_tickets: set[int] = set()
        # hedge_exit RECOVERY SET — tickets of trades kept open because they were losing
        # at a signal flip. Closed at break-even (>=0) regardless of direction/opposite-pair.
        # PERSISTED to disk so it survives bot restarts (in-memory sets are wiped on restart).
        self._recovery_path = ROOT / "data" / f"recovery_{self.magic}.json"
        self._recovery_tickets: set[int] = self._load_recovery()
        # trailing_hedge: ticket → peak profit pips
        self._hedged_trail: dict[int, float] = {}
        # profit_retrace: ticket → peak floating profit in account currency
        self._profit_retrace_peaks: dict[int, float] = {}
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
        elif self.flip_mode == "profit_retrace":
            flip_detail += (
                f"(act={self.profit_retrace_activation:.2f}, "
                f"exit={self.profit_retrace_ratio:.0%} peak)"
            )
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
            f"tick={self.tick_interval:.0f}s  "
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

    def risk_sized_lot(self, *args, **kwargs):
        """Wrap the base risk sizer with a realistic hard cap (self.max_lot).
        Risk-% sizing on a large balance can balloon lots; cap covers all
        trade paths (main / candle_predictor / candle_trail)."""
        lot, eff_sl = super().risk_sized_lot(*args, **kwargs)
        if self.max_lot and lot > self.max_lot:
            self.log(f"Lot {lot:.2f} capped to max_lot {self.max_lot:.2f}")
            lot = self.max_lot
        return lot, eff_sl

    def _ensure_symbol_resolved(self) -> None:
        """Map a disabled/display-only symbol to the broker's tradable variant.
        Many brokers (e.g. HFM) expose suffixed instruments — plain 'EURUSD' has
        trade_mode=0 (disabled) while 'EURUSD.Z' is the tradable one. Resolve once,
        broker-agnostically: pick the same-base symbol with full trading (mode 4),
        preferring visible then shortest name. No-op if the symbol already trades."""
        if self._symbol_resolved:
            return
        self._symbol_resolved = True
        mt5 = getattr(self.conn, "_mt5", None)
        try:
            info = self.conn.symbol_info(self.symbol)
        except Exception:
            info = None
        if info is not None and getattr(info, "trade_mode", 0) == 4:
            try: mt5.symbol_select(self.symbol, True)
            except Exception: pass
            return
        base = self.symbol[:6].upper()
        try:
            allsyms = mt5.symbols_get() or []
        except Exception:
            allsyms = []
        cands = [s for s in allsyms
                 if s.name[:6].upper() == base and getattr(s, "trade_mode", 0) == 4]
        cands.sort(key=lambda s: (not getattr(s, "visible", False), len(s.name)))
        if cands:
            resolved = cands[0].name
            self.log(f"Symbol '{self.symbol}' not tradable "
                     f"(trade_mode={getattr(info,'trade_mode',None)}) — "
                     f"switching to broker symbol '{resolved}'")
            self.symbol = resolved
            try: mt5.symbol_select(resolved, True)
            except Exception: pass
        else:
            self.log(f"WARNING: no tradable variant found for base '{base}' — "
                     f"orders may fail (retcode 10017)")

    def on_tick(self) -> None:
        self._ensure_symbol_resolved()
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
            candle_sig = None
            if self._candle_feature_inject and self.candle_pipe is not None:
                try:
                    candle_sig = self.candle_pipe.predict(ohlcv)
                except Exception:
                    pass
            sig = self.pipe.predict(ohlcv, candle_signal=candle_sig)
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

        # LATEST-SIGNAL PRIORITY: handle opposite-direction positions FIRST (close the
        # profitable ones, keep/hedge the losing ones) BEFORE deciding whether to open.
        # Note: do NOT early-return on a same-direction position — that previously
        # short-circuited the opposite-handling below, so a profitable opposite was left
        # open whenever a same-direction trade was also held. The post-close cap-check
        # (step 6b) still prevents opening a duplicate same-direction trade (no stacking).
        for pos in our_positions:
            pos_dir = "buy" if pos.type == 0 else "sell"
            if pos_dir == direction:
                continue   # same direction — not an opposite to close; handled at cap-check

            in_loss = pos.profit <= 0

            if self.flip_mode in ("hedge_loss", "hedge_exit", "trailing_hedge", "profit_retrace") and in_loss:
                self.log(
                    f"Hedge mode: {pos_dir.upper()} ticket={pos.ticket} at "
                    f"{pos.profit:+.2f} USD — keeping, adding {direction.upper()} hedge"
                )
                if self.flip_mode in ("hedge_exit", "profit_retrace"):
                    self._hedged_tickets.add(pos.ticket)
                    self._mark_recovery(pos.ticket)   # persisted → closed at >=0 even if lone/after restart
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
                       "lock", "ratio_hedge", "partial_close", "zone_recovery",
                       "profit_retrace")
        if not self.dry_run:
            if self.flip_mode == "profit_retrace":
                same_count = self._same_direction_count(direction)
                if same_count >= self.MAX_SAME_DIRECTION_POSITIONS:
                    self.log(
                        f"Same-direction cap ({self.MAX_SAME_DIRECTION_POSITIONS}) "
                        f"already reached for {direction.upper()} — skip"
                    )
                    return
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
        sa_mult = min(sa_mult, 1.0)   # structural multiplier may only DE-RISK, never inflate
        if sa_mult != 1.0:
            lot_before = lot
            lot = round(lot * sa_mult, 2)
            lot = max(lot, self.conn.symbol_info(self.symbol).volume_min)
            lot = min(lot, self.max_lot)
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

        try:
            self.journal.record({
                "bot": self.name,
                "symbol": self.symbol,
                "direction": direction,
                "entry_time": str(bar_time),
                "entry_price": price,
                "exit_time": None,
                "exit_price": None,
                "pnl_pips": 0.0,
                "pnl_dollars": 0.0,
                "model": "candle_trail",
                "confidence": confidence,
                "entry_reason": f"trail:{direction}",
                "exit_reason": "dry_run_open" if self.dry_run else "open",
                "volume": lot,
                "sl_pips": eff_sl,
                "tp_pips": self._candle_tp_pips,
                "magic": self.magic,
                "run_id": self.journal_run_id,
                "dry_run": bool(self.dry_run),
            })
        except Exception:
            pass

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

    def _load_recovery(self) -> set:
        """Load the persisted recovery-trade ticket set (restart-proof)."""
        import json
        try:
            if self._recovery_path.exists():
                return set(json.loads(self._recovery_path.read_text()))
        except Exception as e:
            self.log(f"recovery load error: {e}")
        return set()

    def _save_recovery(self) -> None:
        import json
        try:
            self._recovery_path.write_text(json.dumps(sorted(self._recovery_tickets)))
        except Exception as e:
            self.log(f"recovery save error: {e}")

    def _mark_recovery(self, ticket: int) -> None:
        """Tag a kept-losing trade as a recovery trade (persisted)."""
        if ticket not in self._recovery_tickets:
            self._recovery_tickets.add(ticket)
            self._save_recovery()

    def _symbol_positions(self):
        """All open positions for this resolved symbol, across bot magic numbers."""
        try:
            return self.conn.get_positions(symbol=self.symbol)
        except Exception:
            return []

    def _same_direction_count(self, direction: str) -> int:
        typ = 0 if direction == "buy" else 1
        return sum(1 for p in self._symbol_positions() if p.type == typ)

    def _hedge_loser_to_close(self, positions):
        """Opposite-pair fallback for legacy/untracked recovery trades (no persisted state).

        If we hold an opposite-direction pair (a buy AND a sell under our magic), the
        OLDER leg — the smaller ticket, i.e. opened first — is the kept hedge loser.
        Return its ticket once it has recovered to break-even/positive (profit >= 0).
        Returns None otherwise. Pure/stateless."""
        ours = [p for p in positions if p.magic == self.magic]
        buys = [p for p in ours if p.type == 0]
        sells = [p for p in ours if p.type == 1]
        if not (buys and sells):
            return None
        older = min(ours, key=lambda p: p.ticket)   # monotonic ticket → earliest open
        return older.ticket if older.profit >= 0 else None

    def _recovery_closeable(self, positions) -> set:
        """Tickets to close NOW at break-even: any recovery trade (persisted set, or the
        opposite-pair fallback) that is open under our magic and has profit >= 0.
        Pure given self._recovery_tickets — unit-testable."""
        by_ticket = {p.ticket: p for p in positions if p.magic == self.magic}
        cands = set(self._recovery_tickets)
        fb = self._hedge_loser_to_close(positions)
        if fb is not None:
            cands.add(fb)
        return {t for t in cands
                if t in by_ticket and by_ticket[t].profit >= 0}

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
        for t in list(self._profit_retrace_peaks):
            if t not in open_tickets:
                del self._profit_retrace_peaks[t]
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
            if pos.type == 0:  # MT5: 0=buy, 1=sell
                current_pips = (pos.price_current - pos.price_open) / self._pip_size
            else:
                current_pips = (pos.price_open - pos.price_current) / self._pip_size
            peak = max(self._hedged_trail[pos.ticket], current_pips)
            self._hedged_trail[pos.ticket] = peak
            if peak > 0 and current_pips <= peak - self.trail_pips:
                pos_dir = "buy" if pos.type == 0 else "sell"
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

        # hedge_exit RECOVERY CLOSE — close any kept-losing "recovery trade" the moment
        # its P&L recovers to >=0, regardless of direction or whether an opposite still
        # exists. Authoritative source = the PERSISTED recovery set (survives restarts);
        # plus an opposite-pair fallback for legacy/untracked positions.
        if self.flip_mode in ("hedge_exit", "profit_retrace"):
            # 1) purge recovery tickets that are no longer open
            stale = self._recovery_tickets - open_tickets
            if stale:
                self._recovery_tickets -= stale
                self._save_recovery()
            # 2) close every recovery trade now at break-even (>=0)
            for t in list(self._recovery_closeable(positions)):
                pos = next((p for p in positions if p.ticket == t), None)
                if pos is None:
                    continue
                pos_dir = "buy" if pos.type == 0 else "sell"
                self.log(
                    f"Recovery trade {pos_dir.upper()} ticket={t} now at "
                    f"{pos.profit:+.2f} USD (>=0) — closing at break-even"
                )
                try:
                    self.conn.close_position(pos)
                    self._hedged_tickets.discard(t)
                    self._breakeven_done.discard(t)
                    if t in self._recovery_tickets:
                        self._recovery_tickets.discard(t)
                        self._save_recovery()
                except Exception as e:
                    self.log(f"Recovery close error: {e}")

        # profit_retrace: once a trade reaches the activation profit, track its
        # best floating profit and close when it gives back the configured share.
        if self.flip_mode == "profit_retrace":
            latest_ticket = max(
                (p.ticket for p in positions
                 if p.magic == self.magic and p.ticket not in self._recovery_tickets),
                default=None,
            )
            for pos in positions:
                if pos.magic != self.magic:
                    continue
                if pos.ticket != latest_ticket:
                    self._profit_retrace_peaks.pop(pos.ticket, None)
                    continue
                profit = float(pos.profit)
                peak = self._profit_retrace_peaks.get(pos.ticket)
                if peak is None:
                    if profit >= self.profit_retrace_activation:
                        self._profit_retrace_peaks[pos.ticket] = profit
                        pos_dir = "buy" if pos.type == 0 else "sell"
                        self.log(
                            f"Profit retrace armed: {pos_dir.upper()} ticket={pos.ticket}  "
                            f"peak={profit:+.2f}  "
                            f"exit_at={profit * self.profit_retrace_ratio:+.2f}"
                        )
                    continue

                if profit > peak:
                    peak = profit
                    self._profit_retrace_peaks[pos.ticket] = peak

                exit_profit = peak * self.profit_retrace_ratio
                if profit <= exit_profit:
                    pos_dir = "buy" if pos.type == 0 else "sell"
                    self.log(
                        f"Profit retrace exit: {pos_dir.upper()} ticket={pos.ticket}  "
                        f"peak={peak:+.2f}  now={profit:+.2f}  "
                        f"exit_at={exit_profit:+.2f} — closing"
                    )
                    try:
                        self.conn.close_position(pos)
                        self._profit_retrace_peaks.pop(pos.ticket, None)
                        self._breakeven_done.discard(pos.ticket)
                    except Exception as e:
                        self.log(f"Profit retrace close error: {e}")

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
                            "profit_retrace", "candle_predictor", "candle_trail"],
                   help="Flip mode when an opposite signal arrives on an open position.")
    p.add_argument("--trail-pips", type=float, default=10.0,
                   help="trailing_hedge: pips behind peak before close (default 10)")
    p.add_argument("--hedge-ratio", type=float, default=2.0,
                   help="ratio_hedge: hedge lot multiplier (default 2.0)")
    p.add_argument("--zone-pips", type=float, default=30.0,
                   help="zone_recovery: pip gap before new zone layer (default 30)")
    p.add_argument("--candle-model-dir", default=None,
                   help="Path to candle predictor model dir (required with --flip-mode candle_predictor / candle_trail)")
    p.add_argument("--candle-feature-dir", default=None,
                   help="Path to candle model dir for feature injection into standard pipeline (optional)")
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
    p.add_argument("--max-lot", type=float, default=None,
                   help="hard cap on lot size (default: config trading.max_lot or 0.50)")
    p.add_argument("--tick-interval", type=float, default=None,
                   help="seconds between live management checks (default 60; use 5 with profit_retrace)")
    p.add_argument("--profit-retrace-activation", type=float, default=120.0,
                   help="profit_retrace: arm retrace close once profit reaches this account-currency amount")
    p.add_argument("--profit-retrace-ratio", type=float, default=0.5,
                   help="profit_retrace: close when profit falls to this share of peak (default 0.5)")
    p.add_argument("--platform", default=None, choices=["mt5", "mt4"],
                   help="trading platform (default: config trading.platform or mt5)")
    p.add_argument("--journal-run-id", default="",
                   help="Run id recorded in live_trades.db for dry-run/paper reconciliation")
    args = p.parse_args()
    PipelineBot(
        platform              = args.platform,
        max_lot               = args.max_lot,
        dry_run               = args.dry_run,
        symbol                = args.symbol,
        model_dir             = args.model_dir,
        flip_mode             = args.flip_mode,
        trail_pips            = args.trail_pips,
        hedge_ratio           = args.hedge_ratio,
        zone_pips             = args.zone_pips,
        candle_model_dir      = args.candle_model_dir,
        candle_feature_dir    = args.candle_feature_dir,
        magic                 = args.magic,
        trail_activation_pips = args.trail_activation_pips,
        trail_pips_behind     = args.trail_pips_behind,
        trail_max_bars_low    = args.trail_max_bars_low,
        trail_max_bars_med    = args.trail_max_bars_med,
        trail_max_bars_high   = args.trail_max_bars_high,
        tick_interval         = args.tick_interval,
        profit_retrace_activation = args.profit_retrace_activation,
        profit_retrace_ratio      = args.profit_retrace_ratio,
        journal_run_id             = args.journal_run_id,
    ).run()


if __name__ == "__main__":
    main()
