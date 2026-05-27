"""
Random Baseline Bot — Phase 1.

Purpose: establish a benchmark with zero predictive edge. Any strategy in later
phases must beat this to demonstrate real signal.

Two modes:

  BACKTEST (no MT5 required):
    python src/bots/random_bot.py --backtest
    python src/bots/random_bot.py --backtest --data data/EURUSD_M15.csv --entry-prob 0.01 --seed 42

  LIVE (MT5 must be running):
    python src/bots/random_bot.py

Config section (config.yaml):
    random_bot:
      symbol: "EURUSD"
      timeframe: "M15"
      entry_prob: 0.02   # probability of entering a trade each tick
      sl_pips: 30
      tp_pips: 60
      seed: null         # set for reproducible backtest
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import performance_report
from src.core.trade_journal import TradeJournal


# ─── Shared trade record ──────────────────────────────────────────────────────

@dataclass
class Trade:
    direction:   str
    entry_time:  object
    entry_price: float
    sl:          float
    tp:          float
    exit_time:   object = None
    exit_price:  float  = 0.0
    pnl_pips:    float  = 0.0
    pnl_dollars: float  = 0.0
    exit_reason: str    = ""


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest_random(
    df: pd.DataFrame,
    entry_prob:      float = 0.02,
    sl_pips:         float = 30.0,
    tp_pips:         float = 60.0,
    pip_size:        float = 0.0001,
    initial_balance: float = 10_000.0,
    risk_pct:        float = 0.01,
    seed:            Optional[int] = None,
    journal:         Optional[TradeJournal] = None,
) -> tuple[list[Trade], pd.Series]:
    """
    Bar-by-bar random entry simulation.

    At each bar there is a configurable probability of opening a long or short.
    Only one trade is held at a time. SL and TP are fixed-pip.
    """
    rng = random.Random(seed)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df.index

    sl_pts = sl_pips * pip_size
    tp_pts = tp_pips * pip_size

    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    balance = initial_balance
    equity_curve: list[float] = []

    for i in range(len(df)):

        # ── Check SL / TP ────────────────────────────────────────────────
        if open_trade is not None:
            if open_trade.direction == "buy":
                if lows[i] <= open_trade.sl:
                    _close(open_trade, open_trade.sl, times[i], -sl_pips, "sl")
                elif highs[i] >= open_trade.tp:
                    _close(open_trade, open_trade.tp, times[i], tp_pips, "tp")
            else:
                if highs[i] >= open_trade.sl:
                    _close(open_trade, open_trade.sl, times[i], -sl_pips, "sl")
                elif lows[i] <= open_trade.tp:
                    _close(open_trade, open_trade.tp, times[i], tp_pips, "tp")

            if open_trade.exit_time is not None:
                pnl_dollars = (open_trade.pnl_pips / sl_pips) * (balance * risk_pct)
                open_trade.pnl_dollars = pnl_dollars
                balance += pnl_dollars
                _maybe_record(journal, open_trade, balance, risk_pct, sl_pips, tp_pips)
                trades.append(open_trade)
                open_trade = None

        # ── Random entry ─────────────────────────────────────────────────
        if open_trade is None and rng.random() < entry_prob:
            direction = rng.choice(["buy", "sell"])
            price = closes[i]
            if direction == "buy":
                sl = price - sl_pts
                tp = price + tp_pts
            else:
                sl = price + sl_pts
                tp = price - tp_pts
            open_trade = Trade(
                direction=direction,
                entry_time=times[i],
                entry_price=price,
                sl=sl,
                tp=tp,
            )

        equity_curve.append(balance)

    # ── Force-close any open trade at end ────────────────────────────────
    if open_trade is not None:
        price = closes[-1]
        if open_trade.direction == "buy":
            pips = (price - open_trade.entry_price) / pip_size
        else:
            pips = (open_trade.entry_price - price) / pip_size
        _close(open_trade, price, times[-1], pips, "end")
        pnl_dollars = (open_trade.pnl_pips / sl_pips) * (balance * risk_pct)
        open_trade.pnl_dollars = pnl_dollars
        balance += pnl_dollars
        _maybe_record(journal, open_trade, balance, risk_pct, sl_pips, tp_pips)
        trades.append(open_trade)

    equity = pd.Series(equity_curve, index=df.index)
    return trades, equity


def _close(t: Trade, price, time, pips, reason) -> None:
    t.exit_price  = price
    t.exit_time   = time
    t.pnl_pips    = pips
    t.exit_reason = reason


def _maybe_record(journal, t: Trade, balance, risk_pct, sl_pips, tp_pips) -> None:
    if journal is None:
        return
    journal.record({
        "bot":          "random_bot",
        "symbol":       "EURUSD",
        "direction":    t.direction,
        "entry_time":   str(t.entry_time),
        "entry_price":  t.entry_price,
        "exit_time":    str(t.exit_time),
        "exit_price":   t.exit_price,
        "pnl_pips":     t.pnl_pips,
        "pnl_dollars":  t.pnl_dollars,
        "model":        "random",
        "confidence":   0.5,
        "entry_reason": "random",
        "exit_reason":  t.exit_reason,
        "volume":       0.0,
        "sl_pips":      sl_pips,
        "tp_pips":      tp_pips,
    })


# ─── Live Bot ─────────────────────────────────────────────────────────────────

def _make_live_bot():
    from src.core.bot_base import BotBase

    class RandomBot(BotBase):
        """
        Live random-entry bot. Inherits all lifecycle management from BotBase.
        Entry probability and SL/TP come from config.yaml → random_bot section.
        """

        def __init__(self):
            super().__init__(name="RandomBot", tick_interval=15.0)
            cfg = self.config.get("random_bot", {})
            self.symbol     = cfg.get("symbol", "EURUSD")
            self.timeframe  = cfg.get("timeframe", "M15")
            self.entry_prob = cfg.get("entry_prob", 0.02)
            self.sl_pips    = cfg.get("sl_pips", 30.0)
            self.tp_pips    = cfg.get("tp_pips", 60.0)
            seed = cfg.get("seed")
            self._rng = random.Random(seed)
            self._journal = TradeJournal()

        def on_start(self) -> None:
            self.log(f"Symbol={self.symbol}  entry_prob={self.entry_prob:.1%}  "
                     f"SL={self.sl_pips}p  TP={self.tp_pips}p")

        def on_tick(self) -> None:
            if self.open_count(self.symbol) > 0:
                return
            if self._rng.random() >= self.entry_prob:
                return
            if self.open_count() >= self.max_open_trades:
                return

            direction = self._rng.choice(["buy", "sell"])
            info = self.conn.symbol_info(self.symbol)
            pip  = info.point * 10
            tick = self.conn.get_tick(self.symbol)

            if direction == "buy":
                entry = tick.ask
                sl = entry - self.sl_pips * pip
                tp = entry + self.tp_pips * pip
                self.buy(self.symbol, self.calc_lot(self.symbol, self.sl_pips),
                         sl=sl, tp=tp, comment="random")
            else:
                entry = tick.bid
                sl = entry + self.sl_pips * pip
                tp = entry - self.tp_pips * pip
                self.sell(self.symbol, self.calc_lot(self.symbol, self.sl_pips),
                          sl=sl, tp=tp, comment="random")

            self.log(f"Random {direction.upper()}  sl={sl:.5f}  tp={tp:.5f}")

    return RandomBot


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random Baseline Bot")
    p.add_argument("--backtest",   action="store_true",  help="Run in backtest mode (no MT5 needed)")
    p.add_argument("--data",       default="data/EURUSD_M15.csv")
    p.add_argument("--entry-prob", type=float, default=0.02,  help="Probability of entering each bar")
    p.add_argument("--sl",         type=float, default=30.0)
    p.add_argument("--tp",         type=float, default=60.0)
    p.add_argument("--balance",    type=float, default=10_000.0)
    p.add_argument("--risk",       type=float, default=0.01)
    p.add_argument("--pip-size",   type=float, default=0.0001)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save-journal", action="store_true", help="Write trades to data/trades.db")
    args = p.parse_args()

    if args.backtest:
        csv = Path(args.data)
        if not csv.exists():
            print(f"ERROR: {csv} not found.  Run: python scripts/download_data.py")
            sys.exit(1)

        print(f"Loading {csv}...")
        df = pd.read_csv(csv, index_col="time", parse_dates=True)
        print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

        journal = TradeJournal() if args.save_journal else None
        trades, equity = backtest_random(
            df,
            entry_prob=args.entry_prob,
            sl_pips=args.sl,
            tp_pips=args.tp,
            pip_size=args.pip_size,
            initial_balance=args.balance,
            risk_pct=args.risk,
            seed=args.seed,
            journal=journal,
        )

        symbol = csv.stem.split("_")[0]
        tf = csv.stem.split("_")[1] if "_" in csv.stem else "?"
        performance_report(
            trades, equity, args.balance,
            title="RANDOM BASELINE BACKTEST",
            extra_params={
                "Symbol":     f"{symbol} {tf}",
                "Entry prob": f"{args.entry_prob:.1%} per bar",
                "SL / TP":    f"{args.sl}p / {args.tp}p",
                "Seed":       args.seed,
            },
        )
        if args.save_journal:
            print(f"Trades saved to data/trades.db")
    else:
        RandomBot = _make_live_bot()
        bot = RandomBot()
        bot.run()
