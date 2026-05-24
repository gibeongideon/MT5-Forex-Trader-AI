"""
Rule-Based Signal Bot — Phase 2.

Uses the composable indicator library + rule engine to generate probability
signals. Only trades when P_buy or P_sell exceeds the confidence threshold.
Outputs [P_buy, P_hold, P_sell] — same interface future ML models will use.

Two modes:

  BACKTEST:
    python src/rule_bot.py --backtest
    python src/rule_bot.py --backtest --data data/EURUSD_M15.csv --threshold 0.55

  LIVE (MT5 required):
    python src/rule_bot.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import compute, sma, ema, rsi, macd, bollinger_bands, bollinger_pct_b, atr
from src.metrics import performance_report
from src.rule_engine import (
    SignalCombiner,
    ma_crossover_rule,
    rsi_rule,
    macd_rule,
    bb_reversion_rule,
    price_vs_ma_rule,
)
from src.trade_journal import TradeJournal


# ─── Build default combiner ───────────────────────────────────────────────────

def build_default_combiner(
    fast_ma:        int   = 9,
    slow_ma:        int   = 21,
    rsi_period:     int   = 14,
    bb_period:      int   = 20,
    threshold:      float = 0.55,
    weight_ma:      float = 2.0,
    weight_rsi:     float = 1.5,
    weight_bb:      float = 1.0,
    weight_macd:    float = 1.0,
) -> SignalCombiner:
    combiner = SignalCombiner(threshold=threshold)
    combiner.add(ma_crossover_rule(f"sma_{fast_ma}", f"sma_{slow_ma}"), weight=weight_ma,  name="ma_cross")
    combiner.add(rsi_rule(f"rsi_{rsi_period}"),                          weight=weight_rsi, name="rsi")
    combiner.add(bb_reversion_rule("bb_pct"),                            weight=weight_bb,  name="bb_rev")
    combiner.add(macd_rule("macd_line", "macd_sig"),                     weight=weight_macd, name="macd")
    return combiner


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fast   = cfg.get("fast_ma", 9)
    slow   = cfg.get("slow_ma", 21)
    rsi_p  = cfg.get("rsi_period", 14)
    bb_p   = cfg.get("bb_period", 20)

    return compute(df, [
        (f"sma_{fast}",   sma,              {"period": fast}),
        (f"sma_{slow}",   sma,              {"period": slow}),
        (f"rsi_{rsi_p}",  rsi,              {"period": rsi_p}),
        ("bb_pct",        bollinger_pct_b,  {"period": bb_p}),
        (("macd_line", "macd_sig", "macd_hist"), macd, {}),
        ("atr_14",        atr,              {"period": 14}),
    ])


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest_rules(
    df:              pd.DataFrame,
    cfg:             dict,
    sl_pips:         float = 30.0,
    tp_pips:         float = 60.0,
    pip_size:        float = 0.0001,
    initial_balance: float = 10_000.0,
    risk_pct:        float = 0.01,
    journal:         Optional[TradeJournal] = None,
) -> tuple[list[dict], pd.Series]:

    threshold = cfg.get("threshold", 0.55)
    df = build_features(df, cfg)
    combiner = build_default_combiner(
        fast_ma=cfg.get("fast_ma", 9),
        slow_ma=cfg.get("slow_ma", 21),
        rsi_period=cfg.get("rsi_period", 14),
        bb_period=cfg.get("bb_period", 20),
        threshold=threshold,
        weight_ma=cfg.get("weight_ma_cross", 2.0),
        weight_rsi=cfg.get("weight_rsi", 1.5),
        weight_bb=cfg.get("weight_bb", 1.0),
        weight_macd=cfg.get("weight_macd", 1.0),
    )

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df.index
    sl_pts = sl_pips * pip_size
    tp_pts = tp_pips * pip_size

    # need enough history for indicators to warm up
    warmup = cfg.get("slow_ma", 21) + 10

    trades: list[dict] = []
    open_trade: Optional[dict] = None
    balance = initial_balance
    equity_curve: list[float] = []

    for i in range(len(df)):
        # ── SL / TP check ─────────────────────────────────────────────────
        if open_trade is not None:
            direction = open_trade["direction"]
            if direction == "buy":
                if lows[i] <= open_trade["sl"]:
                    _close_rule_trade(open_trade, open_trade["sl"], times[i], -sl_pips, "sl")
                elif highs[i] >= open_trade["tp"]:
                    _close_rule_trade(open_trade, open_trade["tp"], times[i], tp_pips, "tp")
            else:
                if highs[i] >= open_trade["sl"]:
                    _close_rule_trade(open_trade, open_trade["sl"], times[i], -sl_pips, "sl")
                elif lows[i] <= open_trade["tp"]:
                    _close_rule_trade(open_trade, open_trade["tp"], times[i], tp_pips, "tp")

            if open_trade.get("exit_time") is not None:
                pnl_dollars = (open_trade["pnl_pips"] / sl_pips) * (balance * risk_pct)
                open_trade["pnl_dollars"] = pnl_dollars
                balance += pnl_dollars
                trades.append(open_trade)
                if journal:
                    journal.record({**open_trade, "bot": "rule_bot", "model": "rule_engine"})
                open_trade = None

        # ── Signal generation ──────────────────────────────────────────────
        if open_trade is None and i >= warmup:
            window = df.iloc[max(0, i - 50): i + 1]
            proba = combiner.predict_proba(window)
            p_buy, p_hold, p_sell = proba

            direction = None
            confidence = 0.0
            if p_buy >= threshold:
                direction, confidence = "buy", p_buy
            elif p_sell >= threshold:
                direction, confidence = "sell", p_sell

            if direction is not None:
                price = closes[i]
                if direction == "buy":
                    sl = price - sl_pts
                    tp = price + tp_pts
                else:
                    sl = price + sl_pts
                    tp = price - tp_pts

                open_trade = {
                    "direction":    direction,
                    "entry_time":   times[i],
                    "entry_price":  price,
                    "sl":           sl,
                    "tp":           tp,
                    "confidence":   confidence,
                    "p_buy":        p_buy,
                    "p_hold":       p_hold,
                    "p_sell":       p_sell,
                    "exit_time":    None,
                    "exit_price":   None,
                    "pnl_pips":     0.0,
                    "pnl_dollars":  0.0,
                    "exit_reason":  "",
                    "sl_pips":      sl_pips,
                    "tp_pips":      tp_pips,
                }

        equity_curve.append(balance)

    # ── Force-close at end of data ─────────────────────────────────────────
    if open_trade is not None:
        price = closes[-1]
        pips = (price - open_trade["entry_price"]) / pip_size
        if open_trade["direction"] == "sell":
            pips = -pips
        _close_rule_trade(open_trade, price, times[-1], pips, "end")
        pnl_dollars = (open_trade["pnl_pips"] / sl_pips) * (balance * risk_pct)
        open_trade["pnl_dollars"] = pnl_dollars
        balance += pnl_dollars
        trades.append(open_trade)

    equity = pd.Series(equity_curve, index=df.index)
    return trades, equity


def _close_rule_trade(t: dict, price, time, pips, reason) -> None:
    t["exit_price"]  = price
    t["exit_time"]   = time
    t["pnl_pips"]    = pips
    t["exit_reason"] = reason


# ─── Live Bot ─────────────────────────────────────────────────────────────────

def _make_live_bot():
    from src.bot_base import BotBase

    class RuleBot(BotBase):
        """Live rule-based signal bot with probability-gated execution."""

        def __init__(self):
            super().__init__(name="RuleBot", tick_interval=15.0)
            cfg = self.config.get("rule_bot", {})
            self.symbol    = cfg.get("symbol", "EURUSD")
            self.timeframe = cfg.get("timeframe", "M15")
            self.candles   = cfg.get("candles", 100)
            self.sl_pips   = cfg.get("sl_pips", 30.0)
            self.tp_pips   = cfg.get("tp_pips", 60.0)
            self._cfg      = cfg
            self._combiner = build_default_combiner(
                fast_ma=cfg.get("fast_ma", 9),
                slow_ma=cfg.get("slow_ma", 21),
                rsi_period=cfg.get("rsi_period", 14),
                bb_period=cfg.get("bb_period", 20),
                threshold=cfg.get("threshold", 0.55),
                weight_ma=cfg.get("weight_ma_cross", 2.0),
                weight_rsi=cfg.get("weight_rsi", 1.5),
                weight_bb=cfg.get("weight_bb", 1.0),
                weight_macd=cfg.get("weight_macd", 1.0),
            )
            self._journal   = TradeJournal()
            self._prev_dir  = None

        def on_start(self) -> None:
            self.log(f"Symbol={self.symbol}  Rules: {self._combiner.list_rules()}")
            self.log(f"Threshold={self._combiner.threshold}  SL={self.sl_pips}p  TP={self.tp_pips}p")

        def on_tick(self) -> None:
            raw = self.rates(self.symbol, self.timeframe, count=self.candles)
            df  = build_features(raw, self._cfg)
            proba = self._combiner.predict_proba(df)
            p_buy, p_hold, p_sell = proba
            threshold = self._combiner.threshold

            self.log(f"P_buy={p_buy:.2f}  P_hold={p_hold:.2f}  P_sell={p_sell:.2f}")

            direction = None
            confidence = 0.0
            if p_buy >= threshold:
                direction, confidence = "buy", p_buy
            elif p_sell >= threshold:
                direction, confidence = "sell", p_sell

            if direction is None or direction == self._prev_dir:
                return

            # Close opposing positions
            opposing = "sell" if direction == "buy" else "buy"
            for pos in self.open_positions(self.symbol):
                if (pos.type == 0) == (opposing == "buy"):
                    self.conn.close_position(pos)

            if self.open_count(self.symbol) >= self.max_open_trades:
                return

            info = self.conn.symbol_info(self.symbol)
            pip  = info.point * 10
            tick = self.conn.get_tick(self.symbol)
            vol  = self.calc_lot(self.symbol, self.sl_pips)

            if direction == "buy":
                entry = tick.ask
                sl    = entry - self.sl_pips * pip
                tp    = entry + self.tp_pips * pip
                self.buy(self.symbol, vol, sl=sl, tp=tp, comment=f"rule_{confidence:.2f}")
            else:
                entry = tick.bid
                sl    = entry + self.sl_pips * pip
                tp    = entry - self.tp_pips * pip
                self.sell(self.symbol, vol, sl=sl, tp=tp, comment=f"rule_{confidence:.2f}")

            self.log(f"{direction.upper()}  conf={confidence:.2f}  sl={sl:.5f}  tp={tp:.5f}")
            self._prev_dir = direction

    return RuleBot


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rule-Based Signal Bot")
    p.add_argument("--backtest",   action="store_true")
    p.add_argument("--data",       default="data/EURUSD_M15.csv")
    p.add_argument("--sl",         type=float, default=30.0)
    p.add_argument("--tp",         type=float, default=60.0)
    p.add_argument("--balance",    type=float, default=10_000.0)
    p.add_argument("--risk",       type=float, default=0.01)
    p.add_argument("--pip-size",   type=float, default=0.0001)
    p.add_argument("--threshold",  type=float, default=0.55)
    p.add_argument("--fast-ma",    type=int,   default=9)
    p.add_argument("--slow-ma",    type=int,   default=21)
    p.add_argument("--save-journal", action="store_true")
    args = p.parse_args()

    if args.backtest:
        csv = Path(args.data)
        if not csv.exists():
            print(f"ERROR: {csv} not found.  Run: python scripts/download_data.py")
            sys.exit(1)

        print(f"Loading {csv}...")
        df = pd.read_csv(csv, index_col="time", parse_dates=True)
        print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

        cfg = {
            "fast_ma": args.fast_ma, "slow_ma": args.slow_ma,
            "rsi_period": 14, "bb_period": 20,
            "threshold": args.threshold,
        }
        journal = TradeJournal() if args.save_journal else None
        trades, equity = backtest_rules(
            df, cfg,
            sl_pips=args.sl, tp_pips=args.tp,
            pip_size=args.pip_size,
            initial_balance=args.balance, risk_pct=args.risk,
            journal=journal,
        )

        symbol = csv.stem.split("_")[0]
        tf = csv.stem.split("_")[1] if "_" in csv.stem else "?"
        performance_report(
            trades, equity, args.balance,
            title="RULE-BASED SIGNAL BACKTEST",
            extra_params={
                "Symbol":      f"{symbol} {tf}",
                "Fast MA":     args.fast_ma,
                "Slow MA":     args.slow_ma,
                "Threshold":   f"{args.threshold:.0%}",
                "SL / TP":     f"{args.sl}p / {args.tp}p",
                "Rules":       "MA-Cross, RSI, BB-Rev, MACD",
            },
        )
    else:
        RuleBot = _make_live_bot()
        bot = RuleBot()
        bot.run()
