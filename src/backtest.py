"""
MA Crossover Backtester.

Loads a CSV of OHLCV data, simulates a moving-average crossover strategy
bar-by-bar with SL/TP, and prints a full performance report.

Usage:
    conda activate envmt5
    python src/backtest.py                            # defaults
    python src/backtest.py --fast 9 --slow 21
    python src/backtest.py --data data/GBPUSD_H1.csv --fast 20 --slow 50 --sl 40 --tp 80

Download data first:
    python scripts/download_data.py --symbol EURUSD --timeframe M15 --bars 50000
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Trade record ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction:   str    # "buy" | "sell"
    entry_time:  object
    entry_price: float
    sl:          float
    tp:          float
    exit_time:   object = None
    exit_price:  float  = 0.0
    pnl_pips:    float  = 0.0
    exit_reason: str    = ""   # "tp" | "sl" | "signal" | "end"


# ─── Strategy signals ────────────────────────────────────────────────────────

def ma_signals(closes: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Returns 1 (buy), -1 (sell), 0 (hold) on each bar."""
    sma_f = _sma(closes, fast)
    sma_s = _sma(closes, slow)
    sig   = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if np.isnan(sma_f[i]) or np.isnan(sma_s[i]):
            continue
        cross_up   = sma_f[i] > sma_s[i] and sma_f[i - 1] <= sma_s[i - 1]
        cross_down = sma_f[i] < sma_s[i] and sma_f[i - 1] >= sma_s[i - 1]
        if cross_up:
            sig[i] = 1
        elif cross_down:
            sig[i] = -1
    return sig


# ─── Backtester ──────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    fast: int        = 9,
    slow: int        = 21,
    sl_pips: float   = 30,
    tp_pips: float   = 60,
    pip_size: float  = 0.0001,      # 0.0001 for 5-digit brokers (EURUSD)
    initial_balance: float = 10_000.0,
    risk_pct: float  = 0.01,        # 1 % risk per trade
) -> tuple[list[Trade], pd.Series]:

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    times   = df.index

    signals = ma_signals(closes, fast, slow)
    sl_pts  = sl_pips * pip_size
    tp_pts  = tp_pips * pip_size

    trades: list[Trade]      = []
    open_trade: Optional[Trade] = None
    balance    = initial_balance
    equity_curve: list[float] = []

    for i in range(len(df)):

        # ── Check SL / TP on open trade ───────────────────────────────────
        if open_trade is not None:
            if open_trade.direction == "buy":
                if lows[i] <= open_trade.sl:
                    _close_trade(open_trade, open_trade.sl, times[i], -sl_pips, "sl")
                    balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                    trades.append(open_trade)
                    open_trade = None
                elif highs[i] >= open_trade.tp:
                    _close_trade(open_trade, open_trade.tp, times[i], tp_pips, "tp")
                    balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                    trades.append(open_trade)
                    open_trade = None
            else:  # sell
                if highs[i] >= open_trade.sl:
                    _close_trade(open_trade, open_trade.sl, times[i], -sl_pips, "sl")
                    balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                    trades.append(open_trade)
                    open_trade = None
                elif lows[i] <= open_trade.tp:
                    _close_trade(open_trade, open_trade.tp, times[i], tp_pips, "tp")
                    balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                    trades.append(open_trade)
                    open_trade = None

        # ── New crossover signal ───────────────────────────────────────────
        if signals[i] == 1 and (open_trade is None or open_trade.direction == "sell"):
            if open_trade is not None:
                pips = (open_trade.entry_price - closes[i]) / pip_size
                _close_trade(open_trade, closes[i], times[i], pips, "signal")
                balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                trades.append(open_trade)
            open_trade = Trade(
                direction="buy",
                entry_time=times[i],
                entry_price=closes[i],
                sl=closes[i] - sl_pts,
                tp=closes[i] + tp_pts,
            )

        elif signals[i] == -1 and (open_trade is None or open_trade.direction == "buy"):
            if open_trade is not None:
                pips = (closes[i] - open_trade.entry_price) / pip_size
                _close_trade(open_trade, closes[i], times[i], pips, "signal")
                balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
                trades.append(open_trade)
            open_trade = Trade(
                direction="sell",
                entry_time=times[i],
                entry_price=closes[i],
                sl=closes[i] + sl_pts,
                tp=closes[i] - tp_pts,
            )

        equity_curve.append(balance)

    # ── Close any trade still open at end of data ──────────────────────────
    if open_trade is not None:
        pips = (closes[-1] - open_trade.entry_price) / pip_size
        if open_trade.direction == "sell":
            pips = -pips
        _close_trade(open_trade, closes[-1], times[-1], pips, "end")
        balance += _risk_pnl(open_trade.pnl_pips, balance, risk_pct, sl_pips)
        trades.append(open_trade)

    equity = pd.Series(equity_curve, index=df.index)
    return trades, equity


# ─── Report ──────────────────────────────────────────────────────────────────

def print_report(
    trades: list[Trade],
    equity: pd.Series,
    initial_balance: float,
    params: dict,
) -> None:
    if not trades:
        print("\nNo trades generated — try different parameters.")
        return

    pips  = [t.pnl_pips for t in trades]
    wins  = [p for p in pips if p > 0]
    losses = [p for p in pips if p <= 0]

    final_bal    = equity.iloc[-1]
    total_return = (final_bal - initial_balance) / initial_balance * 100
    win_rate     = len(wins) / len(trades) * 100
    avg_win      = float(np.mean(wins))  if wins   else 0.0
    avg_loss     = float(np.mean(losses)) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")
    max_dd       = _max_drawdown(equity)

    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    w = 52
    sep = "─" * w
    print(f"\n{'═' * w}")
    print(f"  MA CROSSOVER BACKTEST RESULTS")
    print(f"{'═' * w}")
    print(f"  Symbol    : {params.get('symbol', '?')}  {params.get('timeframe', '?')}")
    print(f"  Strategy  : SMA({params['fast']}) × SMA({params['slow']})  "
          f"SL={params['sl']}p  TP={params['tp']}p  RR={params['tp']/params['sl']:.1f}:1")
    print(f"  Period    : {equity.index[0].date()} → {equity.index[-1].date()}")
    print(sep)
    print(f"  Trades    : {len(trades)}  "
          f"({len(wins)} wins / {len(losses)} losses)")
    print(f"  Win rate  : {win_rate:.1f}%")
    print(f"  Avg win   : +{avg_win:.1f} pips")
    print(f"  Avg loss  : {avg_loss:.1f} pips")
    print(f"  Profit factor: {profit_factor:.2f}")
    print(sep)
    print(f"  Max drawdown : {max_dd:.1f}%")
    print(f"  Sharpe ratio : {sharpe:.2f}")
    print(sep)
    print(f"  Initial balance : ${initial_balance:>10,.2f}")
    print(f"  Final balance   : ${final_bal:>10,.2f}")
    print(f"  Total return    : {total_return:>+10.1f}%")
    print(f"{'═' * w}")
    print(f"  Exit reasons: "
          + "  ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))
    print()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _close_trade(t: Trade, price: float, time, pips: float, reason: str) -> None:
    t.exit_price  = price
    t.exit_time   = time
    t.pnl_pips    = pips
    t.exit_reason = reason


def _risk_pnl(pnl_pips: float, balance: float, risk_pct: float, sl_pips: float) -> float:
    return (pnl_pips / sl_pips) * (balance * risk_pct)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        result[i] = values[i - period + 1:i + 1].mean()
    return result


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MA Crossover Backtester")
    p.add_argument("--data",      default="data/EURUSD_M15.csv")
    p.add_argument("--fast",      type=int,   default=9)
    p.add_argument("--slow",      type=int,   default=21)
    p.add_argument("--sl",        type=float, default=30,    help="Stop loss pips")
    p.add_argument("--tp",        type=float, default=60,    help="Take profit pips")
    p.add_argument("--balance",   type=float, default=10000, help="Starting balance $")
    p.add_argument("--risk",      type=float, default=0.01,  help="Risk per trade 0.01=1%%")
    p.add_argument("--pip-size",  type=float, default=0.0001,
                   help="Pip size (0.0001 for EURUSD, 0.01 for JPY pairs)")
    args = p.parse_args()

    csv = Path(args.data)
    if not csv.exists():
        print(f"ERROR: {csv} not found.")
        print("Download data first:  python scripts/download_data.py")
        sys.exit(1)

    print(f"Loading {csv}...")
    df = pd.read_csv(csv, index_col="time", parse_dates=True)
    print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    params = dict(
        symbol=csv.stem.split("_")[0],
        timeframe=csv.stem.split("_")[1] if "_" in csv.stem else "?",
        fast=args.fast, slow=args.slow,
        sl=args.sl, tp=args.tp,
    )
    print(f"Strategy: SMA({args.fast}) × SMA({args.slow})  SL={args.sl}p  TP={args.tp}p\n")

    trades, equity = run_backtest(
        df,
        fast=args.fast,
        slow=args.slow,
        sl_pips=args.sl,
        tp_pips=args.tp,
        pip_size=args.pip_size,
        initial_balance=args.balance,
        risk_pct=args.risk,
    )
    print_report(trades, equity, args.balance, params)
