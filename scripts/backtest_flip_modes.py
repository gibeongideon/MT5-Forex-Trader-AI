"""
backtest_flip_modes.py — Compare always | hedge_loss | hedge_exit

Loads the live champion model for each symbol and simulates all three flip
modes on the full historical dataset.  No model retraining — uses the loaded
scaler, encoder, and XGBoost weights exactly as deployed.

Usage:
    conda run -n envmt5 python scripts/backtest_flip_modes.py
    conda run -n envmt5 python scripts/backtest_flip_modes.py --symbol EURUSD
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PredictorPipeline
from src.evaluation.metrics import sharpe_ratio, max_drawdown, calmar_ratio

# ── Constants ──────────────────────────────────────────────────────────────────

MODES = ["always", "hedge_loss", "hedge_exit"]

SYMBOL_PARAMS = {
    "EURUSD": dict(
        model_dir = "data/models/pipeline_EURUSD",
        data_path = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
    ),
    "USDJPY": dict(
        model_dir = "data/models/pipeline_USDJPY",
        data_path = "data/USDJPY_M15.csv",
        pip_size  = 0.01,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
    ),
}

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01     # fraction of balance risked per trade
SPREAD_PIPS     = 1.0
COMMISSION_PIPS = 0.5      # round-trip flat cost
THRESHOLD       = 0.40

# ── Position dataclass ─────────────────────────────────────────────────────────

@dataclass
class Pos:
    ticket:        int
    direction:     str    # "buy" | "sell"
    entry_bar:     int
    entry_price:   float
    entry_balance: float  # balance snapshot for consistent lot sizing
    sl:            float  # price level
    tp:            float  # price level
    sl_pips:       float
    tp_pips:       float
    is_hedged:     bool   = False   # hedge_exit only: close at first profit
    exit_bar:      Optional[int] = None
    exit_price:    Optional[float] = None
    exit_reason:   str   = ""
    pnl_pips:      float = 0.0
    pnl_dollars:   float = 0.0


def _close(pos: Pos, bar: int, exit_p: float, reason: str,
           pip_size: float, cost_pips: float) -> float:
    """Fill exit fields, compute pnl_pips and pnl_dollars. Returns pnl_dollars."""
    pos.exit_bar   = bar
    pos.exit_price = exit_p
    pos.exit_reason = reason
    raw = (
        (exit_p - pos.entry_price) / pip_size
        if pos.direction == "buy"
        else (pos.entry_price - exit_p) / pip_size
    )
    pos.pnl_pips    = raw - cost_pips
    dpp             = (pos.entry_balance * RISK_PCT) / pos.sl_pips   # $ per pip
    pos.pnl_dollars = pos.pnl_pips * dpp
    return pos.pnl_dollars


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_mode(
    signals:   pd.DataFrame,  # P_buy, P_sell, signal, confidence  (from predict_batch)
    prices:    pd.DataFrame,  # open, high, low, close  aligned to signals index
    mode:      str,
    sl_pips:   float,
    tp_pips:   float,
    pip_size:  float,
) -> dict:
    """
    Bar-by-bar simulation of one flip mode.

    Signal fires at bar close; fill is same bar's close ± spread.
    SL/TP checked via bar high/low (same as Backtester._check_exit).
    hedge_exit positions are closed when price first crosses back through entry.
    """
    cost_pips  = SPREAD_PIPS + COMMISSION_PIPS
    spread_pts = SPREAD_PIPS * pip_size
    sl_pts     = sl_pips  * pip_size
    tp_pts     = tp_pips  * pip_size

    balance       = INITIAL_BALANCE
    peak          = INITIAL_BALANCE
    max_dd_pct    = 0.0
    equity_pts: List[float] = []
    all_trades: List[Pos]   = []
    open_pos:   List[Pos]   = []
    hedged_set: set          = set()
    ticket        = 0

    p_buys  = signals["P_buy"].values
    p_sells = signals["P_sell"].values
    opens   = prices["open"].reindex(signals.index).values
    highs   = prices["high"].reindex(signals.index).values
    lows    = prices["low"].reindex(signals.index).values
    closes  = prices["close"].reindex(signals.index).values
    n = len(signals)

    for i in range(n):
        high  = float(highs[i])
        low   = float(lows[i])
        close = float(closes[i])
        open_ = float(opens[i])

        # ── 1. SL / TP exit ───────────────────────────────────────────────
        alive: List[Pos] = []
        for pos in open_pos:
            if pos.direction == "buy":
                if low <= pos.sl:
                    balance += _close(pos, i, pos.sl, "sl", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    continue
                if high >= pos.tp:
                    balance += _close(pos, i, pos.tp, "tp", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    continue
            else:  # sell
                if high >= pos.sl:
                    balance += _close(pos, i, pos.sl, "sl", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    continue
                if low <= pos.tp:
                    balance += _close(pos, i, pos.tp, "tp", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    continue
            alive.append(pos)
        open_pos = alive

        # ── 2. hedge_exit: close hedged losers at first profit ────────────
        if mode == "hedge_exit" and hedged_set:
            alive = []
            for pos in open_pos:
                if pos.ticket not in hedged_set:
                    alive.append(pos)
                    continue
                turned = False
                if pos.direction == "buy" and high > pos.entry_price:
                    # Close at open if already above entry, else at entry (break-even)
                    exit_p = open_ if open_ >= pos.entry_price else pos.entry_price
                    balance += _close(pos, i, exit_p, "hedge_exit", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    turned = True
                elif pos.direction == "sell" and low < pos.entry_price:
                    exit_p = open_ if open_ <= pos.entry_price else pos.entry_price
                    balance += _close(pos, i, exit_p, "hedge_exit", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket)
                    all_trades.append(pos)
                    turned = True
                if not turned:
                    alive.append(pos)
            open_pos = alive

        # ── 3. Track equity ────────────────────────────────────────────────
        peak        = max(peak, balance)
        dd          = (peak - balance) / peak * 100
        max_dd_pct  = max(max_dd_pct, dd)
        equity_pts.append(balance)

        # ── 4. Signal ──────────────────────────────────────────────────────
        p_buy  = float(p_buys[i])
        p_sell = float(p_sells[i])
        if p_buy >= THRESHOLD and p_buy > p_sell:
            direction = "buy"
        elif p_sell >= THRESHOLD and p_sell > p_buy:
            direction = "sell"
        else:
            continue

        # ── 5. Same-direction guard ────────────────────────────────────────
        if any(p.direction == direction for p in open_pos):
            continue

        # ── 6. Handle opposite-direction positions ─────────────────────────
        opposite = [p for p in open_pos if p.direction != direction]
        for pos in opposite:
            profit_pips = (
                (close - pos.entry_price) / pip_size
                if pos.direction == "buy"
                else (pos.entry_price - close) / pip_size
            )
            in_profit = profit_pips > 0

            if mode == "always" or in_profit:
                # Close immediately (always-mode, or profitable flip)
                balance += _close(pos, i, close, "flip", pip_size, cost_pips)
                hedged_set.discard(pos.ticket)
                open_pos.remove(pos)
                all_trades.append(pos)
            else:
                # Losing opposite — keep open as hedge
                if mode == "hedge_exit":
                    hedged_set.add(pos.ticket)
                # hedge_loss: keep open, let SL/TP close it naturally

        # ── 7. Open new position ───────────────────────────────────────────
        if any(p.direction == direction for p in open_pos):
            continue   # already have this direction after closes

        fill = close + spread_pts if direction == "buy" else close - spread_pts
        if direction == "buy":
            sl = fill - sl_pts
            tp = fill + tp_pts
        else:
            sl = fill + sl_pts
            tp = fill - tp_pts

        ticket += 1
        open_pos.append(Pos(
            ticket        = ticket,
            direction     = direction,
            entry_bar     = i,
            entry_price   = fill,
            entry_balance = balance,
            sl            = sl,
            tp            = tp,
            sl_pips       = sl_pips,
            tp_pips       = tp_pips,
        ))

    # ── Force-close remaining at end of data ──────────────────────────────
    last_close = float(closes[-1])
    for pos in open_pos:
        balance += _close(pos, n - 1, last_close, "end", pip_size, cost_pips)
        all_trades.append(pos)

    # ── Metrics ────────────────────────────────────────────────────────────
    eq_index = signals.index[:len(equity_pts)]
    eq = pd.Series(equity_pts, index=eq_index)
    pnl_pips = [t.pnl_pips for t in all_trades]
    wins = sum(1 for p in pnl_pips if p > 0)
    n_trades = len(all_trades)
    n_hedge_exits = sum(1 for t in all_trades if t.exit_reason == "hedge_exit")
    n_concurrent  = sum(1 for t in all_trades if t.is_hedged)  # currently 0; future use

    return {
        "mode":           mode,
        "n_trades":       n_trades,
        "win_rate":       wins / n_trades if n_trades else 0.0,
        "net_pnl_pct":    (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "max_dd_pct":     max_dd_pct,
        "sharpe":         sharpe_ratio(eq),
        "calmar":         calmar_ratio(eq),
        "n_hedge_exits":  n_hedge_exits,
        "balance":        balance,
    }


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> None:
    params   = SYMBOL_PARAMS[symbol]
    pip_size = params["pip_size"]
    sl_pips  = params["sl_pips"]
    tp_pips  = params["tp_pips"]

    print(f"\n{'=' * 64}")
    print(f"  {symbol}  —  {params['model_dir']}")
    print(f"{'=' * 64}")

    # Load trained champion (no retraining)
    pipe = PredictorPipeline.from_config()
    pipe.load(params["model_dir"])

    # Load raw OHLCV
    df_raw = pd.read_csv(params["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print(f"  Data: {len(df_raw):,} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # Build features with LOADED scaler + encoder — no refitting
    print("  Building features (fit=False — uses loaded artifacts)...", flush=True)
    X_base, y = pipe._fp.build(df_raw, fit=False)

    if pipe._enc is not None:
        latent = pipe._enc.transform(df_raw)
        shared = X_base.index.intersection(latent.index)
        X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
    else:
        X = X_base

    # Align to training column order
    for c in pipe._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[pipe._feature_cols]
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Bulk predictions using loaded model
    signals = pipe.predict_batch(X)
    prices  = df_raw.reindex(signals.index)

    n_buy  = (signals["signal"] == "buy").sum()
    n_sell = (signals["signal"] == "sell").sum()
    n_hold = (signals["signal"] == "hold").sum()
    print(f"  Signals: buy={n_buy:,}  sell={n_sell:,}  hold={n_hold:,}")

    # Run all three modes
    results = []
    for mode in MODES:
        print(f"  Simulating [{mode:<12}]...", end=" ", flush=True)
        r = simulate_mode(signals, prices, mode, sl_pips, tp_pips, pip_size)
        results.append(r)
        extra = f"  hedge_exits={r['n_hedge_exits']}" if r["n_hedge_exits"] else ""
        print(f"done → {r['n_trades']:,} trades  PnL={r['net_pnl_pct']:+.1f}%{extra}")

    # ── Results table ──────────────────────────────────────────────────────
    print()
    title = (
        f"{symbol}  SL={sl_pips:.0f}p  TP={tp_pips:.0f}p  "
        f"threshold={THRESHOLD:.0%}  risk={RISK_PCT:.0%}"
    )
    print(title)
    hdr = (
        f"{'Mode':<14} {'Trades':>7} {'Win%':>7} "
        f"{'Net PnL':>9} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'HedgeExit':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['mode']:<14} "
            f"{r['n_trades']:>7,d} "
            f"{r['win_rate']:>6.1%} "
            f"{r['net_pnl_pct']:>+8.1f}% "
            f"{r['max_dd_pct']:>7.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['calmar']:>8.2f} "
            f"{r['n_hedge_exits']:>10,d}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compare flip modes on champion models")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_PARAMS.keys()),
                   help="Single symbol (default: all)")
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_PARAMS.keys())
    for sym in symbols:
        run_symbol(sym)
    print()


if __name__ == "__main__":
    main()
