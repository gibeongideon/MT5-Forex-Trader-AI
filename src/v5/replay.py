"""Broker-realistic signal replay helpers for V5 validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.v5.validation import BrokerExecutionRules


@dataclass
class V5ReplayResult:
    trades: list[dict]
    equity: pd.Series
    settings: dict


def replay_signal_frame(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    rules: BrokerExecutionRules,
    *,
    sl_pips: float,
    tp_pips: float,
    initial_balance: float = 10_000.0,
    dollars_per_pip_per_lot: float = 10.0,
) -> V5ReplayResult:
    """Replay buy/sell/hold signals with delay, costs, SL/TP, and lot rules.

    This is intentionally simpler than the production backtester: one position at
    a time, deterministic costs, and explicit signal-frame input. Its job is to
    make V5 broker assumptions visible and artifact-friendly.
    """

    aligned = signals.reindex(prices.index)
    balance = float(initial_balance)
    equity_points: list[float] = []
    trades: list[dict] = []
    open_trade: dict | None = None
    pending: dict | None = None

    for i, (ts, row) in enumerate(prices.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if open_trade is not None:
            exit_reason = _exit_reason(open_trade, high, low)
            if exit_reason:
                balance = _close_trade(
                    open_trade,
                    ts,
                    exit_reason,
                    balance,
                    rules,
                    dollars_per_pip_per_lot,
                )
                trades.append(open_trade)
                open_trade = None

        if open_trade is None and pending is not None and i >= pending["entry_index"]:
            open_trade = _open_trade(pending, ts, close, rules, sl_pips, tp_pips)
            pending = None

        if open_trade is None and pending is None and ts in aligned.index:
            sig = aligned.loc[ts]
            direction = str(sig.get("signal", "hold")).lower()
            if direction in ("buy", "sell"):
                volume = rules.normalize_lot(float(sig.get("requested_lot", 0.0)))
                if volume > 0:
                    pending = {
                        "signal_time": ts,
                        "entry_index": i + rules.entry_delay_bars,
                        "direction": direction,
                        "confidence": float(sig.get("confidence", 0.0)),
                        "volume": volume,
                    }
                    if pending["entry_index"] <= i:
                        open_trade = _open_trade(pending, ts, close, rules, sl_pips, tp_pips)
                        pending = None

        equity_points.append(balance)

    if open_trade is not None and len(prices) > 0:
        last_ts = prices.index[-1]
        last_close = float(prices.iloc[-1]["close"])
        raw_pips = (last_close - open_trade["entry_price"]) / rules.pip_size
        if open_trade["direction"] == "sell":
            raw_pips = -raw_pips
        open_trade["exit_time"] = last_ts
        open_trade["exit_price"] = last_close
        open_trade["exit_reason"] = "end"
        open_trade["pnl_pips"] = raw_pips - rules.round_trip_cost_pips
        open_trade["pnl_dollars"] = (
            open_trade["pnl_pips"] * dollars_per_pip_per_lot * open_trade["volume"]
        )
        balance += open_trade["pnl_dollars"]
        trades.append(open_trade)
        equity_points[-1] = balance

    equity = pd.Series(equity_points, index=prices.index[: len(equity_points)], name="equity")
    return V5ReplayResult(
        trades=trades,
        equity=equity,
        settings={
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "initial_balance": initial_balance,
            "dollars_per_pip_per_lot": dollars_per_pip_per_lot,
            "execution_rules": rules.__dict__,
        },
    )


def _open_trade(
    pending: dict,
    ts,
    close: float,
    rules: BrokerExecutionRules,
    sl_pips: float,
    tp_pips: float,
) -> dict:
    spread = rules.spread_pips * rules.pip_size
    slip = rules.slippage_pips * rules.pip_size
    if pending["direction"] == "buy":
        entry = close + spread + slip
        sl = entry - sl_pips * rules.pip_size
        tp = entry + tp_pips * rules.pip_size
    else:
        entry = close - spread - slip
        sl = entry + sl_pips * rules.pip_size
        tp = entry - tp_pips * rules.pip_size
    return {
        "signal_time": pending["signal_time"],
        "entry_time": ts,
        "direction": pending["direction"],
        "confidence": pending["confidence"],
        "volume": pending["volume"],
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
    }


def _exit_reason(trade: dict, high: float, low: float) -> str | None:
    if trade["direction"] == "buy":
        if low <= trade["sl"]:
            return "sl"
        if high >= trade["tp"]:
            return "tp"
    else:
        if high >= trade["sl"]:
            return "sl"
        if low <= trade["tp"]:
            return "tp"
    return None


def _close_trade(
    trade: dict,
    ts,
    reason: str,
    balance: float,
    rules: BrokerExecutionRules,
    dollars_per_pip_per_lot: float,
) -> float:
    if reason == "tp":
        raw_pips = trade["tp_pips"]
        exit_price = trade["tp"]
    else:
        raw_pips = -trade["sl_pips"]
        exit_price = trade["sl"]
    trade["exit_time"] = ts
    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    trade["pnl_pips"] = raw_pips - rules.round_trip_cost_pips
    trade["pnl_dollars"] = trade["pnl_pips"] * dollars_per_pip_per_lot * trade["volume"]
    return balance + trade["pnl_dollars"]
