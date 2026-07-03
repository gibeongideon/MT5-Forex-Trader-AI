"""V5 candle-trail validation with Lumibot-style artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import pandas as pd

from src.v5.artifacts import V5ArtifactWriter
from src.v5.champion_validation import load_ohlcv
from src.v5.validation import BrokerExecutionRules


@dataclass
class V5CandleTrailValidationConfig:
    symbol: str
    run_id: str
    artifact_root: str | Path
    broker_rules: BrokerExecutionRules
    threshold: float
    requested_lot: float
    sl_pips: float
    tp_pips: float
    trail_activation_pips: float = 15.0
    trail_pips_behind: float = 10.0
    max_bars_low: int = 1
    max_bars_med: int = 2
    max_bars_high: int = 4
    initial_balance: float = 10_000.0
    dollars_per_pip_per_lot: float = 10.0
    data_path: str | Path | None = None
    signals_path: str | Path | None = None


@dataclass
class V5CandleTrailValidationResult:
    run_dir: Path
    trades: list[dict]
    equity: pd.Series
    stats: dict


def run_candle_trail_validation(
    cfg: V5CandleTrailValidationConfig,
    *,
    prices: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
) -> V5CandleTrailValidationResult:
    prices = prices if prices is not None else load_ohlcv(_required_path(cfg.data_path, "data_path"))
    signals = signals if signals is not None else _load_signals(_required_path(cfg.signals_path, "signals_path"))
    replay = replay_candle_trail(
        prices,
        signals,
        cfg.broker_rules,
        threshold=cfg.threshold,
        requested_lot=cfg.requested_lot,
        sl_pips=cfg.sl_pips,
        tp_pips=cfg.tp_pips,
        trail_activation_pips=cfg.trail_activation_pips,
        trail_pips_behind=cfg.trail_pips_behind,
        max_bars_low=cfg.max_bars_low,
        max_bars_med=cfg.max_bars_med,
        max_bars_high=cfg.max_bars_high,
        initial_balance=cfg.initial_balance,
        dollars_per_pip_per_lot=cfg.dollars_per_pip_per_lot,
    )
    stats = _stats(cfg, replay["trades"], replay["equity"])
    run_dir = V5ArtifactWriter(cfg.artifact_root).write_run(
        run_id=cfg.run_id,
        settings=_settings(cfg),
        trades=replay["trades"],
        equity=replay["equity"],
        stats=stats,
        folds=[],
        reconciliation={
            "status": "research_replay_only",
            "note": "Candle-trail replay over supplied OOS candle probabilities; not paper/live reconciled.",
        },
    )
    return V5CandleTrailValidationResult(
        run_dir=run_dir,
        trades=replay["trades"],
        equity=replay["equity"],
        stats=stats,
    )


def replay_candle_trail(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    rules: BrokerExecutionRules,
    *,
    threshold: float,
    requested_lot: float,
    sl_pips: float,
    tp_pips: float,
    trail_activation_pips: float,
    trail_pips_behind: float,
    max_bars_low: int,
    max_bars_med: int,
    max_bars_high: int,
    initial_balance: float = 10_000.0,
    dollars_per_pip_per_lot: float = 10.0,
) -> dict:
    signals = _normalize_signals(signals).reindex(prices.index)
    balance = float(initial_balance)
    equity_points: list[float] = []
    trades: list[dict] = []
    open_trade: dict | None = None
    pending: dict | None = None
    volume = rules.normalize_lot(requested_lot)

    for i, (ts, row) in enumerate(prices.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if open_trade is not None:
            open_trade["bars_held"] += 1
            _advance_trailing_stop(
                open_trade,
                high,
                low,
                rules,
                trail_activation_pips,
                trail_pips_behind,
            )
            reason = _trail_exit_reason(open_trade, high, low)
            if reason is None and open_trade["bars_held"] >= open_trade["max_bars"]:
                reason = "bar_end"
            if reason is not None:
                balance += _close(open_trade, ts, close, reason, rules, dollars_per_pip_per_lot)
                trades.append(open_trade)
                open_trade = None

        if open_trade is None and pending is not None and i >= pending["entry_index"]:
            open_trade = _open_trade(
                pending["signal_time"],
                ts,
                close,
                pending["direction"],
                pending["confidence"],
                volume,
                rules,
                sl_pips,
                tp_pips,
                pending["max_bars"],
            )
            pending = None

        equity_points.append(balance)

        if open_trade is not None or pending is not None or volume <= 0:
            continue
        sig = signals.loc[ts]
        direction = _direction(sig, threshold)
        if direction is None:
            continue
        confidence = float(max(sig["P_buy"], sig["P_sell"]))
        pending = {
            "signal_time": ts,
            "entry_index": i + rules.entry_delay_bars,
            "direction": direction,
            "confidence": confidence,
            "max_bars": _max_bars(confidence, max_bars_low, max_bars_med, max_bars_high),
        }
        if pending["entry_index"] <= i:
            open_trade = _open_trade(
                pending["signal_time"],
                ts,
                close,
                pending["direction"],
                pending["confidence"],
                volume,
                rules,
                sl_pips,
                tp_pips,
                pending["max_bars"],
            )
            pending = None

    if open_trade is not None and len(prices) > 0:
        balance += _close(
            open_trade,
            prices.index[-1],
            float(prices.iloc[-1]["close"]),
            "end",
            rules,
            dollars_per_pip_per_lot,
        )
        trades.append(open_trade)
        equity_points[-1] = balance

    equity = pd.Series(equity_points, index=prices.index[: len(equity_points)], name="equity")
    return {"trades": trades, "equity": equity}


def _open_trade(
    signal_time,
    entry_time,
    close: float,
    direction: str,
    confidence: float,
    volume: float,
    rules: BrokerExecutionRules,
    sl_pips: float,
    tp_pips: float,
    max_bars: int,
) -> dict:
    spread = rules.spread_pips * rules.pip_size
    slip = rules.slippage_pips * rules.pip_size
    if direction == "buy":
        entry = close + spread + slip
        sl = entry - sl_pips * rules.pip_size
        tp = entry + tp_pips * rules.pip_size
    else:
        entry = close - spread - slip
        sl = entry + sl_pips * rules.pip_size
        tp = entry - tp_pips * rules.pip_size
    return {
        "signal_time": signal_time,
        "entry_time": entry_time,
        "direction": direction,
        "confidence": confidence,
        "volume": volume,
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "max_bars": max_bars,
        "bars_held": 0,
        "peak_pips": 0.0,
    }


def _advance_trailing_stop(
    trade: dict,
    high: float,
    low: float,
    rules: BrokerExecutionRules,
    activation_pips: float,
    pips_behind: float,
) -> None:
    if trade["direction"] == "buy":
        peak = (high - trade["entry_price"]) / rules.pip_size
    else:
        peak = (trade["entry_price"] - low) / rules.pip_size
    trade["peak_pips"] = max(float(trade["peak_pips"]), float(peak))
    if trade["peak_pips"] < activation_pips:
        return
    offset = trade["peak_pips"] - pips_behind
    if offset <= 0:
        return
    if trade["direction"] == "buy":
        trade["sl"] = max(trade["sl"], trade["entry_price"] + offset * rules.pip_size)
    else:
        trade["sl"] = min(trade["sl"], trade["entry_price"] - offset * rules.pip_size)


def _trail_exit_reason(trade: dict, high: float, low: float) -> str | None:
    if trade["direction"] == "buy":
        if low <= trade["sl"]:
            return "trail_sl"
        if high >= trade["tp"]:
            return "tp"
    else:
        if high >= trade["sl"]:
            return "trail_sl"
        if low <= trade["tp"]:
            return "tp"
    return None


def _close(
    trade: dict,
    ts,
    close: float,
    reason: str,
    rules: BrokerExecutionRules,
    dollars_per_pip_per_lot: float,
) -> float:
    if reason in {"trail_sl", "sl"}:
        exit_price = trade["sl"]
    elif reason == "tp":
        exit_price = trade["tp"]
    else:
        exit_price = close
    raw = (exit_price - trade["entry_price"]) / rules.pip_size
    if trade["direction"] == "sell":
        raw = -raw
    trade["exit_time"] = ts
    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    trade["pnl_pips"] = raw - rules.round_trip_cost_pips
    trade["pnl_dollars"] = trade["pnl_pips"] * dollars_per_pip_per_lot * trade["volume"]
    return trade["pnl_dollars"]


def _direction(row: pd.Series, threshold: float) -> str | None:
    p_buy = float(row.get("P_buy", 0.0))
    p_sell = float(row.get("P_sell", 0.0))
    if p_buy >= threshold and p_buy > p_sell:
        return "buy"
    if p_sell >= threshold and p_sell > p_buy:
        return "sell"
    return None


def _max_bars(confidence: float, low: int, med: int, high: int) -> int:
    if confidence < 0.70:
        return low
    if confidence < 0.80:
        return med
    return high


def _normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    rename = {
        "candle_p_buy": "P_buy",
        "candle_p_hold": "P_hold",
        "candle_p_sell": "P_sell",
        "p_buy": "P_buy",
        "p_hold": "P_hold",
        "p_sell": "P_sell",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    missing = [col for col in ["P_buy", "P_sell"] if col not in out.columns]
    if missing:
        raise ValueError(f"missing candle signal columns: {missing}")
    if "P_hold" not in out.columns:
        out["P_hold"] = 1.0 - out["P_buy"] - out["P_sell"]
    out.index = pd.to_datetime(out.index)
    return out.sort_index()[["P_buy", "P_hold", "P_sell"]]


def _load_signals(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    frame = pd.read_csv(path)
    time_col = next((c for c in frame.columns if "time" in c.lower() or c.lower() in {"date", "datetime"}), None)
    if time_col is not None:
        frame[time_col] = pd.to_datetime(frame[time_col])
        frame = frame.set_index(time_col)
    else:
        frame.index = pd.to_datetime(frame.index)
    return frame


def _stats(cfg: V5CandleTrailValidationConfig, trades: list[dict], equity: pd.Series) -> dict:
    pnl = [float(t.get("pnl_pips", 0.0)) for t in trades]
    wins = [x for x in pnl if x > 0]
    sharpe = _annualized_sharpe(equity)
    daily_sharpe = _daily_sharpe(equity)
    return {
        "symbol": cfg.symbol,
        "mode": "candle_trail",
        "trades": len(trades),
        "win_rate": len(wins) / len(pnl) if pnl else 0.0,
        "sharpe": sharpe,
        "daily_sharpe": daily_sharpe,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "max_drawdown": _max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]) if len(equity) else cfg.initial_balance,
        "research_only": True,
    }


def _settings(cfg: V5CandleTrailValidationConfig) -> dict:
    payload = asdict(cfg)
    payload["broker_rules"] = asdict(cfg.broker_rules)
    payload["mode"] = "candle_trail"
    return payload


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA)
    return float(dd.fillna(0.0).max())


def _annualized_sharpe(equity: pd.Series) -> float:
    returns = equity.pct_change(fill_method=None).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    bars_per_year = _bars_per_year(equity)
    return float(returns.mean() / returns.std() * math.sqrt(bars_per_year))


def _daily_sharpe(equity: pd.Series) -> float:
    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * math.sqrt(252))


def _bars_per_year(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 252.0
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400, 1e-9)
    return float(len(equity) / (days / 365.25))


def _required_path(path: str | Path | None, name: str) -> str | Path:
    if path is None:
        raise ValueError(f"{name} is required when data is not supplied directly")
    return path
