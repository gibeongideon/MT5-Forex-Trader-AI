"""Offline MT5/HFM broker-profile checks for V5 replay artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.v5.validation import BrokerExecutionRules


class BrokerCheckError(AssertionError):
    """Raised when a replay trade cannot satisfy the intended broker profile."""


@dataclass(frozen=True)
class BrokerProfile:
    """Broker intent used before dry-run/demo reconciliation is available."""

    base_symbol: str
    broker_symbol: str | None = None
    magic_number: int | None = None
    min_stop_distance_pips: float = 0.0
    tradable_symbols: list[dict] = field(default_factory=list)


def resolve_broker_symbol(profile: BrokerProfile) -> str:
    """Mirror PipelineBot's suffix preference without calling MT5."""

    if profile.broker_symbol:
        return profile.broker_symbol
    base = profile.base_symbol[:6].upper()
    candidates = [
        item
        for item in profile.tradable_symbols
        if str(item.get("name", ""))[:6].upper() == base and int(item.get("trade_mode", 0)) == 4
    ]
    candidates.sort(key=lambda item: (not bool(item.get("visible", False)), len(str(item["name"]))))
    if candidates:
        return str(candidates[0]["name"])
    return profile.base_symbol


def build_broker_reconciliation(
    profile: BrokerProfile,
    trades: pd.DataFrame | list[dict],
    rules: BrokerExecutionRules,
) -> dict:
    """Validate replay trades against symbol, magic, lot, and stop constraints."""

    frame = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(list(trades))
    broker_symbol = resolve_broker_symbol(profile)
    checks = {
        "symbol": "pass",
        "magic": "pass",
        "volume": "pass",
        "stop_distance": "pass",
    }
    if frame.empty:
        return _report(profile, broker_symbol, checks, 0)

    _require_columns(frame, ["symbol", "magic", "direction", "volume", "entry_price", "sl", "tp"])
    bad_symbol = frame[frame["symbol"].astype(str) != broker_symbol]
    if not bad_symbol.empty:
        raise BrokerCheckError(f"{len(bad_symbol)} trades do not match broker symbol {broker_symbol}")

    if profile.magic_number is not None:
        bad_magic = frame[frame["magic"].astype("int64") != int(profile.magic_number)]
        if not bad_magic.empty:
            raise BrokerCheckError(f"{len(bad_magic)} trades do not match magic {profile.magic_number}")

    bad_volume = frame[
        (frame["volume"].astype(float) < rules.min_lot)
        | (frame["volume"].astype(float) > rules.max_lot)
        | (~frame["volume"].astype(float).map(lambda value: _is_lot_step(value, rules.lot_step)))
    ]
    if not bad_volume.empty:
        raise BrokerCheckError(f"{len(bad_volume)} trades violate broker volume rules")

    distances = _stop_distances(frame, rules)
    too_close = distances[
        (distances["sl_distance_pips"] < profile.min_stop_distance_pips)
        | (distances["tp_distance_pips"] < profile.min_stop_distance_pips)
    ]
    if not too_close.empty:
        raise BrokerCheckError(
            f"{len(too_close)} trades violate min stop distance {profile.min_stop_distance_pips} pips"
        )

    report = _report(profile, broker_symbol, checks, len(frame))
    report["min_stop_distance_pips"] = profile.min_stop_distance_pips
    return report


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BrokerCheckError(f"missing broker check columns: {missing}")


def _is_lot_step(value: float, step: float) -> bool:
    steps = round(value / step)
    return abs(value - steps * step) < 1e-9


def _stop_distances(frame: pd.DataFrame, rules: BrokerExecutionRules) -> pd.DataFrame:
    entry = frame["entry_price"].astype(float)
    sl_column = "initial_sl" if "initial_sl" in frame.columns else "sl"
    tp_column = "initial_tp" if "initial_tp" in frame.columns else "tp"
    sl = frame[sl_column].astype(float)
    tp = frame[tp_column].astype(float)
    return pd.DataFrame(
        {
            "sl_distance_pips": (entry - sl).abs() / rules.pip_size,
            "tp_distance_pips": (tp - entry).abs() / rules.pip_size,
        },
        index=frame.index,
    )


def _report(
    profile: BrokerProfile,
    broker_symbol: str,
    checks: dict,
    checked_trades: int,
) -> dict:
    return {
        "status": "broker_profile_checked",
        "base_symbol": profile.base_symbol,
        "broker_symbol": broker_symbol,
        "magic_number": profile.magic_number,
        "checked_trades": checked_trades,
        "checks": checks,
    }
