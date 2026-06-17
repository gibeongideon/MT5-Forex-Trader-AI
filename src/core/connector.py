"""Platform-agnostic connector interface + factory.

Both MT5Connector (mt5linux/rpyc) and MT4Connector (file bridge) satisfy this Protocol
structurally — no inheritance change to the live MT5 class. Bots obtain their connector via
`get_connector(platform)` so the same bot code can drive either platform.

Platform resolution order (first non-empty wins):
    explicit arg  →  env BOT_PLATFORM  →  config trading.platform  →  "mt5"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


@runtime_checkable
class Connector(Protocol):
    """The trading interface every platform connector must provide."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def reset_singleton(self) -> None: ...
    def account_info(self): ...
    def account_balance(self) -> float: ...
    def get_rates(self, symbol: str, timeframe: str, count: int = 500): ...
    def get_tick(self, symbol: str): ...
    def symbol_info(self, symbol: str): ...
    def open_position(self, symbol: str, order_type: str, volume: float,
                      sl: float = 0.0, tp: float = 0.0, comment: str = "", magic: int = 0) -> dict: ...
    def close_position(self, position) -> dict: ...
    def close_position_partial(self, position, volume: float) -> dict: ...
    def get_positions(self, symbol: str | None = None, magic: int | None = None): ...
    def modify_position(self, ticket: int, sl: float, tp: float) -> dict: ...
    def calc_lot_size(self, symbol: str, sl_pips: float, risk_pct: float = 0.01) -> float: ...


def resolve_platform(platform: str | None = None) -> str:
    if platform:
        return platform.lower()
    env = os.getenv("BOT_PLATFORM")
    if env:
        return env.lower()
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return str(cfg.get("trading", {}).get("platform", "mt5")).lower()
    except Exception:
        return "mt5"


def get_connector(platform: str | None = None) -> Connector:
    """Return an MT5 or MT4 connector. Imports are lazy so using one platform never
    requires the other's dependencies (mt5linux vs the file bridge)."""
    plat = resolve_platform(platform)
    if plat == "mt5":
        from .mt5_connector import MT5Connector
        return MT5Connector()
    if plat == "mt4":
        from .mt4_connector import MT4Connector
        return MT4Connector()
    raise ValueError(f"Unknown platform '{plat}' (expected 'mt5' or 'mt4')")
