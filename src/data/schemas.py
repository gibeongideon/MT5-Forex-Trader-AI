"""OHLCV data schema validation utilities."""
from __future__ import annotations

import pandas as pd


OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}
REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def validate_ohlcv(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Raise ValueError if df is missing required OHLCV columns or has bad values."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    if df.index.duplicated().any():
        raise ValueError(f"{name}: duplicate timestamps found")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index is not sorted in ascending order")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"{name}: high < low on some rows")

    if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
        raise ValueError(f"{name}: close is outside high/low range on some rows")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df
