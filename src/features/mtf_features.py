"""mtf_features.py — leak-free lower-timeframe → 4H feature aggregator.

For a 4H base prediction, summarise what the 15M/30M/1H/2H structure looked like *during the
forming 4H bar*. STRICTLY point-in-time: every lower-TF feature is past-only (uses bars ≤ t), and
each is mapped to the 4H grid by the LAST lower-TF value WITHIN each 4H bin
(`resample("4h").last()`). Those lower-TF bars close at/before the 4H bar's close, and the model
trades the NEXT 4H bar (pos.shift(1) downstream) — so this does not peek ahead. This is the exact
spot where the original +3.14 enc8 champion leaked; do not change the alignment without re-testing
causality (`tests/test_mtf_features.py`).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.cta.signals import ewmac

TAG = {"2h": "h2", "1h": "h1", "30min": "m30", "15min": "m15"}


def _resample_ohlc(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = d.resample(rule, label="left", closed="left")
    return pd.DataFrame({
        "high": o["high"].max(), "low": o["low"].min(), "close": o["close"].last(),
    }).dropna(subset=["close"])


def _tf_features(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Compact past-only feature set on one timeframe's OHLC (all use close ≤ t)."""
    c = df_tf["close"]
    f = pd.DataFrame(index=df_tf.index)
    f["ewmac"] = ewmac(c.to_frame("x"))["x"]                    # continuous trend forecast
    f["ret8"] = c.pct_change(8)
    f["ema_ratio"] = c / c.ewm(span=20, min_periods=10).mean() - 1.0
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-12))
    tr = pd.concat([(df_tf["high"] - df_tf["low"]),
                    (df_tf["high"] - c.shift(1)).abs(),
                    (df_tf["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["mom_atr"] = (c - c.shift(8)) / (atr + 1e-12)
    f["rvol"] = c.pct_change().rolling(20).std()
    return f


def mtf_features(sym: str | None = None, lower=("2h", "1h", "30min", "15min"),
                 base: str = "4h", df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lower-TF features aligned to the 4H grid (leak-free). Index = 4H bar starts.
    Pass `df` (M15 OHLC) to bypass disk load (used by tests)."""
    d = df if df is not None else _load_raw_m15(sym)
    parts = []
    for rule in lower:
        tf = d if rule == "15min" else _resample_ohlc(d, rule)
        feats = _tf_features(tf)
        # last completed lower-TF value within each 4H bin → fair for predicting the NEXT 4H bar
        agg = feats.resample(base, label="left", closed="left").last()
        agg.columns = [f"{TAG[rule]}_{col}" for col in agg.columns]
        parts.append(agg)
    return pd.concat(parts, axis=1)


def _load_raw_m15(sym: str) -> pd.DataFrame:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    df = _load_raw(root / "data" / f"{sym}_M15_long.csv")
    return df[["open", "high", "low", "close", "tick_volume", "spread"]].copy()


def _load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c or c in {"date", "datetime"}), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()
