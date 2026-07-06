"""
Composable rule-based signal engine — Phase 2.

Each Rule maps a feature row to a score in [-1.0, +1.0]:
  +1.0  →  strong buy signal
  -1.0  →  strong sell signal
   0.0  →  neutral

SignalCombiner aggregates multiple rules into a single probability vector
[P_buy, P_hold, P_sell] — the same interface that all later ML models will use.
This makes rule-based and ML-based signals drop-in replacements for each other.

Usage:
    from src.features.indicators import compute, sma, rsi, atr, bollinger_pct_b
    from src.signals.rule_engine import SignalCombiner, ma_crossover_rule, rsi_rule, \
                                bb_reversion_rule, trend_filter_rule

    df = compute(df, [
        ("sma_9",   sma, {"period": 9}),
        ("sma_21",  sma, {"period": 21}),
        ("rsi_14",  rsi, {"period": 14}),
        ("bb_pct",  bollinger_pct_b, {}),
        ("atr_14",  atr, {"period": 14}),
    ])

    combiner = SignalCombiner(threshold=0.55)
    combiner.add(ma_crossover_rule("sma_9", "sma_21"),  weight=2.0, name="ma_cross")
    combiner.add(rsi_rule("rsi_14"),                     weight=1.5, name="rsi")
    combiner.add(bb_reversion_rule("bb_pct"),            weight=1.0, name="bb_rev")

    proba = combiner.predict_proba(df)  # → array([P_buy, P_hold, P_sell])
    print(f"P_buy={proba[0]:.2f}  P_hold={proba[1]:.2f}  P_sell={proba[2]:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ─── Rule primitive ───────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A single trading rule.

    fn       — callable(row: pd.Series) → float in [-1, 1]
    weight   — importance relative to other rules
    name     — human-readable label for logging
    """
    fn:     Callable[[pd.Series], float]
    weight: float = 1.0
    name:   str   = ""

    def evaluate(self, row: pd.Series) -> float:
        try:
            score = float(self.fn(row))
            return max(-1.0, min(1.0, score)) * self.weight
        except Exception:
            return 0.0


# ─── Signal combiner ──────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Aggregates Rule scores into [P_buy, P_hold, P_sell] probabilities.

    The combined score s ∈ [-1, 1] maps to probabilities as:
        P_buy  = max(0, s)
        P_sell = max(0, -s)
        P_hold = 1 - |s|

    This guarantees P_buy + P_hold + P_sell = 1 for all s.
    """

    def __init__(self, threshold: float = 0.55):
        """
        threshold — minimum P_buy or P_sell required to act on the signal.
        The combiner itself does not gate trades; it just reports the threshold
        so callers can use it.
        """
        self.rules:     list[Rule] = []
        self.threshold: float      = threshold

    # ── Building the combiner ─────────────────────────────────────────────

    def add(
        self,
        fn:     Callable[[pd.Series], float],
        weight: float = 1.0,
        name:   str   = "",
    ) -> "SignalCombiner":
        """Add a rule. Returns self for chaining."""
        self.rules.append(Rule(fn=fn, weight=weight, name=name or f"rule_{len(self.rules)}"))
        return self

    def remove(self, name: str) -> None:
        self.rules = [r for r in self.rules if r.name != name]

    def list_rules(self) -> list[str]:
        return [f"{r.name} (w={r.weight})" for r in self.rules]

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Evaluate all rules on the latest bar of df.

        Returns np.array([P_buy, P_hold, P_sell]).
        """
        if not self.rules or df.empty:
            return np.array([0.0, 1.0, 0.0])

        row = df.iloc[-1]
        total_weight = sum(abs(r.weight) for r in self.rules) or 1.0
        raw_score = sum(r.evaluate(row) for r in self.rules)
        score = max(-1.0, min(1.0, raw_score / total_weight))

        p_buy  = max(0.0, score)
        p_sell = max(0.0, -score)
        p_hold = 1.0 - abs(score)
        return np.array([p_buy, p_hold, p_sell])

    def signal_str(self, df: pd.DataFrame) -> str:
        """Human-readable signal string from latest bar."""
        proba = self.predict_proba(df)
        p_buy, p_hold, p_sell = proba
        if p_buy >= self.threshold:
            return f"BUY ({p_buy:.0%})"
        if p_sell >= self.threshold:
            return f"SELL ({p_sell:.0%})"
        return f"HOLD ({p_hold:.0%})"

    def rule_scores(self, df: pd.DataFrame) -> dict[str, float]:
        """Return each rule's raw score — useful for debugging."""
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {r.name: r.evaluate(row) for r in self.rules}


# ─── Pre-built rules ──────────────────────────────────────────────────────────
# These factories return callables (row → score) ready to pass into .add()

def ma_crossover_rule(fast_col: str, slow_col: str) -> Callable:
    """
    +1 if fast MA > slow MA (bullish alignment)
    -1 if fast MA < slow MA (bearish alignment)
    Magnitude scales by distance from cross.
    """
    def fn(row: pd.Series) -> float:
        fast = row.get(fast_col)
        slow = row.get(slow_col)
        if pd.isna(fast) or pd.isna(slow) or slow == 0:
            return 0.0
        return float(np.sign(fast - slow))
    return fn


def rsi_rule(
    rsi_col:    str   = "rsi_14",
    oversold:   float = 30.0,
    overbought: float = 70.0,
) -> Callable:
    """
    Score based on RSI extremes:
      RSI < oversold  →  +1 (reversal buy)
      RSI > overbought → -1 (reversal sell)
      Neutral zone    →  linear gradient
    """
    def fn(row: pd.Series) -> float:
        val = row.get(rsi_col)
        if pd.isna(val):
            return 0.0
        if val <= oversold:
            return 1.0
        if val >= overbought:
            return -1.0
        mid = (oversold + overbought) / 2
        return -(val - mid) / (overbought - mid)
    return fn


def macd_rule(
    macd_col: str = "macd",
    sig_col:  str = "macd_sig",
) -> Callable:
    """
    +1 if MACD > signal (momentum bullish)
    -1 if MACD < signal (momentum bearish)
    """
    def fn(row: pd.Series) -> float:
        m = row.get(macd_col)
        s = row.get(sig_col)
        if pd.isna(m) or pd.isna(s):
            return 0.0
        return float(np.sign(m - s))
    return fn


def bb_reversion_rule(bb_pct_col: str = "bb_pct") -> Callable:
    """
    Bollinger Band mean-reversion rule.
    %B < 0.2  →  buy (price near lower band)
    %B > 0.8  →  sell (price near upper band)
    Linear gradient between.
    """
    def fn(row: pd.Series) -> float:
        pct = row.get(bb_pct_col)
        if pd.isna(pct):
            return 0.0
        pct = max(0.0, min(1.0, pct))
        return -(pct - 0.5) * 2          # maps [0,1] → [+1, -1]
    return fn


def trend_filter_rule(
    adx_col:  str   = "adx",
    adx_min:  float = 25.0,
) -> Callable:
    """
    Neutral if market is not trending (ADX < adx_min).
    Returns 0 always — use this to suppress other signals via an external gate
    rather than as a directional signal.

    For gating: check adx before calling predict_proba.
    This stub returns 0 and is kept here as a reminder pattern.
    """
    def fn(row: pd.Series) -> float:
        return 0.0
    return fn


def price_vs_ma_rule(ma_col: str) -> Callable:
    """
    +1 if close > MA (uptrend)
    -1 if close < MA (downtrend)
    """
    def fn(row: pd.Series) -> float:
        close = row.get("close")
        ma    = row.get(ma_col)
        if pd.isna(close) or pd.isna(ma) or ma == 0:
            return 0.0
        return float(np.sign(close - ma))
    return fn


def stochastic_rule(
    k_col:      str   = "stoch_k",
    d_col:      str   = "stoch_d",
    oversold:   float = 20.0,
    overbought: float = 80.0,
) -> Callable:
    """
    Buy when %K crosses above %D in oversold zone.
    Sell when %K crosses below %D in overbought zone.
    Otherwise neutral.
    """
    def fn(row: pd.Series) -> float:
        k = row.get(k_col)
        d = row.get(d_col)
        if pd.isna(k) or pd.isna(d):
            return 0.0
        if k < oversold:
            return max(0.0, (k - d) / oversold)
        if k > overbought:
            return min(0.0, (k - d) / (100 - overbought))
        return 0.0
    return fn
