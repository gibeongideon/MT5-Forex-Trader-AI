"""
Predictor — swappable inference endpoint.

The bot calls predictor.predict(ohlcv_df) and gets back a signal dict.
Swap the underlying model in config.yaml (llm_signal.model_id) or by passing
a different ModelInterface to __init__ — the bot code never changes.

Usage:
    p = Predictor()                          # uses LLMSignalModel from config
    p = Predictor(model=MyModel())           # custom model
    result = p.predict(ohlcv_df)
    # {"signal": "buy"|"sell"|"hold", "confidence": 0.72,
    #   "P_buy": 0.72, "P_hold": 0.18, "P_sell": 0.10}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class Predictor:
    """
    Thin wrapper around any ModelInterface that computes inline indicators
    (RSI-14, SMA-50) so the LLM tokenizer gets a rich context without
    needing the full feature pipeline.
    """

    def __init__(self, model=None, threshold: float = 0.45):
        """
        Parameters
        ----------
        model     : Any ModelInterface. Defaults to LLMSignalModel loaded from config.
        threshold : Min P_buy or P_sell to emit a non-hold signal.
        """
        self.threshold = threshold

        if model is not None:
            self._model = model
        else:
            self._model = self._build_default_model()

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, ohlcv: pd.DataFrame) -> dict:
        """
        Parameters
        ----------
        ohlcv : DataFrame with columns open, high, low, close (and optionally volume).
                Index must be DatetimeTZIndex or similar. Only last bar is scored
                but the full slice is used for indicator context.

        Returns
        -------
        dict with keys: signal, confidence, P_buy, P_hold, P_sell, timestamp
        """
        df = self._enrich(ohlcv)
        proba = self._model.predict_proba(df)

        # predict_proba returns (N,3) or (3,) — always take last bar
        if proba.ndim == 2:
            p = proba[-1]
        else:
            p = proba

        P_buy, P_hold, P_sell = float(p[0]), float(p[1]), float(p[2])
        confidence = max(P_buy, P_sell)

        if P_buy >= self.threshold and P_buy > P_sell:
            signal = "buy"
        elif P_sell >= self.threshold and P_sell > P_buy:
            signal = "sell"
        else:
            signal = "hold"

        return {
            "signal":     signal,
            "confidence": round(confidence, 4),
            "P_buy":      round(P_buy, 4),
            "P_hold":     round(P_hold, 4),
            "P_sell":     round(P_sell, 4),
            "timestamp":  ohlcv.index[-1],
        }

    # ── Indicator enrichment ───────────────────────────────────────────────────

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rsi_14 and sma_50 columns so the LLM gets richer context."""
        df = df.copy()
        if "rsi_14" not in df.columns:
            df["rsi_14"] = _rsi(df["close"], 14)
        if "sma_50" not in df.columns:
            df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
        return df

    # ── Model loading ──────────────────────────────────────────────────────────

    def _build_default_model(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        llm_cfg = cfg.get("llm_signal", {})

        from src.models.llm_signal_model import LLMSignalModel
        return LLMSignalModel(
            model_id       = llm_cfg.get("model_id",       "claude-haiku-4-5-20251001"),
            n_context_bars = llm_cfg.get("n_context_bars", 32),
            cache_bars     = llm_cfg.get("cache_bars",     4),
            cache_path     = llm_cfg.get("cache_path",     "data/models/llm_cache.parquet"),
            provider       = llm_cfg.get("provider",       "claude_api"),
        )


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss   = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
