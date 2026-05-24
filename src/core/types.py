"""Shared dataclasses used across all layers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Bar:
    """A single OHLCV bar."""
    time:   object
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float = 0.0
    symbol: str   = ""


@dataclass
class Signal:
    """Output of any ModelInterface.predict_proba() call."""
    p_buy:      float
    p_hold:     float
    p_sell:     float
    confidence: float = 0.0       # max(p_buy, p_sell)
    direction:  str   = "hold"    # "buy" | "sell" | "hold"
    model:      str   = ""

    def __post_init__(self):
        self.confidence = max(self.p_buy, self.p_sell)
        if self.p_buy >= self.p_sell:
            self.direction = "buy" if self.p_buy > self.p_hold else "hold"
        else:
            self.direction = "sell" if self.p_sell > self.p_hold else "hold"

    @classmethod
    def from_array(cls, proba: np.ndarray, model: str = "") -> "Signal":
        """Build a Signal from a [P_buy, P_hold, P_sell] array."""
        return cls(p_buy=float(proba[0]), p_hold=float(proba[1]), p_sell=float(proba[2]), model=model)
