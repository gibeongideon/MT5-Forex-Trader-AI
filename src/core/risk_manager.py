"""
Intelligent Risk Management — Phase 8.

Replaces fixed 1% risk-per-trade with confidence-scaled position sizing.
The higher the model's confidence, the more capital is risked on that trade.

Sizing layers (applied in order):
  1. Confidence tier   — maps P_buy/P_sell → base risk %
  2. Fractional Kelly  — optional; caps position using Kelly criterion
  3. ATR stop          — dynamic SL width replaces fixed pips
  4. Drawdown throttle — halves all risk when equity drawdown exceeds threshold
  5. Portfolio cap      — skips new trades when total open risk already at max

Usage:
    from src.core.risk_manager import RiskManager, RiskConfig

    rm = RiskManager(RiskConfig())
    sizing = rm.size(confidence=0.62, balance=10000, sl_pips=25, tp_pips=50)
    if not sizing["skip"]:
        lot = calc_lot(symbol, sizing["sl_pips"], sizing["risk_pct"])

For live bots, use BotBase.risk_sized_lot() which calls this automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    # ── Confidence tiers ──────────────────────────────────────────────────
    # List of (min_confidence, risk_pct) in descending confidence order.
    # The first tier whose threshold is met is used.
    tiers: list = field(default_factory=lambda: [
        (0.75, 0.020),   # P ≥ 0.75 → 2.0 % risk
        (0.65, 0.015),   # P ≥ 0.65 → 1.5 % risk
        (0.55, 0.0075),  # P ≥ 0.55 → 0.75% risk
        (0.40, 0.005),   # P ≥ 0.40 → 0.5 % risk  (for low-threshold models)
    ])
    min_confidence: float = 0.40    # below this → skip trade entirely

    # ── Fractional Kelly ─────────────────────────────────────────────────
    use_kelly: bool = False          # off by default; enable for Kelly sizing
    kelly_fraction: float = 0.25    # use 25 % of full Kelly (conservative)
    kelly_max_risk: float = 0.03    # hard cap even with high Kelly

    # ── ATR-based dynamic stop ────────────────────────────────────────────
    use_atr_stop: bool = False       # off by default; enable for dynamic SL
    atr_multiplier: float = 1.5     # sl_pips = ATR_pips × atr_multiplier
    atr_col: str = "atr_14"         # feature column (price units, not pips)
    min_sl_pips: float = 15.0       # floor on dynamic SL
    max_sl_pips: float = 60.0       # ceiling on dynamic SL

    # ── Portfolio cap ─────────────────────────────────────────────────────
    max_portfolio_risk: float = 0.03   # 3 % max total open risk at once

    # ── Drawdown throttle ─────────────────────────────────────────────────
    drawdown_threshold: float = 0.10   # throttle kicks in above 10 % drawdown
    drawdown_throttle:  float = 0.50   # multiply all risk by this factor


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class SizingResult:
    skip:        bool    # True → do not open this trade
    risk_pct:    float   # fraction of balance to risk
    sl_pips:     float   # stop-loss in pips (may be ATR-adjusted)
    tp_pips:     float   # take-profit in pips (scaled to match RR ratio)
    dollar_risk: float   # balance × risk_pct
    reason:      str     # "tier" | "kelly" | "throttle" | "skip"

    def to_dict(self) -> dict:
        return self.__dict__


# ── RiskManager ────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Stateless risk calculator.  Call size() per trade signal.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ── Public API ─────────────────────────────────────────────────────────

    def size(
        self,
        confidence:    float,
        balance:       float,
        sl_pips:       float,
        tp_pips:       float,
        drawdown_pct:  float = 0.0,
        open_risk_pct: float = 0.0,   # fraction of balance already at risk
        atr_value:     float = 0.0,   # raw ATR (price units); 0 = not provided
        pip_size:      float = 0.0001,
    ) -> SizingResult:
        """
        Compute position sizing for one trade.

        Parameters
        ----------
        confidence    : model's probability for the chosen direction
        balance       : current account balance in dollars
        sl_pips       : baseline stop-loss pips (used when ATR stop is off)
        tp_pips       : baseline take-profit pips
        drawdown_pct  : current peak-to-trough drawdown (0.0–1.0)
        open_risk_pct : total risk fraction of balance already in open trades
        atr_value     : ATR in price units (e.g. 0.0012 for EURUSD)
        pip_size      : pip size for the instrument (default 0.0001)

        Returns
        -------
        SizingResult — check .skip before placing any order.
        """
        cfg = self.config

        # ── 1. Confidence gate ────────────────────────────────────────────
        if confidence < cfg.min_confidence:
            return SizingResult(skip=True, risk_pct=0, sl_pips=sl_pips,
                                tp_pips=tp_pips, dollar_risk=0, reason="skip")

        # ── 2. Confidence tier ────────────────────────────────────────────
        risk_pct = self.confidence_to_risk(confidence)

        # ── 3. Fractional Kelly cap ───────────────────────────────────────
        reason = "tier"
        if cfg.use_kelly:
            rr = tp_pips / sl_pips if sl_pips > 0 else 2.0
            k = self.kelly_fraction(confidence, rr)
            if 0 < k < risk_pct:
                risk_pct = k
                reason = "kelly"
            risk_pct = min(risk_pct, cfg.kelly_max_risk)

        # ── 4. ATR-based dynamic stop ─────────────────────────────────────
        effective_sl = sl_pips
        effective_tp = tp_pips
        if cfg.use_atr_stop and atr_value > 0 and pip_size > 0:
            atr_pips = atr_value / pip_size
            effective_sl = self.atr_stop(atr_pips)
            # Scale TP to maintain same RR ratio as the original config
            rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 2.0
            effective_tp = effective_sl * rr_ratio

        # ── 5. Drawdown throttle ──────────────────────────────────────────
        if drawdown_pct > cfg.drawdown_threshold:
            risk_pct *= cfg.drawdown_throttle
            reason = "throttle"

        # ── 6. Portfolio cap ──────────────────────────────────────────────
        remaining_capacity = cfg.max_portfolio_risk - open_risk_pct
        if remaining_capacity <= 0:
            return SizingResult(skip=True, risk_pct=0, sl_pips=effective_sl,
                                tp_pips=effective_tp, dollar_risk=0,
                                reason="portfolio_cap")
        risk_pct = min(risk_pct, remaining_capacity)

        dollar_risk = balance * risk_pct
        return SizingResult(
            skip=False,
            risk_pct=risk_pct,
            sl_pips=effective_sl,
            tp_pips=effective_tp,
            dollar_risk=dollar_risk,
            reason=reason,
        )

    def confidence_to_risk(self, confidence: float) -> float:
        """Map confidence → risk_pct via the tier table."""
        cfg = self.config
        if confidence < cfg.min_confidence:
            return 0.0
        # Walk tiers from highest to lowest
        for threshold, risk in sorted(cfg.tiers, key=lambda t: -t[0]):
            if confidence >= threshold:
                return risk
        # Below all tiers but above min_confidence → smallest tier
        if cfg.tiers:
            return min(r for _, r in cfg.tiers)
        return 0.01

    def kelly_fraction(self, confidence: float, rr_ratio: float) -> float:
        """
        Fractional Kelly: f* = kelly_fraction × (P×R − (1−P)) / R

        Returns 0 when edge is negative (Kelly would short).
        """
        if rr_ratio <= 0:
            return 0.0
        p = confidence
        full_kelly = (p * rr_ratio - (1.0 - p)) / rr_ratio
        return max(0.0, self.config.kelly_fraction * full_kelly)

    def atr_stop(self, atr_pips: float) -> float:
        """Convert ATR (in pips) to a dynamic stop-loss distance (pips)."""
        cfg = self.config
        sl = atr_pips * cfg.atr_multiplier
        return max(cfg.min_sl_pips, min(cfg.max_sl_pips, sl))

    def check_portfolio_headroom(self, open_risk_pct: float) -> float:
        """Return remaining risk capacity (0.0 if portfolio is full)."""
        return max(0.0, self.config.max_portfolio_risk - open_risk_pct)

    def describe(self, confidence: float, balance: float,
                 sl_pips: float, tp_pips: float) -> None:
        """Print a human-readable sizing breakdown (useful for debugging)."""
        result = self.size(confidence, balance, sl_pips, tp_pips)
        w = 44
        print(f"\n{'─' * w}")
        print(f"  Risk sizing  (conf={confidence:.0%}, bal=${balance:,.0f})")
        print(f"{'─' * w}")
        print(f"  Skip?         : {'YES' if result.skip else 'NO'}")
        if not result.skip:
            print(f"  Risk %        : {result.risk_pct:.2%}")
            print(f"  Dollar risk   : ${result.dollar_risk:,.2f}")
            print(f"  SL / TP       : {result.sl_pips:.1f}p / {result.tp_pips:.1f}p")
            print(f"  Reason        : {result.reason}")
        print(f"{'─' * w}\n")
