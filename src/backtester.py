"""
Event-Driven Backtester — Phase 7.

Simulates a bar-by-bar trading session with realistic transaction costs:
  - Spread (applied at fill — widens entry price)
  - Commission (flat pips per round-trip)
  - Slippage (random within [0, max_slippage] at fill)

Supports any ModelInterface as the signal source.  The backtester never
touches model training — it only calls predict_proba(X) and simulates
the resulting trade through price history.

Optional ADX regime filter:  when enabled, signals are suppressed on bars
where ADX < adx_threshold (ranging market, no directional edge).

Usage (standalone):
    from src.backtester import Backtester, BacktestConfig
    cfg = BacktestConfig(threshold=0.40, spread_pips=1.0, use_regime_filter=True)
    result = Backtester().run(model, X, prices, cfg)
    result.report()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics import performance_report, sharpe_ratio, max_drawdown
from src.risk_manager import RiskManager, RiskConfig


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Signal
    threshold: float = 0.55         # min P_buy / P_sell to open a trade
    # Instrument
    pip_size: float = 0.0001        # EURUSD default
    sl_pips: float = 30.0
    tp_pips: float = 60.0
    # Cost model
    spread_pips: float = 1.0        # added to entry price (buy: pays spread, sell: pays spread)
    commission_pips: float = 0.5    # flat round-trip cost per trade (deducted at close)
    max_slippage_pips: float = 0.3  # uniform random slippage per fill
    # Risk (fixed)
    initial_balance: float = 10_000.0
    risk_pct: float = 0.01          # fraction of balance risked per trade (used when risk_manager is None)
    # Phase 8: intelligent risk management (optional — set risk_manager to enable)
    risk_manager: Optional[RiskManager] = None   # if set, overrides fixed risk_pct
    # Regime filter
    use_regime_filter: bool = False
    adx_threshold: float = 20.0     # below this ADX → skip signal (ranging market)
    adx_col: str = "adx_14"         # column name in feature DataFrame


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    fold:              int
    direction:         str    # "buy" | "sell"
    entry_time:        object
    entry_price:       float  # after spread/slippage
    exit_time:         object
    exit_price:        float
    sl:                float  # price level
    tp:                float  # price level
    sl_pips:           float  # effective SL (may differ from config if ATR stop)
    tp_pips:           float  # effective TP
    pnl_pips:          float  # before cost deductions
    pnl_dollars:       float  # after all costs, based on risk sizing
    confidence:        float
    exit_reason:       str    # "sl" | "tp" | "end"
    cost_pips:         float  # spread + commission + slippage
    risk_pct:          float = 0.01   # actual risk fraction used for this trade

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BacktestResult:
    trades:  list[dict]
    equity:  pd.Series
    config:  BacktestConfig

    @property
    def sharpe(self) -> float:
        return sharpe_ratio(self.equity)

    @property
    def drawdown(self) -> float:
        return max_drawdown(self.equity)

    def report(self, title: str = "BACKTEST RESULTS", extra: Optional[dict] = None) -> None:
        if not self.trades:
            print("No trades generated.")
            return
        params = {
            "Threshold":   f"{self.config.threshold:.0%}",
            "SL / TP":     f"{self.config.sl_pips}p / {self.config.tp_pips}p",
            "Spread":      f"{self.config.spread_pips}p",
            "Commission":  f"{self.config.commission_pips}p",
            "Max slippage":f"{self.config.max_slippage_pips}p",
            "Regime filter": ("ON (ADX < " + str(self.config.adx_threshold) + ")")
                             if self.config.use_regime_filter else "OFF",
        }
        if extra:
            params.update(extra)
        performance_report(
            self.trades, self.equity,
            self.config.initial_balance,
            title=title, extra_params=params,
        )


# ── Backtester ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    Runs a bar-by-bar simulation.

    Parameters
    ----------
    seed : int | None
        Random seed for slippage sampling (reproducibility).
    """

    def __init__(self, seed: Optional[int] = 42):
        self._rng = random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        model,                    # ModelInterface — already trained
        X:      pd.DataFrame,     # feature matrix (same index as prices)
        prices: pd.DataFrame,     # OHLCV with columns: open, high, low, close
        config: BacktestConfig,
        fold:   int = 0,
    ) -> BacktestResult:
        """
        Run a full simulation over X / prices.

        Returns BacktestResult with trades and equity curve.
        """
        proba = model.predict_proba(X)              # (n, 3) or (3,)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        trades, equity = self._simulate(
            index=X.index,
            proba=proba,
            prices=prices,
            X=X,
            config=config,
            fold=fold,
        )
        return BacktestResult(
            trades=[t.to_dict() for t in trades],
            equity=equity,
            config=config,
        )

    # ── Internal simulation ────────────────────────────────────────────────────

    def _simulate(
        self,
        index:   pd.Index,
        proba:   np.ndarray,
        prices:  pd.DataFrame,
        X:       pd.DataFrame,
        config:  BacktestConfig,
        fold:    int,
    ) -> tuple[list[Trade], pd.Series]:

        cfg        = config
        balance    = cfg.initial_balance
        equity_pts = []
        trades     = []
        open_trade: Optional[Trade] = None

        for i, ts in enumerate(index):
            if ts not in prices.index:
                equity_pts.append(balance)
                continue

            row   = prices.loc[ts]
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])

            # ── Check SL / TP on open trade ───────────────────────────────────
            if open_trade is not None:
                hit = self._check_exit(open_trade, high, low)
                if hit:
                    self._close_trade(open_trade, hit, ts)
                    pnl_dollars = self._pnl_dollars(open_trade, balance)
                    open_trade.pnl_dollars = pnl_dollars
                    balance += pnl_dollars
                    trades.append(open_trade)
                    open_trade = None

            # ── New signal ────────────────────────────────────────────────────
            if open_trade is None:
                p     = proba[i]
                p_buy = float(p[0])
                p_sell= float(p[2])

                # Regime filter: skip if ADX below threshold
                if cfg.use_regime_filter and cfg.adx_col in X.columns:
                    adx_val = float(X.iloc[i][cfg.adx_col])
                    if adx_val < cfg.adx_threshold:
                        equity_pts.append(balance)
                        continue

                direction = None
                confidence = 0.0
                if p_buy >= cfg.threshold:
                    direction, confidence = "buy", p_buy
                elif p_sell >= cfg.threshold:
                    direction, confidence = "sell", p_sell

                if direction is not None:
                    # ── Phase 8: dynamic risk sizing ──────────────────────────
                    if cfg.risk_manager is not None:
                        rm  = cfg.risk_manager
                        # Current drawdown for throttle check
                        peak         = max(equity_pts + [balance]) if equity_pts else balance
                        drawdown_pct = max(0.0, (peak - balance) / peak)
                        # ATR value (price units) from features if ATR stop enabled
                        atr_val = 0.0
                        if rm.config.use_atr_stop:
                            atr_col = rm.config.atr_col
                            if atr_col in X.columns:
                                atr_val = float(X.iloc[i][atr_col])
                        sizing = rm.size(
                            confidence=confidence,
                            balance=balance,
                            sl_pips=cfg.sl_pips,
                            tp_pips=cfg.tp_pips,
                            drawdown_pct=drawdown_pct,
                            atr_value=atr_val,
                            pip_size=cfg.pip_size,
                        )
                        if sizing.skip:
                            equity_pts.append(balance)
                            continue
                        eff_sl_pips  = sizing.sl_pips
                        eff_tp_pips  = sizing.tp_pips
                        eff_risk_pct = sizing.risk_pct
                    else:
                        eff_sl_pips  = cfg.sl_pips
                        eff_tp_pips  = cfg.tp_pips
                        eff_risk_pct = cfg.risk_pct

                    slippage_pts = self._rng.uniform(0, cfg.max_slippage_pips) * cfg.pip_size
                    spread_pts   = cfg.spread_pips * cfg.pip_size
                    sl_pts       = eff_sl_pips * cfg.pip_size
                    tp_pts       = eff_tp_pips * cfg.pip_size

                    if direction == "buy":
                        fill_price = close + spread_pts + slippage_pts
                        sl = fill_price - sl_pts
                        tp = fill_price + tp_pts
                    else:
                        fill_price = close - spread_pts - slippage_pts
                        sl = fill_price + sl_pts
                        tp = fill_price - tp_pts

                    total_cost_pips = (
                        cfg.spread_pips
                        + cfg.commission_pips
                        + slippage_pts / cfg.pip_size
                    )

                    open_trade = Trade(
                        fold=fold,
                        direction=direction,
                        entry_time=ts,
                        entry_price=fill_price,
                        exit_time=None,
                        exit_price=0.0,
                        sl=sl,
                        tp=tp,
                        sl_pips=eff_sl_pips,
                        tp_pips=eff_tp_pips,
                        pnl_pips=0.0,
                        pnl_dollars=0.0,
                        confidence=confidence,
                        exit_reason="",
                        cost_pips=total_cost_pips,
                        risk_pct=eff_risk_pct,
                    )

            equity_pts.append(balance)

        # Force-close any open trade at end of window
        if open_trade is not None and len(index) > 0:
            last_ts = index[-1]
            last_price = (
                float(prices.loc[last_ts, "close"])
                if last_ts in prices.index
                else open_trade.entry_price
            )
            raw_pips = (last_price - open_trade.entry_price) / cfg.pip_size
            if open_trade.direction == "sell":
                raw_pips = -raw_pips
            open_trade.exit_time   = last_ts
            open_trade.exit_price  = last_price
            open_trade.pnl_pips    = raw_pips - open_trade.cost_pips
            open_trade.exit_reason = "end"
            pnl_dollars = self._pnl_dollars(open_trade, balance)
            open_trade.pnl_dollars = pnl_dollars
            balance += pnl_dollars
            trades.append(open_trade)

        equity = pd.Series(equity_pts, index=index[: len(equity_pts)])
        return trades, equity

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _check_exit(self, t: Trade, high: float, low: float) -> Optional[str]:
        """Return 'sl', 'tp', or None."""
        if t.direction == "buy":
            if low <= t.sl:
                return "sl"
            if high >= t.tp:
                return "tp"
        else:
            if high >= t.sl:
                return "sl"
            if low <= t.tp:
                return "tp"
        return None

    def _close_trade(self, t: Trade, exit_reason: str, ts) -> None:
        """Close a trade at SL or TP, using the trade's own effective pips."""
        if exit_reason == "sl":
            raw_pips   = -t.sl_pips
            exit_price = t.sl
        else:  # tp
            raw_pips   = t.tp_pips
            exit_price = t.tp
        t.exit_time   = ts
        t.exit_price  = exit_price
        t.pnl_pips    = raw_pips - t.cost_pips
        t.exit_reason = exit_reason

    def _pnl_dollars(self, t: Trade, balance: float) -> float:
        """Convert pnl_pips → dollars using the trade's own risk_pct and sl_pips."""
        sl   = t.sl_pips  if t.sl_pips  > 0 else 30.0
        risk = t.risk_pct if t.risk_pct > 0 else 0.01
        dollar_per_pip = (balance * risk) / sl
        return t.pnl_pips * dollar_per_pip
