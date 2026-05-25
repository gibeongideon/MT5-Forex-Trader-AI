"""
Monte Carlo Validation — Phase 7.

Tests whether a backtest's performance is statistically real or just lucky
ordering of trades.

Method:
  1. Take the list of completed trades (pnl_dollars per trade).
  2. Shuffle the trade order randomly N times (default 1000).
  3. Rebuild the equity curve for each shuffle and compute Sharpe.
  4. Report the 5th / 50th / 95th percentile Sharpe across all shuffles.
  5. Compare the original Sharpe to the shuffle distribution.

If the original Sharpe is near the 50th percentile of shuffles → the strategy
has no alpha beyond the individual trade outcomes (trade order doesn't matter).
If the original Sharpe is near the 95th percentile → the strategy exploits
sequencing/timing, which is a red flag for overfitting.
Typically we want:
  - 5th-percentile Sharpe > 0.5  (even unlucky orderings are profitable)
  - Original Sharpe ≈ 50th percentile (robust, not order-dependent)

Usage:
    from src.monte_carlo import run_monte_carlo
    result = run_monte_carlo(trades, initial_balance=10000, n_simulations=1000)
    result.report()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    n_simulations:   int
    original_sharpe: float
    sharpe_p05:      float
    sharpe_p25:      float
    sharpe_p50:      float
    sharpe_p75:      float
    sharpe_p95:      float
    # Fraction of simulations that beat the original Sharpe
    # High value → original is not particularly special (good — robust)
    # Very low value → original depends heavily on trade order (bad)
    pct_beating_original: float
    all_sharpes: np.ndarray  # shape (n_simulations,)

    def report(self) -> None:
        w = 54
        print(f"\n{'═' * w}")
        print(f"  MONTE CARLO VALIDATION ({self.n_simulations:,} simulations)")
        print(f"{'═' * w}")
        print(f"  Original Sharpe     : {self.original_sharpe:>7.3f}")
        print(f"{'─' * w}")
        print(f"  Shuffle 5th pct     : {self.sharpe_p05:>7.3f}  ← even bad luck")
        print(f"  Shuffle 25th pct    : {self.sharpe_p25:>7.3f}")
        print(f"  Shuffle median      : {self.sharpe_p50:>7.3f}  ← typical random order")
        print(f"  Shuffle 75th pct    : {self.sharpe_p75:>7.3f}")
        print(f"  Shuffle 95th pct    : {self.sharpe_p95:>7.3f}  ← lucky order")
        print(f"{'─' * w}")
        pct = self.pct_beating_original * 100
        print(f"  Shuffles beating original: {pct:.1f}%")
        print()
        # Interpretation
        if self.sharpe_p05 > 0.5:
            print("  PASS: 5th-percentile Sharpe > 0.5 — strategy is robust.")
        else:
            print("  WARN: 5th-percentile Sharpe ≤ 0.5 — edge may not survive bad luck.")
        if self.original_sharpe > self.sharpe_p95:
            print("  WARN: Original > 95th pct — performance may be order-dependent (overfitting risk).")
        elif self.original_sharpe > self.sharpe_p50:
            print("  INFO: Original above median shuffle — slight positive timing effect.")
        else:
            print("  INFO: Original near shuffle median — performance not order-sensitive (good).")
        print(f"{'═' * w}\n")

    def histogram(self, bins: int = 20, width: int = 50) -> None:
        """Print a text-mode histogram of shuffle Sharpe distribution."""
        arr  = self.all_sharpes
        lo   = float(arr.min())
        hi   = float(arr.max())
        step = (hi - lo) / bins if hi != lo else 1.0
        print(f"\n  Shuffle Sharpe distribution  (n={self.n_simulations:,})")
        print(f"  Range: [{lo:.3f}, {hi:.3f}]")
        print()
        for b in range(bins):
            left  = lo + b * step
            right = left + step
            count = int(((arr >= left) & (arr < right)).sum())
            bar   = "█" * int(count / len(arr) * width * bins)
            marker = " ← original" if left <= self.original_sharpe < right else ""
            print(f"  {left:>6.3f} │ {bar}{marker}")
        print()


# ── Main function ──────────────────────────────────────────────────────────────

def run_monte_carlo(
    trades:          list[dict],
    initial_balance: float = 10_000.0,
    n_simulations:   int   = 1_000,
    seed:            int   = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on a trade list.

    Parameters
    ----------
    trades          : list of trade dicts — must have 'pnl_dollars' key
    initial_balance : starting account balance
    n_simulations   : number of random trade-order shuffles
    seed            : reproducibility seed

    Returns
    -------
    MonteCarloResult with percentile Sharpe statistics.
    """
    if not trades:
        raise ValueError("No trades provided for Monte Carlo simulation.")

    pnl = np.array([t["pnl_dollars"] for t in trades], dtype=float)
    n   = len(pnl)

    if n < 10:
        raise ValueError(f"Too few trades for Monte Carlo ({n}). Need at least 10.")

    # Original equity curve and Sharpe
    orig_sharpe = _sharpe_from_pnl(pnl, initial_balance)

    rng          = np.random.default_rng(seed)
    shuffle_sharpes = np.empty(n_simulations, dtype=float)

    for i in range(n_simulations):
        shuffled = rng.permutation(pnl)
        shuffle_sharpes[i] = _sharpe_from_pnl(shuffled, initial_balance)

    pct_beating = float((shuffle_sharpes > orig_sharpe).mean())

    return MonteCarloResult(
        n_simulations=n_simulations,
        original_sharpe=orig_sharpe,
        sharpe_p05=float(np.percentile(shuffle_sharpes, 5)),
        sharpe_p25=float(np.percentile(shuffle_sharpes, 25)),
        sharpe_p50=float(np.percentile(shuffle_sharpes, 50)),
        sharpe_p75=float(np.percentile(shuffle_sharpes, 75)),
        sharpe_p95=float(np.percentile(shuffle_sharpes, 95)),
        pct_beating_original=pct_beating,
        all_sharpes=shuffle_sharpes,
    )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _sharpe_from_pnl(pnl: np.ndarray, initial_balance: float) -> float:
    """Build a daily equity series from per-trade P&L and compute Sharpe."""
    # Assign one trade per synthetic "day" — this is a simplification but
    # sufficient for comparing shuffle distributions.
    equity_vals = np.concatenate([[initial_balance], initial_balance + np.cumsum(pnl)])
    # Use trade number as proxy for time (index in business days from 2024-01-01)
    idx = pd.bdate_range("2024-01-01", periods=len(equity_vals), freq="B")
    equity = pd.Series(equity_vals, index=idx)
    return sharpe_ratio(equity)
