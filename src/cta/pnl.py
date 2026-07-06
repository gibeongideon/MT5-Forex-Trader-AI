"""Daily portfolio P&L — deliberately trivial and vectorized (bug-avoidance).

NO state machine, NO barriers, NO rolling hedge ratios (those caused the prior fake
+2.38 Sharpe). Daily mark-to-market: position formed at close t is held over the next
day's return via positions.shift(1) — the single anti-lookahead line.
"""
from __future__ import annotations
import pandas as pd


def portfolio_pnl(positions: pd.DataFrame, returns: pd.DataFrame,
                  spread_panel: pd.DataFrame, pip_sizes: pd.Series,
                  close_panel: pd.DataFrame) -> pd.DataFrame:
    """
    positions    : target position weight per instrument per day (cols=instruments).
    returns      : daily simple returns, same shape/index.
    spread_panel : mean daily spread in PIPS, same shape.
    pip_sizes    : Series indexed by instrument → pip size in price units.
    close_panel  : daily close price, same shape.
    Returns DataFrame[gross, net, turnover] indexed by date.
    """
    pos_lag  = positions.shift(1)                     # held over NEXT day's return
    gross    = (pos_lag * returns).sum(axis=1)
    turnover = (positions - positions.shift(1)).abs()
    # cost per instrument (return units) = traded weight × spread-as-fraction-of-price
    spread_frac = spread_panel.mul(pip_sizes, axis=1) / close_panel
    cost = (turnover * spread_frac).sum(axis=1)
    net = gross - cost
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover.sum(axis=1)})
