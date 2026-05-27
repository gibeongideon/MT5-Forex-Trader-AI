"""
Performance metrics for backtests and live trading.

All functions are pure — they take pandas Series / lists and return scalars.
`performance_report()` prints the full formatted summary used by every bot
and backtest in this project.

Conventions:
  equity  — pd.Series of balance values indexed by datetime
  pnl     — list or Series of per-trade P&L in pips or dollars
"""

import numpy as np
import pandas as pd


# ─── Core metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from an equity curve (daily resampling)."""
    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(periods_per_year))


def sortino_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    daily = equity.resample("D").last().pct_change(fill_method=None).dropna()
    downside = daily[daily < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(daily.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive percentage."""
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    n_days = (equity.index[-1] - equity.index[0]).days or 1
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
    annual_return = total_return * (365 / n_days)
    return float(annual_return / (mdd / 100))


def win_rate(pnl) -> float:
    """Fraction of winning trades (pnl > 0)."""
    pnl = list(pnl)
    if not pnl:
        return 0.0
    return sum(1 for p in pnl if p > 0) / len(pnl)


def profit_factor(pnl) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    wins = sum(p for p in pnl if p > 0)
    losses = abs(sum(p for p in pnl if p <= 0))
    return wins / losses if losses > 0 else float("inf")


def expectancy(pnl) -> float:
    """Average P&L per trade."""
    pnl = list(pnl)
    return float(np.mean(pnl)) if pnl else 0.0


# ─── Report ───────────────────────────────────────────────────────────────────

def performance_report(
    trades: list[dict] | pd.DataFrame,
    equity: pd.Series,
    initial_balance: float,
    title: str = "BACKTEST RESULTS",
    extra_params: dict = None,
) -> None:
    """
    Print a formatted performance report.

    trades — list of dicts or DataFrame with 'pnl_pips' and 'pnl_dollars' columns
    equity — pd.Series of balance indexed by datetime
    """
    if isinstance(trades, pd.DataFrame):
        pnl_pips = trades["pnl_pips"].tolist()
        pnl_dollars = trades["pnl_dollars"].tolist() if "pnl_dollars" in trades else pnl_pips
    else:
        def _get(t, key, default=0):
            return t.get(key, default) if isinstance(t, dict) else getattr(t, key, default)
        pnl_pips = [_get(t, "pnl_pips") for t in trades]
        pnl_dollars = [_get(t, "pnl_dollars") for t in trades]

    n = len(pnl_pips)
    if n == 0:
        print(f"\nNo trades generated.")
        return

    wins_pips = [p for p in pnl_pips if p > 0]
    loss_pips = [p for p in pnl_pips if p <= 0]

    final_bal = equity.iloc[-1]
    total_return = (final_bal - initial_balance) / initial_balance * 100

    sharpe  = sharpe_ratio(equity)
    sortino = sortino_ratio(equity)
    mdd     = max_drawdown(equity)
    calmar  = calmar_ratio(equity)
    wr      = win_rate(pnl_pips) * 100
    pf      = profit_factor(pnl_dollars)
    exp     = expectancy(pnl_dollars)

    w = 54
    sep = "─" * w
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")
    if extra_params:
        for k, v in extra_params.items():
            print(f"  {k:<18}: {v}")
        print(sep)
    print(f"  Period        : {equity.index[0].date()} → {equity.index[-1].date()}")
    print(sep)
    print(f"  Total trades  : {n}  ({len(wins_pips)}W / {len(loss_pips)}L)")
    print(f"  Win rate      : {wr:.1f}%")
    if wins_pips:
        print(f"  Avg win       : +{np.mean(wins_pips):.1f} pips")
    if loss_pips:
        print(f"  Avg loss      :  {np.mean(loss_pips):.1f} pips")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Expectancy    : ${exp:+.2f} per trade")
    print(sep)
    print(f"  Max drawdown  : {mdd:.1f}%")
    print(f"  Sharpe ratio  : {sharpe:.2f}")
    print(f"  Sortino ratio : {sortino:.2f}")
    print(f"  Calmar ratio  : {calmar:.2f}")
    print(sep)
    print(f"  Initial balance : ${initial_balance:>10,.2f}")
    print(f"  Final balance   : ${final_bal:>10,.2f}")
    print(f"  Total return    : {total_return:>+10.1f}%")
    print(f"{'═' * w}\n")
