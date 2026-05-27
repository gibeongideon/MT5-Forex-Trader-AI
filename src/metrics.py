# Compatibility shim — module moved to src/evaluation/metrics.py
from src.evaluation.metrics import *  # noqa: F401, F403
from src.evaluation.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    win_rate, profit_factor, expectancy, performance_report,
)
