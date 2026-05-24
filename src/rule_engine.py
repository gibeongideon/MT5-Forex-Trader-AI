# Compatibility shim — module moved to src/signals/rule_engine.py
from src.signals.rule_engine import *  # noqa: F401, F403
from src.signals.rule_engine import (
    Rule, SignalCombiner,
    ma_crossover_rule, rsi_rule, macd_rule,
    bb_reversion_rule, price_vs_ma_rule, stochastic_rule,
)
