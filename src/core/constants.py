"""Shared constants across the trading system."""

# ── Pip sizes by instrument ───────────────────────────────────────────────────
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
    "XAUUSD": 0.01,   # Gold
}

DEFAULT_PIP_SIZE = 0.0001

# ── Label convention (must match feature_pipeline.py) ────────────────────────
LABEL_BUY  =  1
LABEL_HOLD =  0
LABEL_SELL = -1

# ── Probability output order (must match ModelInterface contract) ─────────────
# predict_proba() always returns [P_buy, P_hold, P_sell]
IDX_BUY  = 0
IDX_HOLD = 1
IDX_SELL = 2

# ── Default risk parameters ───────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.55   # below this → no trade
DEFAULT_RISK_PCT              = 0.01  # 1% per trade
DEFAULT_SL_PIPS               = 30.0
DEFAULT_TP_PIPS               = 60.0
