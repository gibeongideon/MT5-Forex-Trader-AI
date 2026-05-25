"""Phase 8 A/B/C comparison: fixed vs tiered vs tiered+ATR risk sizing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from src.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

FEATURES  = "data/features/EURUSD_M15_features.parquet"
LABELS    = "data/features/EURUSD_M15_labels.parquet"
PRICES    = "data/EURUSD_M15.csv"

print("Loading data...")
X      = pd.read_parquet(FEATURES)
y      = pd.read_parquet(LABELS)["label"]
prices = pd.read_csv(PRICES, index_col="time")
prices.index = pd.to_datetime(prices.index)
print(f"X: {X.shape}  prices: {len(prices):,}")

def wf(bc, verbose=True):
    v = WalkForwardValidator(verbose=verbose)
    return v.run(X, y, prices, WalkForwardConfig(
        model_type="xgboost", window_type="expanding",
        train_days=180, test_days=30, backtest=bc))

BASE = dict(threshold=0.40, sl_pips=30.0, tp_pips=60.0, pip_size=0.0001,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.0,
            initial_balance=10000.0, risk_pct=0.01, use_regime_filter=False)

# A: Fixed 1%
print("\n=== A: Fixed 1% risk ===")
res_a = wf(BacktestConfig(**BASE))
res_a.report(title="FIXED 1% RISK (Phase 7 baseline)")

# B: Confidence-tiered
print("\n=== B: Confidence-tiered risk ===")
res_b = wf(BacktestConfig(**BASE, risk_manager=RiskManager(RiskConfig())))
res_b.report(title="CONFIDENCE-TIERED RISK (Phase 8)")

# C: Tiered + ATR stop
print("\n=== C: Tiered risk + ATR stop ===")
res_c = wf(BacktestConfig(**BASE, risk_manager=RiskManager(RiskConfig(
    use_atr_stop=True, atr_multiplier=1.5, min_sl_pips=15.0, max_sl_pips=60.0))))
res_c.report(title="TIERED RISK + ATR STOP (Phase 8)")

# Summary
print("\n" + "═" * 54)
print("  PHASE 8 COMPARISON SUMMARY")
print("═" * 54)
for lbl, r in [("A Fixed 1%  ", res_a), ("B Tiered    ", res_b), ("C Tiered+ATR", res_c)]:
    eq = r.equity
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 0 else 0.0
    print(f"  {lbl}: Sharpe={r.sharpe:+.2f}  DD={r.drawdown:.1%}"
          f"  Return={ret:+.1%}  Trades={len(r.trades)}")
print("═" * 54)
