"""Phase 8 A/B/C comparison: fixed vs tiered vs tiered+ATR risk sizing."""
import warnings; warnings.filterwarnings("ignore")
from src.data_loader import load_data
from src.features.pipeline import build_features
from src.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

print("Loading data...")
df = load_data()
X, y = build_features(df)
prices = df[["open", "high", "low", "close"]].loc[X.index]
print(f"X shape: {X.shape}")

def wf(bc):
    v = WalkForwardValidator(verbose=True)
    return v.run(X, y, prices, WalkForwardConfig(
        model_type="xgboost", window_type="expanding",
        train_days=180, test_days=30, backtest=bc))

# A: Fixed 1% (baseline)
print("\n=== A: Fixed 1% risk ===")
res_a = wf(BacktestConfig(threshold=0.40, sl_pips=30.0, tp_pips=60.0,
    pip_size=0.0001, spread_pips=1.0, commission_pips=0.0,
    max_slippage_pips=0.0, initial_balance=10000.0, risk_pct=0.01))
res_a.report(title="FIXED 1% RISK (Phase 7 baseline)")

# B: Confidence-tiered
print("\n=== B: Confidence-tiered risk ===")
res_b = wf(BacktestConfig(threshold=0.40, sl_pips=30.0, tp_pips=60.0,
    pip_size=0.0001, spread_pips=1.0, commission_pips=0.0,
    max_slippage_pips=0.0, initial_balance=10000.0, risk_pct=0.01,
    risk_manager=RiskManager(RiskConfig())))
res_b.report(title="CONFIDENCE-TIERED RISK (Phase 8)")

# C: Tiered + ATR stop
print("\n=== C: Tiered risk + ATR stop ===")
res_c = wf(BacktestConfig(threshold=0.40, sl_pips=30.0, tp_pips=60.0,
    pip_size=0.0001, spread_pips=1.0, commission_pips=0.0,
    max_slippage_pips=0.0, initial_balance=10000.0, risk_pct=0.01,
    risk_manager=RiskManager(RiskConfig(
        use_atr_stop=True, atr_multiplier=1.5,
        min_sl_pips=15.0, max_sl_pips=60.0))))
res_c.report(title="TIERED RISK + ATR STOP (Phase 8)")

# Summary
print("\n" + "═" * 54)
print("  PHASE 8 COMPARISON SUMMARY")
print("═" * 54)
for label, r in [("A Fixed 1%  ", res_a), ("B Tiered    ", res_b), ("C Tiered+ATR", res_c)]:
    print(f"  {label}: Sharpe={r.sharpe:+.2f}  DD={r.max_drawdown:.1%}"
          f"  Return={r.total_return:.1%}  Trades={r.total_trades}")
print("═" * 54)
