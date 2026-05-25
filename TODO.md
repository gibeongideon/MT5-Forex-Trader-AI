# MT5 Trading Bot — TODO Tracker

> Cross-phase task list. Phases completed here stay for reference.
> Mark tasks with ✅ when done, 🔄 when in progress, ⬜ when not started.

---

## Phase 1 — Random Baseline + Logging Infrastructure ✅

- [x] `src/random_bot.py` — random entry bot with `--backtest` mode
- [x] `src/trade_journal.py` — SQLite trade logger
- [x] `src/metrics.py` — Sharpe, Sortino, max drawdown, profit factor, expectancy

---

## Phase 2 — Composable Indicator Library + Rule Engine ✅

- [x] `src/features/indicators.py` — SMA, EMA, RSI, MACD, BB, ATR, Stochastic, ADX, OBV
- [x] `src/signals/rule_engine.py` — composable rule combiner → `[P_buy, P_hold, P_sell]`
- [x] `src/rule_bot.py` — rule-based signal bot

---

## Phase 3 — Feature Engineering Pipeline ✅

- [x] `src/features/feature_pipeline.py` — 31 features, shift(1) lookahead guard, StandardScaler
- [x] `scripts/build_features.py` — CLI: builds Parquet feature matrix + labels
- [x] `data/features/EURUSD_M15_features.parquet` — 49,892 rows × 31 features
- [x] Lookahead validation: PASSED (0 violations)

---

## Phase 4 — XGBoost with Calibrated Probabilities ✅

- [x] `src/model_interface.py` — abstract base: `predict_proba`, `train`, `save`, `load`, `metadata`
- [x] `src/models/xgboost_model.py` — CalibratedClassifierCV(isotonic), label remapping
- [x] `scripts/train_model.py` — CLI train + save
- [x] `scripts/walk_forward.py` — Phase 4/5 walk-forward prototype (Sharpe 1.34 out-of-sample)

---

## Phase 5 — Pluggable Model Registry ✅

- [x] `src/model_registry.py` — register / get / set_active / from_config
- [x] `src/models/lightgbm_model.py`
- [x] `src/models/random_forest_model.py`
- [x] Model swap via `config.yaml: active_model` — zero code changes
- [x] Restructured `src/` into `core/`, `data/`, `features/`, `signals/`, `models/`

---

## Phase 6 — Signal Stacking & Meta-Learning ✅

- [x] `src/models/catboost_model.py` — CatBoost 3-class, well-calibrated
- [x] `src/models/lstm_model.py` — 2-layer PyTorch LSTM, seq_len=20
- [x] `src/ensemble.py` — two-layer stacking: OOF base predictions → LightGBM meta-learner
- [x] `scripts/train_ensemble.py` — CLI to train + evaluate ensemble
- [x] Ensemble log-loss 1.0869 (beats random baseline 1.099)
- [x] LightGBM meta chosen: lower log-loss, 2× more confident bars vs logistic

---

## Phase 7 — Robust Backtesting & Walk-Forward Validation ✅

- [x] `src/backtester.py` — event-driven bar-by-bar simulator
  - [x] Spread, commission, slippage transaction costs
  - [x] ADX regime filter (suppress signals in ranging market)
  - [x] `BacktestConfig` dataclass, `BacktestResult` with `.report()`
- [x] `src/walk_forward.py` — full walk-forward engine
  - [x] Expanding and sliding window modes
  - [x] Ensemble support (per-fold retrain with disk cache)
  - [x] Uses `Backtester` for simulation (unified cost model)
  - [x] `WalkForwardResult` with fold table + aggregate report
- [x] `src/monte_carlo.py` — Monte Carlo trade shuffler
  - [x] 1000-shuffle default, reproducible seed
  - [x] 5th / 25th / 50th / 75th / 95th percentile Sharpe
  - [x] Text-mode histogram
- [x] `config.yaml` Phase 7 sections: `backtester`, `walk_forward`, `monte_carlo`
- [x] Walk-forward verified: Sharpe 0.72 (with costs), DD 14.6%, 126 trades
- [x] Monte Carlo: edge is not order-dependent (53% of shuffles beat original)

---

## Phase 8 — Intelligent Risk Management ✅

- [x] `src/risk_manager.py`
  - [x] `confidence_to_risk(P) → risk_pct` tiered sizing
  - [x] Fractional Kelly sizing (off by default)
  - [x] ATR-based dynamic stop: `sl = ATR(14) × 1.5` (off by default)
  - [x] Portfolio cap: max 3% total open risk
  - [x] Drawdown throttle: if DD > 10%, reduce sizes by 50%
- [x] `src/backtester.py` updated — `risk_manager` field, per-trade `risk_pct`
- [x] `BotBase.risk_sized_lot()` — tiered sizing for live bots
- [x] Fixed bug: `src/walk_forward.py` was not propagating `risk_manager` to fold configs
- [x] Verified: tiered risk cuts max DD from 14.6% → 7.5% with zero Sharpe cost (both +0.72)

---

## Phase 9 — LLM Integration as Probability Signal ⬜

- [ ] `src/models/llm_signal_model.py`
  - [ ] Implements `ModelInterface`
  - [ ] Claude API with `{"P_buy", "P_hold", "P_sell", "reasoning"}` JSON output
  - [ ] Anthropic prompt caching (`cache_control: ephemeral`) on system prompt
  - [ ] Rate limit: call once per N bars, cache result in between
- [ ] `src/models/llm_news_model.py` (optional — news sentiment)
- [ ] Register in model registry; add to ensemble Layer 0
- [ ] Compare ensemble-with-LLM vs ensemble-without-LLM Sharpe

---

## Phase 10 — Production Enterprise System ⬜

- [ ] `src/paper_trader.py` — shadow execution, virtual P&L, switch via config `mode: paper | live`
- [ ] `src/api/server.py` — FastAPI dashboard: `/metrics`, `/trades`, `/model/swap`, `/model/retrain`
- [ ] `src/explainer.py` — SHAP values per trade, top-3 features logged
- [ ] `src/drift_detector.py` — accuracy vs baseline, alert if drop > 15%
- [ ] `src/alerter.py` — email / Slack webhook for drawdown, drift, daily summary
- [ ] `scripts/retrain_schedule.py` — weekly auto-retrain + auto-promote if Sharpe +5%
- [ ] Multi-symbol support (separate signal pipelines, shared risk manager)
- [ ] 30-day paper mode before any live capital

---

## Housekeeping / Cross-Phase ⬜

- [ ] Add `data/models/wf_cache/` and `data/models/*.joblib` to `.gitignore`
- [ ] Add `data/features/*.parquet` to `.gitignore` (if not already)
- [ ] Write `tests/unit/test_backtester.py`
- [ ] Write `tests/unit/test_walk_forward.py`
- [ ] Write `tests/unit/test_monte_carlo.py`
- [ ] Write `tests/e2e/test_phase7_e2e.py`
