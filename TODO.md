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

## Phase 9 — LLM Integration as Probability Signal ✅ (partial)

### Completed

- [x] `src/features/bar_tokenizer.py` — OHLCV → DIR_SIZE_WICK tokens (vocab=47); `fit/encode_sequence/encode_ids/context_prefix`
- [x] `src/models/llm_signal_model.py` — Claude API signal model
  - [x] Implements `ModelInterface` — plugs into registry + ensemble with zero wiring changes
  - [x] Two providers: `claude_cli` (terminal auth) and `claude_api` (ANTHROPIC_API_KEY)
  - [x] Anthropic prompt caching (`cache_control: ephemeral`) — ~80% token cost reduction
  - [x] Disk-backed parquet cache (DatetimeIndex → P_buy/P_hold/P_sell) — no API calls during walk-forward
  - [x] `provider` saved/loaded in joblib artifact; switchable via `config.yaml`
- [x] `src/models/bar_lm_model.py` — local tiny Transformer (~200k params) on token sequences (offline, no API)
- [x] `scripts/precompute_llm_signals.py` — offline cache builder; `--dry-run`, `--stride`, `--start` resume
- [x] `scripts/train_bar_lm.py` — CLI to train + evaluate local bar language model
- [x] `src/model_registry.py` — `llm_signal` and `bar_lm` registered; reads config for provider/model_id
- [x] `config.yaml` — `llm_signal`, `bar_lm`, `bar_tokenizer` sections added
- [x] Verified: LLM Sharpe -0.629 vs XGBoost (2yr trained) -0.497 on Apr–May 2026 test window

### Findings (May 2026 experiment)

- LLM signal outperforms ensemble Sharpe by +0.86 on the 3-month out-of-sample window
- LLM alone beats a simple XGB+LLM equal-weight blend — blending only helps when both signals
  are independently strong; XGBoost needs its full 2-year training window to contribute
- Sequential bar-pattern context (32-bar token history) carries signal the tree models miss

---

## Improvements to Explore

### 9-A — Full Dataset LLM Cache + Proper Ensemble Blend  ⬜  ← NEXT

- [ ] Precompute LLM signals for all 49,892 bars at stride=4 (~$1.50, ~60 min)
      `python scripts/precompute_llm_signals.py --provider claude_api`
- [ ] Add `llm_signal` to `config.yaml` ensemble `base_models` list
- [ ] Retrain ensemble (XGBoost + LightGBM + CatBoost + RF + LLM) on full dataset
- [ ] Run walk-forward: compare `ensemble_with_llm` vs `ensemble_without_llm`
- [ ] Expected gain: meta-learner learns to use LLM for sequential patterns + trees for cross-sectional features

### 9-B — Bar Language Model (Local, No API Cost)  ⬜

- [ ] Train `BarLMModel` on full dataset: `python scripts/train_bar_lm.py --epochs 30`
- [ ] Compare Sharpe vs `llm_signal` — can a local 200k-param model match Claude?
- [ ] If competitive, use as cheaper replacement for live trading (zero API cost per signal)
- [ ] Try larger architecture: d_model=64, n_layers=6, seq_len=64

### 9-C — LLM Signal Quality Improvements  ⬜

- [ ] Try `claude-sonnet-4-6` instead of `claude-haiku-4-5-20251001` for richer reasoning
      (10× more expensive but may generate higher-quality probability estimates)
- [ ] Extend context window from 32 to 64 bars — more pattern history per prompt
- [ ] Add volume context to the token (e.g., HIGH_VOL / LOW_VOL flag per bar)
- [ ] Include session context in prompt (London/NY/Asia overlap indicator)
- [ ] Experiment with multi-timeframe tokens: M15 + H1 + H4 combined in one prompt

### 9-D — Adaptive Blend / Dynamic Weighting  ⬜

- [ ] Replace equal-weight blend with confidence-weighted blend:
      `w_llm = P_llm.max() / (P_llm.max() + P_xgb.max())` — weight by relative certainty
- [ ] Rolling correlation between LLM signal and XGBoost signal — high correlation → reduce LLM weight
- [ ] Regime-dependent blending: use LLM more in trending markets (ADX > 25), trees in ranging

### 9-E — Cost Optimisation  ⬜

- [ ] Stride=8 vs stride=4 comparison — halves API cost, check if Sharpe degrades
- [ ] Cache invalidation: re-call API only when market regime shifts (ADX crossover / volatility spike)
- [ ] Track actual API spend per trading session in trade journal

### 10-A — Production / Dashboard  ⬜

- [ ] `src/paper_trader.py` — shadow execution, virtual P&L, switch via `config.yaml: mode: paper|live`
- [ ] `src/api/server.py` — FastAPI: `/metrics`, `/trades`, `/model/swap`, `/model/retrain`
- [ ] `src/explainer.py` — SHAP values per trade, top-3 features logged
- [ ] `src/drift_detector.py` — accuracy vs baseline, alert if drop > 15%
- [ ] `src/alerter.py` — email / Slack webhook for drawdown, drift, daily summary
- [ ] `scripts/retrain_schedule.py` — weekly auto-retrain + auto-promote if Sharpe improves ≥5%
- [ ] Multi-symbol support (separate pipelines, shared risk manager)
- [ ] 30-day paper mode before any live capital

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
