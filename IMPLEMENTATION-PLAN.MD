# MT5 Scalable Trading Bot — 10-Phase Implementation Plan

## Context

The goal is to build a scalable, probability-first MT5 trading system that grows from a simple random trader to a full enterprise-grade ML+LLM ensemble. The architecture must support swapping prediction models like plug-ins, stacking one model's output as another's input, and emitting calibrated probabilities rather than hard buy/sell decisions. The existing codebase already has a solid foundation: MT5 connector, base bot class, example MA bot, an AI bot (Claude API), a backtester, and a data downloader.

The system should be designed so that the trading engine never changes — only the signal/prediction layer evolves.

---

## Architecture Philosophy

```
OHLCV Data
    ↓
Feature Pipeline (composable indicators)
    ↓
Signal Layer (pluggable — any model, rule, or LLM)
    ↓   → P(buy), P(sell), P(hold)   ← probability, not absolute
Risk Engine (position sizing by confidence)
    ↓
MT5 Execution Layer (existing mt5_connector.py)
    ↓
Trade Journal / Performance Tracking
```

**Key invariants:**
- Every signal/model must output `predict_proba(features) → [P_buy, P_hold, P_sell]`
- Confidence threshold gates execution (default: P > 0.60 to trade)
- Models are swappable without touching the execution or risk layer
- Any model's output can be added as a feature input to another model

---

## Phase 1 — Random Baseline + Logging Infrastructure

**Goal:** Establish a random-entry benchmark (expected ≈ 0 or negative Sharpe). Proves there is no edge yet — anything better in later phases is real signal.

**What to build:**
- `src/random_bot.py` — inherits `BotBase`, enters long/short with configurable probability per tick
- `src/trade_journal.py` — SQLite-backed trade logger (entry, exit, P&L, model used, confidence score, reason)
- `src/metrics.py` — Sharpe ratio, win rate, max drawdown, profit factor, expectancy
- Update `config.yaml` with `random_bot` section (entry_prob, seed)

**Tools/Libraries:** sqlite3 (stdlib), pandas, numpy

**Success metric:** Backtest shows Sharpe < 0.3, confirming no inherent edge

---

## Phase 2 — Composable Technical Indicator Library + Rule-Based Signals

**Goal:** Build a reusable, lookahead-safe indicator library and a rule engine that produces a simple signal score.

**What to build:**
- `src/indicators.py` — indicator registry:
  - SMA(n), EMA(n), RSI(14), MACD(12,26,9), Bollinger Bands(20,2σ), ATR(14), Stochastic(14,3,3)
  - Each function takes a DataFrame and returns a Series, no internal state
  - `compute(df, indicator_list)` returns enriched DataFrame
- `src/rule_engine.py` — composable rule system:
  - Rules are lambdas on the feature DataFrame: `lambda row: row['rsi'] < 30`
  - `SignalCombiner` votes across rules → emits `signal_score ∈ [-1, +1]`
  - Convert score to `[P_buy, P_hold, P_sell]` via softmax-like normalization
- Update `src/backtest.py` to support any signal source (not just MA crossover)

**Tools/Libraries:** pandas, numpy, TA-Lib (optional, fallback to hand-rolled)

**Success metric:** Rule-based signals beat random baseline by ≥ 20% on Sharpe

---

## Phase 3 — Feature Engineering Pipeline

**Goal:** Transform raw OHLCV + indicator outputs into an ML-ready feature matrix, with strict no-lookahead guarantees.

**What to build:**
- `src/feature_pipeline.py`:
  - `FeaturePipeline` class — registers feature generators by name
  - Lag features (close[t-1], close[t-2], returns[t-1..t-5])
  - Rolling statistics (std, skew, kurtosis over 10/20/50 bars)
  - Multi-timeframe support: fetch M15, H1, H4 bars → merge on time index
  - Normalization: `fit_scaler(train_df)` → `transform(df)` (never fit on test)
  - `validate_no_lookahead(df)` — asserts all features are shift(1) or older
- `scripts/build_features.py` — offline feature matrix builder for a date range
- Store feature matrices as Parquet files under `data/features/`

**Tools/Libraries:** pandas, numpy, scikit-learn (StandardScaler, MinMaxScaler), pyarrow

**Success metric:** Feature matrix passes lookahead validator; train/test split is clean

---

## Phase 4 — First ML Model: XGBoost with Calibrated Probabilities

**Goal:** Replace rule-based signals with a trained classifier that outputs calibrated probabilities.

**What to build:**
- `src/models/xgboost_model.py`:
  - Implements `ModelInterface` (see Phase 5)
  - `train(X_train, y_train)` → fits XGBoost classifier
  - `predict_proba(X)` → `[P_buy, P_hold, P_sell]`
  - Wraps in `CalibratedClassifierCV(method='isotonic')` for probability accuracy
  - `save(path)` / `load(path)` via joblib
- `scripts/train_model.py` — CLI to train on date range, save model artifact
- `scripts/walk_forward.py` — walk-forward validator:
  - Expanding window: train on [start → t], test on [t → t+30 days]
  - Roll forward by 30 days, collect out-of-sample predictions
  - Report aggregated Sharpe, win rate, drawdown across all folds
- Label generation: `y = 1 (buy)` if next-N-bar return > threshold, `y = -1 (sell)`, `y = 0 (hold)`

**Tools/Libraries:** xgboost, scikit-learn, joblib, matplotlib (feature importance plot)

**Success metric:** Out-of-sample Sharpe ≥ 1.0 on walk-forward

---

## Phase 5 — Pluggable Model Registry

**Goal:** Decouple the trading engine from any specific model. Swap models at runtime or via config.

**What to build:**
- `src/model_interface.py` — abstract base:
  ```python
  class ModelInterface(ABC):
      def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...
      def train(self, X, y): ...
      def save(self, path: str): ...
      def load(self, path: str): ...
      def metadata(self) -> dict: ...  # name, version, trained_on, features_used
  ```
- `src/model_registry.py` — runtime registry:
  - `register(name, model_instance)`
  - `get(name) → ModelInterface`
  - `set_active(name)` — hot-swap active model
  - `list_models()` with metadata
- `src/models/lightgbm_model.py` — LightGBM implementation of `ModelInterface`
- `src/models/random_forest_model.py` — Random Forest implementation
- Update `BotBase` to call `registry.get_active().predict_proba(features)` instead of hard-coded logic
- Update `config.yaml` with `active_model` key

**Tools/Libraries:** lightgbm, scikit-learn

**Success metric:** Swap from XGBoost to LightGBM by changing one config key, no other code changes

---

## Phase 6 — Signal Stacking & Meta-Learning

**Goal:** Combine outputs of multiple models + rule signals into one final probability via a trained meta-learner. One model's probability score becomes another model's input feature.

**What to build:**
- `src/ensemble.py`:
  - `Ensemble` — takes a list of `ModelInterface` instances
  - Calls each model's `predict_proba()`, concatenates outputs as feature columns
  - Adds rule-based signal scores as additional features
  - Meta-learner: `LogisticRegression` or lightweight `LightGBM` trained on stacked outputs
  - Output: single `[P_buy, P_hold, P_sell]` vector
  - Train meta-learner only on **validation folds** of base models (not their training data) to prevent leakage
- Stacking layers:
  - Layer 0: Raw features → [XGBoost, LightGBM, RandomForest, RuleEngine]
  - Layer 1: Layer-0 probabilities → MetaLearner → final probability
- `src/models/neural_net_model.py` — simple LSTM or MLP using PyTorch/Keras

**Tools/Libraries:** scikit-learn (StackingClassifier), lightgbm, torch or keras

**Success metric:** Ensemble out-of-sample Sharpe ≥ layer-0 best model by ≥ 15%

---

## Phase 7 — Robust Backtesting & Walk-Forward Validation Framework

**Goal:** Replace the simple MA backtester with a rigorous, realistic simulation engine that prevents overfitting.

**What to build:**
- `src/backtester.py` — event-driven backtester:
  - Simulates bar-by-bar with proper transaction costs (spread, commission, slippage)
  - Supports any `ModelInterface` as signal source
  - Tracks equity curve, drawdown series
  - Outputs: trade list, equity curve, full performance report
- `src/walk_forward.py` — full walk-forward engine:
  - Expanding or sliding window configuration
  - Retrain model + meta-learner at each fold boundary
  - Aggregate out-of-sample equity curve
- `src/monte_carlo.py` — Monte Carlo validation:
  - Randomly shuffle trade order 1000× to check if performance is luck
  - Report 5th/95th percentile Sharpe across shuffles
- Regime filter: detect trending vs. ranging using ADX(14); optionally suppress signals in low-ADX regimes

**Tools/Libraries:** pandas, numpy, matplotlib, scipy (for statistics)

**Success metric:** Monte Carlo 5th-percentile Sharpe > 0.5; walk-forward equity curve consistently positive

---

## Phase 8 — Intelligent Risk Management

**Goal:** Size every position according to prediction confidence, not a fixed 1% rule.

**What to build:**
- `src/risk_manager.py`:
  - `confidence_to_risk(P, base_risk=0.01) → risk_pct`:
    - `P ≥ 0.75` → 2.0% risk
    - `P ≥ 0.65` → 1.5% risk
    - `P ≥ 0.55` → 0.75% risk
    - `P < 0.55` → skip trade
  - Fractional Kelly sizing: `f = (P×R − (1−P)) / R` where R = TP/SL ratio
  - ATR-based dynamic stop: `sl_pips = ATR(14) × 1.5`
  - Portfolio limits: max 3% total open risk at any time
  - Drawdown throttle: if drawdown > 10%, reduce all position sizes by 50%
- Update `BotBase` to call `RiskManager.size(symbol, P, direction)` for every order

**Tools/Libraries:** numpy (risk math only)

**Success metric:** Max drawdown reduced vs. fixed-size baseline; Sharpe improves or holds

---

## Phase 9 — LLM Integration as a Probability Signal

**Goal:** Treat the LLM (Claude) as one signal in the ensemble — outputting probability scores, not hard decisions. Its output becomes a feature for the meta-learner.

**What to build:**
- `src/models/llm_signal_model.py` — implements `ModelInterface`:
  - Calls Claude API with market context (OHLCV summary, key indicator values)
  - **Prompt redesign**: ask Claude to output `{"P_buy": 0.72, "P_hold": 0.18, "P_sell": 0.10, "reasoning": "..."}` instead of categorical action
  - Add anthropic prompt caching (`cache_control: ephemeral`) on system prompt + static context to cut API costs by ~80%
  - Rate limit: call LLM once per N bars, cache result in between
  - `predict_proba()` returns the three probabilities directly
- `src/models/llm_news_model.py` (optional):
  - Fetch recent news headlines via NewsAPI or similar
  - Ask LLM to rate bullish/bearish sentiment → probability score
- Register `llm_signal_model` in Model Registry; add to ensemble Layer 0
- Meta-learner will learn how much weight to give LLM vs. ML models based on historical performance

**Tools/Libraries:** anthropic SDK (already in requirements.txt), requests (news)

**Success metric:** Adding LLM signal to ensemble improves out-of-sample Sharpe vs. ML-only baseline

---

## Phase 10 — Production Enterprise System

**Goal:** Harden for live trading with monitoring, alerting, automated retraining, and paper trading mode.

**What to build:**
- `src/paper_trader.py` — shadow-execution mode:
  - Intercepts `open_position()` calls, logs them but never sends to MT5
  - Tracks virtual P&L against real market prices
  - Switch via `config.yaml: mode: paper | live`
- `src/api/server.py` — FastAPI dashboard backend:
  - `/metrics` — live Sharpe, drawdown, open positions
  - `/trades` — trade history with model confidence and SHAP explanations
  - `/model/swap` — hot-swap active model via API call
  - `/model/retrain` — trigger manual retraining
- `src/explainer.py` — SHAP explainability:
  - For each trade, compute SHAP values showing which features drove the decision
  - Log top-3 features to trade journal
- `src/drift_detector.py` — model drift monitoring:
  - Compare recent out-of-sample accuracy vs. baseline
  - If accuracy drops > 15%, trigger retraining alert
- `src/alerter.py` — notifications:
  - Email (SMTP) or Slack webhook on: large drawdown, model drift, daily summary
- `scripts/retrain_schedule.py` — automated weekly retraining:
  - Fetch new data, rebuild feature matrix, retrain all models, run walk-forward
  - If new model beats current by > 5% Sharpe → auto-promote
- Multi-symbol support: run separate signal pipelines per symbol, shared risk manager

**Tools/Libraries:** FastAPI, uvicorn, shap, smtplib, schedule or APScheduler

**Success metric:** System runs 30 days in paper mode with positive Sharpe before any live capital

---

## Critical Files (Existing — Reuse These)

| File | Role |
|---|---|
| [src/mt5_connector.py](src/mt5_connector.py) | All MT5 I/O — never change the interface |
| [src/bot_base.py](src/bot_base.py) | Lifecycle, daily loss limit, helpers — extend, don't rewrite |
| [src/ai_bot.py](src/ai_bot.py) | Refactor into `llm_signal_model.py` (Phase 9) |
| [src/backtest.py](src/backtest.py) | Extend into full backtester (Phase 7) |
| [scripts/download_data.py](scripts/download_data.py) | Data acquisition — already complete |
| [config.yaml](config.yaml) | Add sections per phase; keep backward compatible |
| [data/EURUSD_M15.csv](data/EURUSD_M15.csv) | Training data — Phase 4+ |

---

## New File Map (by Phase)

```
src/
  random_bot.py          (Phase 1)
  trade_journal.py       (Phase 1)
  metrics.py             (Phase 1)
  indicators.py          (Phase 2)
  rule_engine.py         (Phase 2)
  feature_pipeline.py    (Phase 3)
  model_interface.py     (Phase 5)
  model_registry.py      (Phase 5)
  ensemble.py            (Phase 6)
  backtester.py          (Phase 7)
  walk_forward.py        (Phase 7)
  monte_carlo.py         (Phase 7)
  risk_manager.py        (Phase 8)
  paper_trader.py        (Phase 10)
  explainer.py           (Phase 10)
  drift_detector.py      (Phase 10)
  alerter.py             (Phase 10)
  models/
    xgboost_model.py     (Phase 4)
    lightgbm_model.py    (Phase 5)
    random_forest_model.py (Phase 5)
    neural_net_model.py  (Phase 6)
    llm_signal_model.py  (Phase 9)
    llm_news_model.py    (Phase 9)
  api/
    server.py            (Phase 10)

scripts/
  build_features.py      (Phase 3)
  train_model.py         (Phase 4)
  walk_forward.py        (Phase 4)
  retrain_schedule.py    (Phase 10)

data/
  features/              (Phase 3 — Parquet files)
  models/                (Phase 4 — saved model artifacts)
```

---

## Recommended Libraries to Add (requirements.txt)

```
xgboost>=2.0
lightgbm>=4.0
scikit-learn>=1.4
joblib>=1.3
pyarrow>=14.0        # Parquet
shap>=0.44           # Explainability (Phase 10)
fastapi>=0.110       # Dashboard (Phase 10)
uvicorn>=0.29        # (Phase 10)
torch>=2.2           # Neural net (Phase 6)
```

---

## Verification Approach (Each Phase)

1. **Phase 1–2**: Run `python src/random_bot.py --backtest` → confirm negative/flat Sharpe baseline; rule bot beats it
2. **Phase 3**: Run `scripts/build_features.py --validate` → zero lookahead violations reported
3. **Phase 4**: Run `scripts/walk_forward.py` → 10 folds, out-of-sample Sharpe printed
4. **Phase 5**: Change `active_model: lightgbm` in config → bot uses different model with no code change
5. **Phase 6**: Run ensemble backtest → compare Sharpe vs. best single model
6. **Phase 7**: Run `src/monte_carlo.py` → 5th-percentile Sharpe > 0 confirms signal is real
7. **Phase 8**: Compare equity curves: fixed 1% risk vs. confidence-scaled → verify improved risk-adjusted returns
8. **Phase 9**: Compare ensemble-with-LLM vs. ensemble-without-LLM on held-out data
9. **Phase 10**: Run `src/paper_trader.py` for 30 days → review dashboard metrics before live capital

---

## Starting Point (Phase 1)

Begin with `src/random_bot.py` and `src/trade_journal.py`. The random bot gives us a clean baseline and the journal gives us the data infrastructure every subsequent phase depends on. All phases build on the ones before.
