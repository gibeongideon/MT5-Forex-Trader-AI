# MT5 Trading Bot — Build Progress

> **Purpose:** Session handoff document. A new Claude session should read this + `IMPLEMENTATION-PLAN.MD` before doing any work. Do NOT re-explore files that are already described here.

Last updated: 2026-05-24

---

## Current Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Random Baseline + Logging Infrastructure | ✅ COMPLETE |
| 2 | Composable Indicator Library + Rule Engine | ✅ COMPLETE |
| 3 | Feature Engineering Pipeline | ✅ COMPLETE |
| 4 | XGBoost with Calibrated Probabilities | ✅ COMPLETE |
| 5 | Pluggable Model Registry | ✅ COMPLETE |
| 6 | Signal Stacking & Meta-Learning | ⬜ NOT STARTED |
| 7 | Robust Backtesting & Walk-Forward | ⬜ NOT STARTED |
| 8 | Intelligent Risk Management | ⬜ NOT STARTED |
| 9 | LLM Integration as Probability Signal | ⬜ NOT STARTED |
| 10 | Production Enterprise System | ⬜ NOT STARTED |

**Next task:** Start Phase 6 — Signal Stacking & Meta-Learning

---

## Environment

- **OS:** Ubuntu (Linux)
- **Conda env:** `envmt5` (Python 3.10) — always activate before running
- **MT5 bridge:** Wine + rpyc on localhost:18812 — needed only for live bots
- **Data file:** `data/EURUSD_M15.csv` — 50,000 bars, 2024-05-13 → 2026-05-18
- **Run prefix:** `conda run -n envmt5 python ...` or activate first

---

## Architecture Decisions (Do Not Change These)

1. **Every signal/model must output `[P_buy, P_hold, P_sell]`** — probability triple, sums to 1. Never output hard buy/sell/hold strings from the signal layer.

2. **Confidence threshold gates trades** — default 0.55 (55%). Below threshold → hold. This is set per-bot in `config.yaml`.

3. **The trading engine never changes** — only the signal/prediction layer evolves. `BotBase` and `MT5Connector` are stable; new bots extend `BotBase`.

4. **Models are swappable via `config.yaml: active_model`** — no code changes needed to switch models (implemented in Phase 5).

5. **Any model's probability output can be a feature input to another model** — this is the stacking pattern (Phase 6).

6. **No lookahead in features** — all rolling computations use only past data. Feature pipeline validates this explicitly (Phase 3).

7. **SL/TP in pips, not price** — all risk math uses pip distances. `MT5Connector.calc_lot_size()` handles currency conversion.

---

## Files Built So Far

### Pre-existing (do not modify interfaces)

| File | What it does |
|------|-------------|
| [src/mt5_connector.py](src/mt5_connector.py) | MT5 connection via Wine/rpyc. Singleton. Never change the public interface. |
| [src/bot_base.py](src/bot_base.py) | Abstract base class. Lifecycle, daily-loss limit, helpers (`buy`, `sell`, `rates`, `calc_lot`). All bots extend this. |
| [src/example_bot.py](src/example_bot.py) | MA(9)×MA(21) crossover reference bot. Working. |
| [src/ai_bot.py](src/ai_bot.py) | Claude API bot. Will be refactored into `src/models/llm_signal_model.py` in Phase 9. |
| [src/backtest.py](src/backtest.py) | MA-crossover backtester (original). Will be superseded by `src/backtester.py` in Phase 7. |
| [scripts/download_data.py](scripts/download_data.py) | Downloads OHLCV from MT5 → CSV. Already complete. |
| [config.yaml](config.yaml) | All bot config. Add new sections per phase; never remove existing sections. |
| [data/EURUSD_M15.csv](data/EURUSD_M15.csv) | Training data. 50k M15 bars. |

### Phase 1 — Built

| File | What it does |
|------|-------------|
| [src/trade_journal.py](src/trade_journal.py) | SQLite trade logger. `TradeJournal().record(trade_dict)` saves a completed trade. `get_trades()` returns DataFrame. DB at `data/trades.db`. |
| [src/metrics.py](src/metrics.py) | Pure performance functions: `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `expectancy`, `performance_report`. Works with Trade dataclasses OR dicts OR DataFrames. |
| [src/random_bot.py](src/random_bot.py) | Random entry bot. `--backtest` mode (standalone, no MT5). Live mode extends BotBase. Entry probability configurable. |

### Phase 5 — Built

| File | What it does |
|------|-------------|
| [src/model_registry.py](src/model_registry.py) | Singleton registry. `register(name, model)`, `get(name)`, `get_active()`, `set_active(name)`, `list_models()`. `from_config(config.yaml)` auto-loads all models listed under `models:` and sets active from `active_model:`. `_build_model(type_str)` factory creates any model by name. |
| [src/models/lightgbm_model.py](src/models/lightgbm_model.py) | LightGBM 3-class classifier. Same interface as XGBoostModel. `CalibratedClassifierCV(isotonic)`. LightGBM handles -1/0/1 labels natively (no remapping needed). Output reordered to [P_buy, P_hold, P_sell]. |
| [src/models/random_forest_model.py](src/models/random_forest_model.py) | Random Forest 3-class classifier. Same interface. `n_jobs=-1` for parallel tree building. Calibrated. |

**`config.yaml` changes:**
```yaml
active_model: xgboost   # ← change this one line to swap models, no code changes

models:
  xgboost:
    type: xgboost
    path: data/models/xgboost.joblib
  lightgbm:
    type: lightgbm
    path: data/models/lightgbm.joblib
  random_forest:
    type: random_forest
    path: data/models/random_forest.joblib
```

**`scripts/walk_forward.py` updated:** Now accepts `--model xgboost|lightgbm|random_forest`. Reads from `config.yaml: active_model` when no flag given. Model type resolved by `_resolve_model_type()`.

**Phase 5 verified results (walk-forward, threshold=0.40):** *(see Verified Backtest Results below)*

**Key commands:**
```bash
python scripts/walk_forward.py --model xgboost        # XGBoost (default)
python scripts/walk_forward.py --model lightgbm       # LightGBM
python scripts/walk_forward.py --model random_forest  # Random Forest
# OR: edit config.yaml active_model and run without --model flag
```

---

### Pre-Phase 5 Housekeeping — Complete

**New top-level docs:**

| File | What it does |
|------|-------------|
| [RULES_OF_BUILDING_THIS_APP.MD](RULES_OF_BUILDING_THIS_APP.MD) | Mandatory reading for every developer/AI before writing code. Session start protocol, architecture rules, code quality rules, environment rules, session end protocol. |
| [RANDOM_IDEAS.MD](RANDOM_IDEAS.MD) | Reference ideas doc. Already reviewed and incorporated — good ideas (folder structure, idempotent scripts, raw data immutability, split configs) are now in the architecture. |

**`src/` restructure** — files moved into logical subfolders, old paths kept as compatibility shims:

| New path | Old path (now a shim) | What it does |
|----------|-----------------------|-------------|
| [src/features/indicators.py](src/features/indicators.py) | [src/indicators.py](src/indicators.py) | Indicator library (real code now here) |
| [src/features/feature_pipeline.py](src/features/feature_pipeline.py) | [src/feature_pipeline.py](src/feature_pipeline.py) | Feature pipeline (real code now here) |
| [src/signals/rule_engine.py](src/signals/rule_engine.py) | [src/rule_engine.py](src/rule_engine.py) | Rule engine (real code now here) |
| [src/core/types.py](src/core/types.py) | NEW | Shared `Bar`, `Signal` dataclasses |
| [src/core/exceptions.py](src/core/exceptions.py) | NEW | `MT5BotError`, `ModelError`, `DataError`, `ConfigError`, `BrokerError` |
| [src/core/constants.py](src/core/constants.py) | NEW | Pip sizes, label values, default risk params |
| [src/data/schemas.py](src/data/schemas.py) | NEW | `validate_ohlcv()`, `normalize_columns()` |

**Walk-forward re-verified after restructure:** Sharpe 1.34, +14.2% — identical to pre-restructure.

---

### Phase 4 — Built

| File | What it does |
|------|-------------|
| [src/model_interface.py](src/model_interface.py) | Abstract base class every model must implement. Contract: `predict_proba(X)→[P_buy,P_hold,P_sell]`, `train()`, `save()`, `load()`, `metadata()`. Also provides `.signal()` and `.confidence()` helpers. |
| [src/models/__init__.py](src/models/__init__.py) | Package init for models directory. |
| [src/models/xgboost_model.py](src/models/xgboost_model.py) | XGBoost 3-class classifier wrapped in `CalibratedClassifierCV(isotonic)`. Internally remaps labels -1/0/1 → 0/1/2 for XGBoost, then reorders output back to `[P_buy, P_hold, P_sell]`. Save/load via joblib. |
| [scripts/train_model.py](scripts/train_model.py) | CLI: loads Parquet features, trains XGBoost, prints classification report + log-loss + confidence distribution, saves model to `data/models/xgboost.joblib`. |
| [scripts/walk_forward.py](scripts/walk_forward.py) | Expanding-window walk-forward validator. Retrains XGBoost at each fold boundary, evaluates out-of-sample, aggregates equity curve across all folds. |

**Phase 4 verified results (EURUSD M15, walk-forward, threshold=0.40):**

| Metric | Value | vs Random Baseline |
|--------|-------|--------------------|
| Sharpe ratio | **1.34** | Random was -0.17 |
| Sortino ratio | 1.83 | — |
| Total return | **+14.2%** | Random was -2.5% |
| Max drawdown | 10.5% | Random was 21.6% |
| Profit factor | 1.12 | Random was 0.99 |
| Trades | 176 across 19 folds | — |
| Win rate | 36.4% | — |

- Log-loss: 1.078 (random baseline = 1.099) — small but genuine edge
- Confidence is clustered below 0.45 (normal for calibrated Forex model); threshold 0.40 is appropriate

**Key commands:**
```bash
python scripts/train_model.py --no-importance         # train and save model
python scripts/walk_forward.py --threshold 0.40       # out-of-sample validation
```

---

### Phase 3 — Built

| File | What it does |
|------|-------------|
| [src/feature_pipeline.py](src/feature_pipeline.py) | `FeaturePipeline` class. Computes 31 features (SMA, EMA, RSI, MACD, BB, ATR, Stochastic, ADX, lag returns, rolling std, MA spreads). Shifts all indicators by 1 bar to prevent lookahead. Fits `StandardScaler` on train only. Generates labels: y=1 (buy) if next-4-bar return > 0.03%, y=-1 (sell), y=0 (hold). Saves/loads scaler via joblib. |
| [scripts/build_features.py](scripts/build_features.py) | CLI to build and save the full feature matrix. Outputs `data/features/EURUSD_M15_features.parquet`, `_labels.parquet`, `_split.parquet`, `data/models/scaler.joblib`. Includes `--validate` flag for lookahead check. |

**Phase 3 verified results (EURUSD M15, 50k bars):**
- Feature matrix: 49,892 rows × 31 features
- Label split: SELL 29.7% / HOLD 40.0% / BUY 30.3%
- Lookahead validation: **PASSED** (31 features, 0 violations)
- Train/test split: 80/20 (2024-05-13→2025-12-19 train, 2025-12-19→2026-05-18 test)

**Environment note (important for new sessions):**
- After installing scikit-learn/joblib/pyarrow, pypi numpy 2.2.6 and pandas 2.3.3 conflicted with conda numpy 1.26.4
- Fixed by: `pip install "numpy==1.26.4" "pandas==2.2.3" --force-reinstall` using the env's pip directly
- Correct pinned versions: numpy=1.26.4, pandas=2.2.3, scipy=1.11.4, scikit-learn=1.1.1

---

### Phase 2 — Built

| File | What it does |
|------|-------------|
| [src/indicators.py](src/indicators.py) | Composable indicator library. Pure functions. `sma`, `ema`, `rsi`, `macd`, `bollinger_bands`, `bollinger_pct_b`, `atr`, `stochastic`, `adx`, `obv`. Single entry point: `compute(df, spec)` adds columns to df. |
| [src/rule_engine.py](src/rule_engine.py) | Rule combiner → probability output. `Rule` dataclass + `SignalCombiner`. Pre-built rule factories: `ma_crossover_rule`, `rsi_rule`, `macd_rule`, `bb_reversion_rule`, `price_vs_ma_rule`, `stochastic_rule`. `predict_proba(df)` returns `[P_buy, P_hold, P_sell]`. |
| [src/rule_bot.py](src/rule_bot.py) | Rule-based signal bot. `--backtest` mode and live mode. Uses indicators + rule engine. |

### Config sections added

```yaml
random_bot:        # Phase 1 — entry_prob, sl_pips, tp_pips, seed
rule_bot:          # Phase 2 — threshold, fast_ma, slow_ma, rsi_period, weights
```

---

## Model Recommendations (for Phase 6 Ensemble)

Documented 2026-05-24. Priority order for adding to the ensemble:

| Priority | Model | Why | Phase |
|----------|-------|-----|-------|
| **1 — CatBoost** | CatBoost | Same boosting family as XGBoost/LightGBM but handles categorical features (day-of-week, session, market regime) natively. Often 5–10% better Sharpe than XGBoost on Forex. Easy to add — same `ModelInterface`. | Phase 6 |
| **2 — LSTM** | 2-layer LSTM | Adds *sequence memory* — each bar knows about the bars before it. XGBoost/LightGBM treat each bar independently. Different error profile = valuable ensemble member. | Phase 6 |
| **3 — TFT** | Temporal Fusion Transformer | Purpose-built for financial time series. Learns long-range dependencies + variable importance. Only worth it with 200k+ rows or multi-symbol data. | Phase 9+ |

**Why Random Forest underperformed (Sharpe 0.24):**
Bagging models (RF) train trees independently on random subsets — they don't learn from each other's mistakes. Boosting models (XGBoost, LightGBM, CatBoost) do. For structured financial data, boosting consistently wins. RF is still useful in the ensemble because its *errors are different* from XGBoost — the meta-learner exploits that.

**Transformer note:** Can fit, but our 50k-bar dataset is too small. Standard Transformers need hundreds of thousands of rows. The Temporal Fusion Transformer (TFT, `pytorch-forecasting`) is the right architecture when we scale to multi-symbol or longer history.

**Decision logged:** CatBoost will be added in Phase 6 alongside LightGBM and LSTM in the ensemble layer.

---

## Verified Backtest Results

Run on `data/EURUSD_M15.csv` (50,000 bars, EURUSD M15, 2024-05-13 → 2026-05-18):

| Bot | Command | Sharpe | Win Rate | Return | Max DD | Trades |
|-----|---------|--------|----------|--------|--------|--------|
| Random baseline | `python src/random_bot.py --backtest --seed 42` | **-0.17** | 33.3% | -2.5% | 21.6% | 354 |
| Rule-based | `python src/rule_bot.py --backtest` | **-0.45** | 33.0% | -9.7% | — | 430 |
| **XGBoost** (walk-forward) | `python scripts/walk_forward.py --model xgboost --threshold 0.40` | **1.34** | 36.4% | **+14.2%** | 10.5% | 176 |
| **LightGBM** (walk-forward) | `python scripts/walk_forward.py --model lightgbm --threshold 0.40` | **0.72** | 35.4% | +3.8% | 6.8% | 99 |
| **Random Forest** (walk-forward) | `python scripts/walk_forward.py --model random_forest --threshold 0.40` | **0.24** | 34.4% | -0.1% | 14.8% | 93 |

**Phase 5 interpretation:**
- XGBoost is the strongest standalone model (Sharpe 1.34)
- LightGBM is profitable but conservative with default params (Sharpe 0.72, fewest trades)
- Random Forest barely breaks even — bagging loses to boosting on tabular financial data
- All three have *different error profiles* — the Phase 6 meta-learner will exploit this
- CatBoost (Phase 6 priority) is expected to outperform all three individually

---

## Phase 3 — What to Build Next

**Goal:** Build `src/feature_pipeline.py` — an ML-ready feature matrix builder with strict no-lookahead guarantees.

### Files to create

**`src/feature_pipeline.py`** — core module:
- `FeaturePipeline` class
- Registers feature generators by name (wraps `indicators.py` functions)
- Lag features: `close_lag_1`, `close_lag_2`, `return_1`, `return_5`, `return_10`
- Rolling stats: `rolling_std_10`, `rolling_std_20`, `rolling_skew_20`, `rolling_kurt_20`
- Multi-timeframe: fetch M15 + H1 + H4, merge on time index (H1/H4 bars forward-filled to M15)
- `fit_scaler(train_df)` → saves StandardScaler fitted only on training window
- `transform(df)` → applies saved scaler (never refit)
- `validate_no_lookahead(df)` → raises if any feature column correlates with future returns above threshold
- `build(df, fit=True)` → full pipeline: compute indicators + lag features + rolling stats + normalize

**`scripts/build_features.py`** — CLI:
- Loads CSV, runs `FeaturePipeline.build()`, saves to `data/features/EURUSD_M15_features.parquet`
- `--validate` flag runs the lookahead checker

### Key design rules for Phase 3
- All features must use `.shift(1)` so bar[t] features are computed from bar[t-1] data
- Scaler must be `fit` only on training data — never on the full dataset
- Store scaler as `data/models/scaler.joblib` alongside the Parquet output
- Feature names must be stable strings (used as column names throughout Phases 4–9)

### Suggested feature list for Phase 4 training

```python
features = [
    # Price-based
    "close_lag_1", "close_lag_2", "close_lag_5",
    "return_1", "return_5", "return_10",
    "rolling_std_10", "rolling_std_20",
    # Indicators (all shifted by 1 bar)
    "sma_9", "sma_21", "sma_50",
    "ema_20",
    "rsi_14",
    "macd_line", "macd_hist",
    "bb_pct",           # position within Bollinger Bands
    "atr_14",
    "stoch_k", "stoch_d",
    "adx_14",
    # Cross-timeframe (H1 and H4, resampled to M15)
    "h1_rsi_14", "h1_sma_20",
    "h4_rsi_14", "h4_sma_50",
]
```

### Label generation (also needed for Phase 4)
Create labels in `build_features.py`:
- `y = 1` (buy)  if `close[t+N] / close[t] - 1 > threshold` (e.g. N=4 bars, threshold=0.0003)
- `y = -1` (sell) if `close[t+N] / close[t] - 1 < -threshold`
- `y = 0` (hold) otherwise
- **Critical:** labels must be computed then `.shift(-N)` aligned, and the last N rows dropped

---

## Key Patterns to Know

### How indicators.py works
```python
from src.indicators import compute, sma, rsi, atr, bollinger_pct_b, macd

df = compute(df, [
    ("sma_9",   sma,  {"period": 9}),
    ("rsi_14",  rsi,  {"period": 14}),
    ("atr_14",  atr,  {"period": 14}),   # needs full df (OHLCV)
    ("bb_pct",  bollinger_pct_b, {}),
    (("macd_line", "macd_sig", "macd_hist"), macd, {}),  # multi-output
])
```
- Single-output indicators: `("col_name", fn, kwargs)`
- Multi-output indicators: `(("col1", "col2", "col3"), fn, kwargs)`
- OHLCV indicators (`atr`, `stochastic`, `adx`, `obv`) auto-receive full df; all others receive `df["close"]`

### How rule_engine.py works
```python
from src.rule_engine import SignalCombiner, ma_crossover_rule, rsi_rule

combiner = SignalCombiner(threshold=0.55)
combiner.add(ma_crossover_rule("sma_9", "sma_21"), weight=2.0, name="ma_cross")
combiner.add(rsi_rule("rsi_14"),                    weight=1.5, name="rsi")

proba = combiner.predict_proba(df)   # → array([P_buy, P_hold, P_sell])
```

### How metrics.py works
```python
from src.metrics import performance_report, sharpe_ratio

# Works with: list[Trade dataclass], list[dict], or pd.DataFrame
performance_report(trades, equity_series, initial_balance=10000, title="MY BACKTEST")

# Individual metrics
sharpe = sharpe_ratio(equity_series)
```

### How trade_journal.py works
```python
from src.trade_journal import TradeJournal

journal = TradeJournal()          # creates data/trades.db
journal.record({
    "bot": "rule_bot", "symbol": "EURUSD", "direction": "buy",
    "entry_time": "...", "entry_price": 1.085,
    "exit_time": "...", "exit_price": 1.088,
    "pnl_pips": 30.0, "pnl_dollars": 30.0,
    "model": "rule_engine", "confidence": 0.68,
    "entry_reason": "ma_cross+rsi", "exit_reason": "tp",
    "volume": 0.01, "sl_pips": 30.0, "tp_pips": 60.0,
})
df = journal.get_trades()
journal.print_summary()
```

---

## What NOT to Change

- **`src/mt5_connector.py`** — never modify the public interface
- **`src/bot_base.py`** — extend via subclass only; do not edit the base class
- **`data/EURUSD_M15.csv`** — source of truth for backtesting; re-download with `python scripts/download_data.py` if needed
- **`config.yaml` existing sections** — only add new sections; never remove or rename existing keys

---

## Quick Reference Commands

```bash
conda activate envmt5

# Phase 1 — random baseline
python src/random_bot.py --backtest --seed 42

# Phase 2 — rule-based signals
python src/rule_bot.py --backtest --threshold 0.55

# Original MA crossover (for comparison)
python src/backtest.py --fast 9 --slow 21

# Download fresh data
python scripts/download_data.py --symbol EURUSD --timeframe M15 --bars 50000

# Test MT5 connection (requires MT5 running via start_mt5.sh)
python tests/test_connection.py
```
