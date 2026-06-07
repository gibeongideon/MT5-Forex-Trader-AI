# MT5 Trading Bot — Build Progress

> **Purpose:** Session handoff document. A new Claude session should read this + `IMPLEMENTATION-PLAN.MD` before doing any work. Do NOT re-explore files that are already described here.

Last updated: 2026-06-07

---

## Current Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Random Baseline + Logging Infrastructure | ✅ COMPLETE |
| 2 | Composable Indicator Library + Rule Engine | ✅ COMPLETE |
| 3 | Feature Engineering Pipeline | ✅ COMPLETE |
| 4 | XGBoost with Calibrated Probabilities | ✅ COMPLETE |
| 5 | Pluggable Model Registry | ✅ COMPLETE |
| 6 | Signal Stacking & Meta-Learning | ✅ COMPLETE |
| 7 | Robust Backtesting & Walk-Forward | ✅ COMPLETE |
| 8 | Intelligent Risk Management | ✅ COMPLETE |
| 9 | LLM Integration as Probability Signal | ⬜ NOT STARTED |
| 10 | Production Enterprise System | ⬜ NOT STARTED |

**Next task:** Start Phase 9 — LLM Integration as Probability Signal

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

### Phase 8 — Built

| File | What it does |
|------|-------------|
| [src/risk_manager.py](src/risk_manager.py) | Confidence-tiered position sizer. `RiskConfig` controls tiers, Kelly, ATR stop, portfolio cap, drawdown throttle. `RiskManager.size(confidence, balance, sl_pips, tp_pips, ...)` → `SizingResult` with `skip`, `risk_pct`, `sl_pips`, `tp_pips`, `dollar_risk`. |
| [src/backtester.py](src/backtester.py) | Updated: `BacktestConfig.risk_manager` field wires `RiskManager` for per-trade dynamic sizing. `Trade` dataclass gains `risk_pct` field. `_close_trade()` and `_pnl_dollars()` use per-trade pips/risk instead of fixed config values. |
| [src/bot_base.py](src/bot_base.py) | Extended (not rewritten): `risk_sized_lot(symbol, confidence, sl_pips, tp_pips, atr_value, drawdown_pct)` → `(lot, effective_sl_pips)`. Returns `(0.0, sl_pips)` when confidence is below minimum. |
| [scripts/phase8_compare.py](scripts/phase8_compare.py) | A/B/C walk-forward comparison script. Loads parquet features and runs 3 configs through `WalkForwardValidator`. |

**`config.yaml` additions:**
```yaml
risk_manager:
  tiers: [[0.75, 0.020], [0.65, 0.015], [0.55, 0.0075], [0.40, 0.005]]
  min_confidence: 0.40
  use_kelly: false  / kelly_fraction: 0.25 / kelly_max_risk: 0.03
  use_atr_stop: false / atr_multiplier: 1.5 / min_sl_pips: 15.0 / max_sl_pips: 60.0
  max_portfolio_risk: 0.03
  drawdown_threshold: 0.10 / drawdown_throttle: 0.50
```

**Phase 8 verified results (XGBoost, walk-forward, 19 folds, threshold=0.40, spread=1.0p):**

| Config | Sharpe | Max DD | Return | Trades | Key change |
| --- | --- | --- | --- | --- | --- |
| A — Fixed 1% risk (baseline) | +0.72 | 14.6% | -0.3% | 126 | — |
| **B — Confidence-tiered risk** | **+0.72** | **7.5%** | **+0.2%** | 126 | Same Sharpe, **−49% drawdown** |
| C — Tiered + ATR stop | +0.22 | 3.4% | +0.3% | 69 | Lower drawdown, fewer trades, lower Sharpe |

**Key findings:**

- **Tiered risk (B) is the winner**: identical Sharpe (0.72) but drawdown cut from 14.6% → 7.5% at zero cost. Return flips from -0.3% → +0.2%.
- ATR stop (C) further reduces drawdown to 3.4% but wide ATR stops reduce pips/risk efficiency → Sharpe falls to 0.22. Useful for very risk-averse accounts.
- The confidence tiers (0.75→2%, 0.65→1.5%, 0.55→0.75%, 0.40→0.5%) naturally downsize losing runs (low confidence streaks) and upsize winning runs — the core risk management benefit.
- `BotBase.risk_sized_lot()` makes tiered sizing available to all live bots with one line.

**Bug fixed during Phase 8:** `src/walk_forward.py` `fold_cfg` constructor was missing `risk_manager=config.backtest.risk_manager`. All three configs appeared identical until this was fixed.

**Key commands:**
```bash
conda run -n envmt5 --cwd /home/rock/Desktop/2026_Projects/MT5 python scripts/phase8_compare.py
```

---

### Phase 7 — Built

| File | What it does |
|------|-------------|
| [src/backtester.py](src/backtester.py) | Event-driven bar-by-bar simulator. `BacktestConfig` controls threshold, spread, commission, slippage, regime filter. `Backtester().run(model, X, prices, cfg)` returns `BacktestResult` with trades + equity curve + `.report()`. |
| [src/walk_forward.py](src/walk_forward.py) | Full walk-forward engine. `WalkForwardConfig` sets model, window type (expanding/sliding), train/test days, backtest config. Supports all single models and ensemble (per-fold retrain with optional disk cache). Returns `WalkForwardResult` with fold table + aggregate equity curve. |
| [src/monte_carlo.py](src/monte_carlo.py) | Monte Carlo trade-order shuffler. `run_monte_carlo(trades, n_simulations=1000)` → `MonteCarloResult`. Reports 5th/25th/50th/75th/95th-percentile Sharpe across shuffles + text histogram. |
| [TODO.md](TODO.md) | Phase-by-phase TODO tracker. Mark tasks ✅ / 🔄 / ⬜ as work progresses. |

**`config.yaml` additions:**
```yaml
backtester:
  threshold: 0.40
  sl_pips: 30.0 / tp_pips: 60.0
  spread_pips: 1.0 / commission_pips: 0.5 / max_slippage_pips: 0.3
  initial_balance: 10000.0 / risk_pct: 0.01
  use_regime_filter: false / adx_threshold: 20.0

walk_forward:
  model_type: xgboost
  window_type: expanding
  train_days: 180 / test_days: 30
  cache_dir: data/models/wf_cache

monte_carlo:
  n_simulations: 1000 / seed: 42
```

**Phase 7 key findings:**

| Model | Threshold | Spread+Comm+Slip | Sharpe | Return | Trades |
| --- | --- | --- | --- | --- | --- |
| XGBoost (walk-forward, no costs) | 0.40 | 0p | **1.34** | +14.2% | 176 |
| **XGBoost (walk-forward, with costs)** | 0.40 | 1.0p+0.5p+0.3p | **0.52** | -3.8% | 127 |

**Critical insight: Transaction costs cut Sharpe from 1.34 → 0.52.** The cost model (spread 1.0p, commission 0.5p, slippage up to 0.3p = ~1.8p total round-trip) reduces the XGBoost edge to near-breakeven. This is realistic — most Forex scalping strategies fail once proper costs are applied.

**Monte Carlo (1000 shuffles, XGBoost trades):**

- 5th-percentile Sharpe: **-1.17** — fails the > 0.5 target (edge is not large enough to survive bad luck)
- Original Sharpe near shuffle median (53.3% of shuffles beat it) — **good: performance is not order-dependent**
- Conclusion: the *trade outcomes* carry genuine edge, but the *expected return* with costs is marginal

**What this means for Phase 8:**
The ensemble (Sharpe 39% accuracy, 41.8% P≥0.40 coverage) has higher accuracy than XGBoost alone. Phase 8's confidence-based position sizing should let winners run more while cutting losers smaller — the key to making the edge survive costs. The regime filter (ADX < 20 = skip) should also be evaluated.

**Key commands:**
```bash
# Full walk-forward with transaction costs
python -c "
from src.backtester import BacktestConfig
from src.walk_forward import WalkForwardValidator, WalkForwardConfig
import pandas as pd
X = pd.read_parquet('data/features/EURUSD_M15_features.parquet')
y = pd.read_parquet('data/features/EURUSD_M15_labels.parquet')['label']
prices = pd.read_csv('data/EURUSD_M15.csv', index_col='time')
prices.index = pd.to_datetime(prices.index)
cfg = WalkForwardConfig(model_type='xgboost', backtest=BacktestConfig(threshold=0.40))
WalkForwardValidator().run(X, y, prices, cfg).report()
"

# With regime filter
# Set use_regime_filter=True in BacktestConfig to suppress signals when ADX < 20

# Monte Carlo on any trade list
# from src.monte_carlo import run_monte_carlo
# mc = run_monte_carlo(trades); mc.report(); mc.histogram()
```

---

### Phase 6 — Built

| File | What it does |
|------|-------------|
| [src/models/catboost_model.py](src/models/catboost_model.py) | CatBoost 3-class classifier. No label remapping needed. `bootstrap_type="Bernoulli"` required to enable `subsample`. `calibration_cv=0` (CatBoost is already well-calibrated). `allow_writing_files=False`. |
| [src/models/lstm_model.py](src/models/lstm_model.py) | 2-layer LSTM using PyTorch. `seq_len=20` bar lookback window. Adds sequence memory that tree models lack. `_TORCH_AVAILABLE` guard — degrades gracefully if PyTorch not installed. Save/load via `torch.save`/`torch.load`. |
| [src/ensemble.py](src/ensemble.py) | Two-layer stacking ensemble implementing `ModelInterface`. Layer-0: all base models output `[P_buy, P_hold, P_sell]`. Layer-1: logistic or LightGBM meta-learner on stacked OOF predictions. Uses `StratifiedKFold` to generate out-of-fold predictions (leakage-free). `model_weights()` shows per-model trust. |
| [scripts/train_ensemble.py](scripts/train_ensemble.py) | CLI to train and evaluate the ensemble. `--base` selects which models to stack, `--meta` selects logistic or lightgbm meta-learner, `--folds` sets CV folds. Prints classification report, log-loss, confidence distribution, and meta-learner weights. |

**`config.yaml` additions:**
```yaml
active_model: xgboost
models:
  catboost: {type: catboost, path: data/models/catboost.joblib}
  lstm:     {type: lstm,     path: data/models/lstm.pt}
  ensemble: {type: ensemble, path: data/models/ensemble.joblib}
ensemble:
  base_models: [xgboost, lightgbm, catboost, random_forest]
  meta_model: logistic
  n_folds: 5
  use_original_features: false
```

**Phase 6 evaluation results (test set, 9,946 bars):**

| Meta-learner | Log-loss | Accuracy | P≥0.40 coverage | P≥0.45 coverage |
| --- | --- | --- | --- | --- |
| Logistic | 1.0955 | 38% | 18.0% | 3.8% |
| **LightGBM** | **1.0869** | **39%** | **41.8%** | **21.8%** |
| Random baseline | 1.099 | 33% | — | — |

LightGBM meta-learner is the winner: lower log-loss, higher accuracy, and 2× more confident bars (41.8% reach P≥0.40 vs. 18% with logistic). The ensemble model saved to `data/models/ensemble.joblib` uses the LightGBM meta-learner.

**Key insight:** All 4 base models (XGBoost, LightGBM, CatBoost, RF) share the same 31 tabular features → their predictions are highly correlated → the meta-learner has limited complementarity to exploit. Adding LSTM (sequence memory, different error profile) and LLM signal (Phase 9) will unlock more ensemble benefit.

**Ensemble walk-forward deferred to Phase 7:** Full ensemble walk-forward requires retraining 4 base models × 5 CV folds per walk-forward fold (19 folds × ~20 training runs = 380+ training runs). This is done efficiently in Phase 7's optimized backtester with checkpoint/cache support.

**Key commands:**
```bash
python scripts/train_ensemble.py                        # train with logistic meta (fast)
python scripts/train_ensemble.py --meta lightgbm        # train with LightGBM meta (better)
python scripts/train_ensemble.py --no-catboost          # skip CatBoost if not installed
```

---

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
| **CatBoost** (walk-forward) | `python scripts/walk_forward.py --model catboost --threshold 0.40` | **1.17** | 34.4% | +5.8% | 24.7% | 543 |
| **Random Forest** (walk-forward) | `python scripts/walk_forward.py --model random_forest --threshold 0.40` | **0.24** | 34.4% | -0.1% | 14.8% | 93 |
| **Ensemble** (test set, LightGBM meta) | `python scripts/train_ensemble.py --meta lightgbm` | log-loss 1.087 | 39% acc | — | — | — |

**Phase 6 interpretation:**

- CatBoost (Sharpe 1.17) is competitive with XGBoost (1.34) but trades far more (543 vs 176) with higher drawdown (24.7% vs 10.5%)
- Ensemble test-set log-loss 1.0869 beats random baseline (1.099) — genuine edge above noise floor
- LightGBM meta gives 41.8% of bars at P≥0.40 vs. only 18% with logistic — much more actionable
- Ensemble walk-forward deferred to Phase 7 (too expensive per-fold without caching/optimization)
- Adding LSTM + LLM signal (Phase 9) will increase ensemble benefit by diversifying error profiles

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

---

## Deferred Next-Step Options (documented 2026-06-07)

After 20 research phases, champion is locked: **XGBoost + enc8, M15 → +3.13 Sharpe, 13.3% MaxDD, +358% return**.
Two paths forward once current SMC signal research (Phase 21) concludes:

### Option A — Deploy Champion (Production)

Retrain XGBoost + enc8 on the full 49k dataset (no walk-forward splits), wire into a live
`PipelineBot` extending `BotBase`, run 30 days paper trading before live capital.

Files needed:
- `src/pipeline_bot.py` — extends BotBase; calls PredictorPipeline.predict_live() each tick
- `scripts/retrain_champion.py` — full-data retrain + saves encoder.pt, scaler.joblib, xgboost.joblib
- `config.yaml` additions: `mode: paper|live`, `paper_balance`

Key constraint: 30-day paper mode is mandatory before any live capital (see Phase 10 in TODO.md).

### Option B — LLM Integration (Phase 9-A from TODO.md)

Precompute LLM signals for all 49k bars (~$1.50, ~60 min API cost):

```bash
conda run -n envmt5 python scripts/precompute_llm_signals.py --provider claude_api
```

Then add `llm_signal` to ensemble `base_models` list and compare walk-forward vs +3.13 champion.
Hypothesis: LLM uses sequential bar patterns (32-bar token context) — orthogonal to enc8's
regime fingerprint from OHLCV windows. The only untested complementary signal type remaining.


---

## Phase 21 Results — SMC/ICT Signals (2026-06-07)

**Saturation confirmed after 21 experiments. Champion holds.**

| Config | Features | Sharpe | MaxDD | Return | Verdict |
|--------|----------|--------|-------|--------|---------|
| A (baseline) | 39 | +2.31 | 8.0% | +43% | reference (fresh enc8 vs +3.13 cached) |
| B (+OB+FVG+DailyHL) | 47 | +1.42 | 9.9% | +16.3% | worse |
| C (all 6 SMC types) | 55 | +1.16 | 10.9% | +18.4% | worse |

- All 6 SMC signal types tested: Order Blocks, Fair Value Gaps, Previous Day Levels, Andean Oscillator, SuperTrend, Heiken Ashi
- All hurt Sharpe. enc8 latent vectors already implicitly capture what SMC patterns describe.
- Source: `scripts/compare_smc_signals.py`, signals in `src/features/smc_signals.py`

---

## Phase 22 — Production Hardening + Volume Signal Experiment (2026-06-07)

### aiomql Evaluation Decision

Scanned `/trader_reference/aiomql/` (302 files, full async MT5 framework).
**Decision: borrow 3 patterns only. Do NOT replace BotBase + MT5Connector.**

Reasons: our stack is battle-tested over 21 phases; aiomql has no backtesting,
no encoder, no custom signal generation; replacing = high risk, zero signal benefit.

Patterns borrowed (ideas, not the library):
- **PositionTracker** → `PipelineBot._manage_positions()` — breakeven SL move at 1× profit
- **Sessions** → `BotBase.in_session()` + `config.yaml: trading.session_filter`
- **Trade recording** → `TradeJournal` wired into `PipelineBot.on_tick()`

### Files Created / Modified

| File | Change |
|------|--------|
| `scripts/retrain_champion.py` | NEW — full-data retrain, saves all artifacts |
| `src/core/mt5_connector.py` | ADD — `modify_position(ticket, sl, tp)` |
| `src/core/bot_base.py` | ADD — `in_session()` reads session_filter from config |
| `config.yaml` | ADD — `trading.session_filter` block (disabled by default) |
| `src/bots/pipeline_bot.py` | ADD — `_manage_positions()`, session check, journal wiring |
| `src/features/volume_signals.py` | NEW — 3 volume anomaly features |
| `scripts/compare_volume_signals.py` | NEW — Phase 22-A A/B walk-forward |
| `deploy/mt5_bot.service` | NEW — systemd service for auto-start |

### Phase 22-A: Volume Signal Experiment (PENDING)

```bash
conda run -n envmt5 --no-capture-output python scripts/compare_volume_signals.py
```

Features: `vol_ratio`, `vol_zscore`, `vol_fast_slow` (from `tick_volume` column)
- Config A: 39 feat (baseline, reuses SMC cache)
- Config B: 42 feat (+3 volume features)

**Saturation guard:** If volume signals also hurt → accept full saturation, deploy champion.
If volume signals improve → run Phase 22-B (LLM signals, ~$1.50 API cost).

### Deploy Champion (when ready)

```bash
# Step 1: Retrain on all data
conda run -n envmt5 python scripts/retrain_champion.py

# Step 2: Dry-run
conda run -n envmt5 python src/bots/pipeline_bot.py --dry-run

# Step 3: Paper trading (30 days minimum before live capital)
conda run -n envmt5 python src/bots/pipeline_bot.py

# Step 4: Systemd service (auto-start on reboot)
sudo cp deploy/mt5_bot.service /etc/systemd/system/
sudo systemctl enable mt5_bot
sudo systemctl start mt5_bot
journalctl -u mt5_bot -f
```
