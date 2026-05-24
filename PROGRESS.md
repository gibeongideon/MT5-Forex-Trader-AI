# MT5 Trading Bot — Build Progress

> **Purpose:** Session handoff document. A new Claude session should read this + `IMPLEMENTATION-PLAN.MD` before doing any work. Do NOT re-explore files that are already described here.

Last updated: 2026-05-24

---

## Current Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Random Baseline + Logging Infrastructure | ✅ COMPLETE |
| 2 | Composable Indicator Library + Rule Engine | ✅ COMPLETE |
| 3 | Feature Engineering Pipeline | ⬜ NOT STARTED |
| 4 | XGBoost with Calibrated Probabilities | ⬜ NOT STARTED |
| 5 | Pluggable Model Registry | ⬜ NOT STARTED |
| 6 | Signal Stacking & Meta-Learning | ⬜ NOT STARTED |
| 7 | Robust Backtesting & Walk-Forward | ⬜ NOT STARTED |
| 8 | Intelligent Risk Management | ⬜ NOT STARTED |
| 9 | LLM Integration as Probability Signal | ⬜ NOT STARTED |
| 10 | Production Enterprise System | ⬜ NOT STARTED |

**Next task:** Start Phase 3 — `src/feature_pipeline.py` and `scripts/build_features.py`

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

## Verified Backtest Results

Run on `data/EURUSD_M15.csv` (50,000 bars, EURUSD M15, 2024-05-13 → 2026-05-18):

| Bot | Command | Sharpe | Win Rate | Return | Trades |
|-----|---------|--------|----------|--------|--------|
| Random baseline | `python src/random_bot.py --backtest --seed 42` | **-0.17** | 33.3% | -2.5% | 354 |
| Rule-based | `python src/rule_bot.py --backtest` | **-0.45** | 33.0% | -9.7% | 430 |
| MA crossover (original) | `python src/backtest.py --fast 9 --slow 21` | (run to check) | — | — | — |

**Interpretation:** Random baseline is correctly negative (proves no edge). Rule-based is also negative because the rules use unoptimized weights on a fixed SL/TP in a mixed trending/ranging market. This is expected at Phase 2 — Phase 4 (XGBoost) will learn optimal feature weights from data.

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
