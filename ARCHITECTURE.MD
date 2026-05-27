# ARCHITECTURE.MD
## MT5 Trading Bot — Developer Reference

> This file is the single source of truth for the codebase structure, contracts,
> and extension points. Read this before adding any new component.

---

## 1. Data Flow (Raw OHLCV → Trade)

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA SOURCE                                                         │
│  CSV file (historical)   OR   MT5 live bars via rpyc bridge          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ OHLCV DataFrame
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Feature Engineering          src/features/               │
│  FeaturePipeline                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  RSI(14)  MACD  BB(20)  ATR(14)  ADX  SMA(9/21/50)  OBV  Stoch    │
│  StandardScaler fitted on train portion only (no leakage)           │
│  → 31-dim feature matrix                                             │
│                                                                      │
│  EXTEND: add indicators in src/features/indicators.py               │
│          add feature columns in src/features/feature_pipeline.py    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ 31 features
           ┌───────────────────────┼──────────────────────────────┐
           │  PLUG-IN A (toggle)   ▼                              │
           │  ┌──────────────────────────────────────────────┐    │
           │  │  LAYER 2 — Latent Encoder                    │    │
           │  │  src/features/latent_encoder.py              │    │
           │  │  ─────────────────────────────────────────── │    │
           │  │  50-bar OHLCV window → z-normalise per window│    │
           │  │  MLP encoder (trained w/ CrossEntropyLoss)   │    │
           │  │  → 8-dim regime fingerprint (latent_0..7)    │    │
           │  │                                               │    │
           │  │  ON/OFF: config.yaml → encoder.enabled       │    │
           │  │  EXTEND: add new encoder classes here        │    │
           │  └──────────────────────┬───────────────────────┘    │
           │                         │ +8 latent dims              │
           └─────────────────────────┘                             │
                                   ▼
                        39-dim feature matrix (31 + 8)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Prediction Model             src/models/                 │
│  ModelInterface contract (ONE method: predict_proba)                │
│  ─────────────────────────────────────────────────────────────────  │
│  xgboost_model.py      lightgbm_model.py     catboost_model.py     │
│  lstm_model.py         ensemble.py            llm_signal_model.py  │
│  bar_lm_model.py                                                    │
│                                                                      │
│  → [P_buy, P_hold, P_sell]  (three floats that sum to 1.0)         │
│                                                                      │
│  SWAP:   config.yaml → model_type: xgboost | lightgbm | ...        │
│  EXTEND: add new file in src/models/, implement ModelInterface      │
│          register in src/models/model_registry.py                  │
│          add entry in config.yaml → models: section                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ probabilities
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Risk Manager                 src/core/risk_manager.py    │
│  Confidence tiers → position size and SL/TP                         │
│  ─────────────────────────────────────────────────────────────────  │
│  P ≥ 0.75 → 2.0% risk    P ≥ 0.65 → 1.5% risk                     │
│  P ≥ 0.55 → 0.75% risk   P ≥ 0.40 → 0.5% risk                     │
│  Kelly sizing optional    ATR-based dynamic SL optional             │
│                                                                      │
│  EXTEND: add tiers in config.yaml → risk_manager.tiers              │
│          add sizing logic in RiskManager.size_position()            │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ signal + sizing dict
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Bot / Execution              src/bots/                   │
│  BotBase subclass (implements on_tick)                               │
│  ─────────────────────────────────────────────────────────────────  │
│  pipeline_bot.py     ai_bot.py     rule_bot.py     random_bot.py   │
│  Every 60s: fetch bars → predict → manage positions                 │
│  HOLD / skip → nothing   BUY → close sells, open 1 buy             │
│  magic=20260101 isolation  5% daily loss guard                      │
│                                                                      │
│  EXTEND: new strategy → new file in src/bots/, extend BotBase       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ orders
                                   ▼
                      MT5 Terminal (ICMarketsKE-Demo)
                      rpyc bridge  localhost:18812
```

---

## 2. Folder Structure

```
MT5/
├── config.yaml                  ← SINGLE config file for everything
├── .env                         ← secrets (MT5_LOGIN, ANTHROPIC_API_KEY, etc.)
│
├── src/
│   ├── pipeline.py              ← TOP-LEVEL ORCHESTRATOR (kept at root)
│   │                              PredictorPipeline: build_features → fit → predict
│   │
│   ├── core/                    ← FROZEN infrastructure — do not change contracts
│   │   ├── bot_base.py          ← BotBase ABC: run loop, daily loss guard, helpers
│   │   ├── mt5_connector.py     ← rpyc bridge wrapper: rates, tick, orders
│   │   ├── risk_manager.py      ← RiskManager: confidence → lot size / SL
│   │   ├── trade_journal.py     ← SQLite trade log
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   └── types.py
│   │
│   ├── features/                ← FEATURE ENGINEERING — safe to extend
│   │   ├── feature_pipeline.py  ← FeaturePipeline: OHLCV → 31 indicators + scaler
│   │   ├── indicators.py        ← pure functions: sma, ema, rsi, macd, bb, atr ...
│   │   ├── latent_encoder.py    ← LatentEncoder: OHLCV windows → latent dims
│   │   └── bar_tokenizer.py     ← BarTokenizer: OHLCV → discrete tokens (for LLM)
│   │
│   ├── models/                  ← ML MODELS — add new models here
│   │   ├── model_interface.py   ← ModelInterface ABC (3 methods: train/predict/save)
│   │   ├── model_registry.py    ← ModelRegistry: name → ModelInterface singleton
│   │   ├── ensemble.py          ← Ensemble: stacking meta-learner
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── random_forest_model.py
│   │   ├── lstm_model.py
│   │   ├── llm_signal_model.py  ← calls Anthropic API (cloud model example)
│   │   └── bar_lm_model.py      ← small transformer trained on bar tokens
│   │
│   ├── evaluation/              ← BACKTESTING & METRICS
│   │   ├── backtester.py        ← Backtester: bar-by-bar simulation with costs
│   │   ├── walk_forward.py      ← WalkForwardValidator: expanding/sliding folds
│   │   ├── metrics.py           ← sharpe_ratio, max_drawdown, performance_report
│   │   ├── monte_carlo.py       ← trade-sequence shuffle significance test
│   │   └── backtest.py          ← simple MA-crossover baseline backtester
│   │
│   ├── bots/                    ← RUNNABLE STRATEGIES
│   │   ├── pipeline_bot.py      ← LIVE BOT: uses PredictorPipeline (CURRENT)
│   │   ├── ai_bot.py            ← uses Anthropic LLM for signals
│   │   ├── rule_bot.py          ← MA + RSI + BB rule-based signals
│   │   ├── random_bot.py        ← random entry baseline
│   │   ├── llm_bot.py           ← LLM signal model bot
│   │   └── example_bot.py       ← minimal BotBase template
│   │
│   ├── signals/                 ← SIGNAL LOGIC
│   │   ├── rule_engine.py       ← Rule, SignalCombiner: weighted rule voting
│   │   └── predictor.py        ← legacy swappable predictor endpoint
│   │
│   └── data/
│       └── schemas.py
│
├── scripts/                     ← CLI runners (not importable library code)
│   ├── run_pipeline.py          ← main runner: backtest | train | predict | features
│   ├── train_autoencoder.py     ← train LatentEncoder standalone
│   ├── build_latent_features.py ← merge latent cols into feature parquet
│   ├── compare_latent_vs_baseline.py  ← A/B Sharpe comparison
│   └── ...
│
└── data/
    ├── EURUSD_M15.csv           ← RAW DATA (immutable — never modify)
    ├── features/                ← cached feature matrices (.parquet)
    └── models/                  ← saved artifacts (.joblib, .pt, .json)
        └── pipeline/            ← PredictorPipeline artifacts (live bot uses these)
```

**Root-level shims:** Files at `src/*.py` that mirror subfolder names (e.g. `src/bot_base.py`,
`src/metrics.py`) are backward-compatibility re-exports. They let existing scripts keep working
without path changes. New code should always import from the canonical subfolder path.

---

## 3. Key Contracts

### 3.1 ModelInterface  `src/models/model_interface.py`

Every prediction model — local or cloud — implements exactly three methods:

```python
class MyModel(ModelInterface):

    def train(self, X: pd.DataFrame, y: pd.Series) -> "MyModel":
        # fit the model; return self
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # return shape (N, 3): [[P_buy, P_hold, P_sell], ...]
        # values must sum to 1.0 per row
        ...

    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> "MyModel": ...
```

The pipeline, backtester, and bots only ever call `predict_proba()`.
They are completely unaware of what's behind it.

### 3.2 BotBase  `src/core/bot_base.py`

Every live bot extends BotBase and implements `on_tick()`:

```python
class MyBot(BotBase):

    def on_start(self) -> None:
        # called once after MT5 connects
        ...

    def on_tick(self) -> None:
        # called every tick_interval seconds (default 60s)
        # put all strategy logic here
        ...

    def on_stop(self) -> None:
        # called on Ctrl+C or daily loss limit
        ...
```

BotBase provides: `self.conn` (MT5Connector), `self.log()`, `self.buy()`, `self.sell()`,
`self.open_positions()`, `self.risk_sized_lot()`, daily loss guard, magic number isolation.

---

## 4. Extension Guide — "Where Do I Put It?"

| What you want to add | File to create / edit | What to implement |
|---|---|---|
| New technical indicator | `src/features/indicators.py` | Add a pure function `def my_ind(df) -> pd.Series` |
| New feature column | `src/features/feature_pipeline.py` | Call your indicator in `_compute()`, append column |
| New local ML model | `src/models/my_model.py` | Implement `ModelInterface` (train/predict_proba/save/load) |
| **Cloud prediction** | `src/models/cloud_model.py` | Implement `ModelInterface`; call remote API in `predict_proba()` |
| New encoder / representation | `src/features/my_encoder.py` | Fit on train split, `transform()` returns extra columns; wire into `PredictorPipeline.build_features()` |
| New ensemble member | `src/models/ensemble.py` | Add to `base_models` list in config.yaml |
| New bot strategy | `src/bots/my_bot.py` | Extend `BotBase`, implement `on_tick()` |
| New risk tier | `config.yaml` | Edit `risk_manager.tiers` list |
| New backtest metric | `src/evaluation/metrics.py` | Add pure function, call from `performance_report()` |
| New data source | `src/core/mt5_connector.py` | Add method (keep same DataFrame index/column format) |

### 4.1 Adding a Cloud Model (step-by-step)

**Step 1** — Create `src/models/cloud_model.py`:

```python
import requests
import numpy as np
import pandas as pd
from src.models.model_interface import ModelInterface

class CloudModel(ModelInterface):
    """Calls a remote inference endpoint for [P_buy, P_hold, P_sell]."""

    def __init__(self, endpoint_url: str = "", api_key: str = ""):
        self.endpoint_url = endpoint_url or os.getenv("CLOUD_ENDPOINT", "")
        self.api_key      = api_key      or os.getenv("CLOUD_API_KEY", "")
        self._feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> "CloudModel":
        self._feature_names = list(X.columns)
        return self  # training happens in the cloud

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        payload = {"features": X.values.tolist(),
                   "columns":  list(X.columns)}
        resp = requests.post(
            self.endpoint_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=5,
        )
        resp.raise_for_status()
        return np.array(resp.json()["probabilities"])  # shape (N, 3)

    def save(self, path) -> None:
        import json, pathlib
        pathlib.Path(path).write_text(
            json.dumps({"endpoint": self.endpoint_url,
                        "feature_names": self._feature_names}))

    def load(self, path) -> "CloudModel":
        import json, pathlib
        d = json.loads(pathlib.Path(path).read_text())
        self.endpoint_url  = d.get("endpoint", self.endpoint_url)
        self._feature_names = d.get("feature_names", [])
        return self
```

**Step 2** — Register in `config.yaml`:

```yaml
active_model: cloud_model        # ← swap here; no code change

models:
  cloud_model:
    type: cloud_model
    path: data/models/cloud_model.json
```

**Step 3** — Add one line in `src/models/model_registry.py` `_build()` method:

```python
"cloud_model": lambda cfg: CloudModel(),
```

**Step 4** — Set env vars in `.env`:

```
CLOUD_ENDPOINT=https://my-api.example.com/predict
CLOUD_API_KEY=sk-...
```

**Tip:** Put the cloud model inside the Ensemble (`config.yaml → ensemble.base_models`)
to blend it with local XGBoost — reduces single-point-of-failure risk if the endpoint
is slow or down.

### 4.2 Adding a New Encoder

Template — copy from `src/features/latent_encoder.py`:

```python
class MyEncoder:
    def fit(self, df: pd.DataFrame, y=None) -> "MyEncoder": ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns DataFrame with extra columns (e.g. regime_0, regime_1, ...)
        # Same index as input — no row dropping
        ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> "MyEncoder": ...
```

Wire it in `src/pipeline.py` `build_features()` alongside `LatentEncoder`.

---

## 5. Config Reference  `config.yaml`

```
pipeline:
  model_type:        xgboost | lightgbm | catboost | ensemble  ← swap model
  data_path:         path to raw CSV
  train_frac:        0.8  (80% train / 20% test for scalers + encoder)

  encoder:
    enabled:         true | false   ← toggle latent encoder
    mode:            supervised | autoencoder
    latent_dim:      8              ← size of regime fingerprint
    window_size:     50             ← OHLCV bars per window
    epochs:          30

  walk_forward:
    window_type:     expanding | sliding
    train_days:      180
    test_days:       30

  backtest:
    threshold:       0.40           ← min confidence to open trade
    sl_pips:         30.0
    tp_pips:         60.0

  artifacts:
    directory:       data/models/pipeline   ← live bot loads from here

risk_manager:
  tiers:             [[0.75, 0.02], [0.65, 0.015], ...]   ← confidence→risk%
  use_kelly:         false
  use_atr_stop:      false

active_model:        xgboost   ← used by standalone ModelRegistry (non-pipeline)
```

---

## 6. Key Commands

```bash
# Train the full pipeline (encoder + model) and save artifacts
conda run -n envmt5 python scripts/run_pipeline.py train

# Walk-forward backtest (evaluate without saving)
conda run -n envmt5 python scripts/run_pipeline.py backtest

# Single live prediction on latest bars from CSV
conda run -n envmt5 python scripts/run_pipeline.py predict

# Run live bot — dry run (no orders)
/home/rock/anaconda3/envs/envmt5/bin/python -u src/bots/pipeline_bot.py --dry-run

# Run live bot — real orders
/home/rock/anaconda3/envs/envmt5/bin/python -u src/bots/pipeline_bot.py

# Monitor live bot log
tail -f /tmp/pipeline_bot.log

# Train the latent encoder standalone
conda run -n envmt5 python scripts/train_autoencoder.py --mode supervised --epochs 30

# A/B comparison: baseline vs. supervised latent encoder
conda run -n envmt5 python scripts/compare_latent_vs_baseline.py
```

---

## 7. Rules — Never Break These

| Rule | Why |
|---|---|
| `src/core/bot_base.py` and `src/core/mt5_connector.py` are FROZEN | All bots depend on their contracts; changes ripple everywhere |
| `ModelInterface` contract is FROZEN | Every model, backtest, walk-forward, and bot depends on `predict_proba(X) → (N,3)` |
| `data/EURUSD_M15.csv` is READ-ONLY | Immutable raw data; all derived files go in `data/features/` or `data/models/` |
| No lookahead in features or encoder | Window for bar `t` = rows `[t-W, t-1]`; first W-1 bars get zeros |
| Scaler + encoder fit on TRAIN portion only | Prevents leakage; `pipeline.build_features()` enforces the split |
| New models go in `src/models/` | Keeps the contract discoverable; registry auto-picks them up |
| New bots go in `src/bots/` | All share the same BotBase safety guards (loss limit, magic number) |
| Config changes in `config.yaml` only | No hard-coded values in source files |
