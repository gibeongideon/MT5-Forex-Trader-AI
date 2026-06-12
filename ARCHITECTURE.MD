# ARCHITECTURE.MD

## MT5 Trading Bot — Developer Reference

> Single source of truth for codebase structure, contracts, and extension points.
> Read this before adding any new component.
> Last updated: 2026-06-12

---

## 1. Three-Tier Model Architecture

The system has three distinct model families, each with different horizons and roles:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — CatBoost Candle Predictor (1-bar specialist)                       │
│                                                                              │
│  Horizon: 1 M15 bar (~15 minutes)                                            │
│  Input: 52 features (32 base + 6 session flags + 6 MTF EMA + fractal_corr   │
│          + 7 more)                                                           │
│  Window: 50-bar lookback, sliding 120d training window                       │
│  Output: P_buy, P_hold, P_sell (softmax)                                     │
│  Threshold: 0.60 min confidence to trade                                     │
│  Artifacts: data/models/candle_{SYMBOL}/                                     │
│    ├── scaler.joblib                                                         │
│    ├── encoder.pt          (supervised enc8 for candle features)             │
│    ├── model.joblib        (CatBoost)                                        │
│    ├── meta.json           (feature_cols, threshold, sl_pips, tp_pips)       │
│    └── pair_meta.json      (version, session params, trail params)           │
│                                                                              │
│  Modes: candle_predictor — force-close at 1 bar, SL=10p/TP=30p              │
│         candle_trail     — trailing SL + confidence-tier hold (1/2/4 bars)  │
├──────────────────────────────────────────────────────────────────────────────┤
│  TIER 2 — XGBoost Pipeline v1 (40-feature baseline)                         │
│                                                                              │
│  Horizon: 4 M15 bars (~1 hour)                                               │
│  Input: 32 base features + 8 enc8 latent dims = 40 features                 │
│  Artifacts: data/models/pipeline_{SYMBOL}/                                   │
│  Status: Superseded by v2 — kept as fallback only                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  TIER 3 — Hybrid v2 (42-feature, XGBoost + enc8 + candle injection) ★       │
│                                                                              │
│  Horizon: 4 M15 bars (~1 hour)                                               │
│  Input: 32 base features + 8 enc8 latent dims + 2 candle probs = 42 feats   │
│  candle_p_buy, candle_p_sell: OOS predictions from Tier 1 model             │
│  Artifacts: data/models/pipeline_{SYMBOL}_v2/                                │
│  Status: CURRENT CHAMPION — use for all new deployments                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow (Raw OHLCV → Trade)

```text
CSV / MT5 live bars (OHLCV DataFrame)
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Feature Engineering          src/features/               │
│  FeaturePipeline                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  RSI(14)  MACD  BB(20)  ATR(14)  ADX  SMA(9/21/50)  EMA(20)        │
│  OBV  Stoch  fractal_corr  session_flags(6)  MTF_EMA(6)             │
│  StandardScaler fitted on train portion only (no leakage)           │
│  → 32 base features                                                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ 32 features
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Supervised Encoder (enc8)    src/models/encoder.py       │
│  ─────────────────────────────────────────────────────────────────  │
│  Input: 50-bar rolling window of 5 OHLCV values = 250 input dims    │
│  Architecture: MLP, 250→128→64→8, CrossEntropyLoss on 4-bar labels  │
│  Output: 8 latent dimensions                                         │
│  Trained on train portion only, same split as FeaturePipeline       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ 40 features (32 base + 8 latent)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Candle Feature Injection     (hybrid v2 only)            │
│  ─────────────────────────────────────────────────────────────────  │
│  Source: data/features/candle_signal_{SYMBOL}.parquet               │
│          (51,689 OOS predictions, indexed by timestamp)             │
│  Injected: candle_p_buy, candle_p_sell (two new columns)            │
│  Fallback: 0.5 (neutral) when no signal available                   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ 42 features (hybrid v2)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — XGBoost Classifier           src/models/                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Input: 42-feature vector per bar                                    │
│  Output: P(buy), P(hold), P(sell) via predict_proba                  │
│  Threshold: cfg.threshold (default 0.40) to generate signal          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ signal dict {direction, confidence}
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Risk Manager                 src/risk/                   │
│  ─────────────────────────────────────────────────────────────────  │
│  Tiered position sizing: 0.5% / 1.0% / 2.0% based on confidence     │
│  SL/TP calculation from cfg.sl_pips / tp_pips + bt_pip_size         │
│  Flip mode routing: always / hedge_exit / hedge_loss / trailing_hedge│
│                     candle_predictor / candle_trail / lock / etc.    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ trade order
                                   ▼
MT5 via rpyc bridge (mt5linux)  →  live_trades.db (SQLite journal)
```

---

## 3. Key Files

| File | Role |
| ---- | ---- |
| `src/pipeline.py` | `PredictorPipeline`: build_features / walk_forward / predict / save / load |
| `src/bots/pipeline_bot.py` | Live bot: all flip modes, hybrid injection, candle standalone |
| `src/features/feature_pipeline.py` | FeaturePipeline: 32-dim indicator matrix + StandardScaler |
| `src/features/indicators.py` | Individual indicator functions |
| `src/models/encoder.py` | LatentEncoder (MLP enc8, supervised) |
| `scripts/train_candle_model.py` | Train + WF validate CatBoost candle model |
| `scripts/retrain_champion.py` | Retrain XGBoost champion on all data (auto-injects candle features) |
| `scripts/build_candle_features.py` | Generate OOS candle signal parquet (13 fold models) |
| `scripts/compare_candle_features.py` | WF A/B comparison: with vs without candle features |
| `scripts/backtest_candle_trail.py` | Bar-by-bar candle_trail backtest |
| `scripts/tune_candle_trail.py` | Grid search for optimal candle_trail params |
| `scripts/verify_candle_oos.py` | Independent OOS Sharpe verification of candle model |
| `config.yaml` | Global config (pipeline, risk_manager, bt_pip_size per symbol) |
| `data/live_trades.db` | SQLite trade journal |

---

## 4. PipelineConfig Fields (src/pipeline.py)

Key fields for multi-symbol support:

```python
@dataclass
class PipelineConfig:
    model_type: str = "xgboost"
    encoder_mode: str = "supervised"
    encoder_latent_dim: int = 8
    encoder_epochs: int = 30
    data_path: str = "data/EURUSD_M15.csv"
    artifacts_dir: str = "data/models/pipeline"
    threshold: float = 0.40
    sl_pips: float = 30.0
    tp_pips: float = 60.0
    bt_pip_size: float = 0.0001   # 0.0001 for EURUSD/GBPUSD, 0.01 for USDJPY
    # ... risk manager fields
```

`bt_pip_size` MUST be set correctly per symbol when running `compare_candle_features.py` or any script that calls `_make_backtest_cfg()`. Wrong value causes SL/TP to be in wrong price units → garbage trade counts and −100% MaxDD.

---

## 5. Candle Feature Injection (retrain + inference)

### During retraining (`scripts/retrain_champion.py`)

```python
# After build_features() returns X (40 cols), before fit_full():
candle_feat_path = ROOT / "data" / "features" / f"candle_signal_{symbol}.parquet"
if candle_feat_path.exists():
    cf = pd.read_parquet(candle_feat_path)[["candle_p_buy", "candle_p_sell"]]
    shared = X.index.intersection(cf.index)
    X = pd.concat([X.loc[shared], cf.loc[shared]], axis=1)
    y = y.reindex(shared)
    pipe._feature_cols = list(X.columns)   # CRITICAL: sync meta.json to 42 cols
```

If `_feature_cols` is not updated after injection, `meta.json` records 40 features but the model was trained on 42. At inference, `predict()` silently drops candle columns. Always update `_feature_cols` after injection.

### During inference (`src/pipeline.py :: predict()`)

```python
def predict(self, df_raw: pd.DataFrame,
            candle_signal: dict | None = None) -> dict:
    ...
    _has_candle_feats = self._feature_cols and "candle_p_buy" in self._feature_cols
    if _has_candle_feats:
        if candle_signal is not None:
            X_live["candle_p_buy"]  = candle_signal.get("P_buy",  0.5)
            X_live["candle_p_sell"] = candle_signal.get("P_sell", 0.5)
        else:
            X_live["candle_p_buy"]  = 0.5   # neutral fallback
            X_live["candle_p_sell"] = 0.5
```

Old v1 models (pipeline_EURUSD/ without candle cols in `_feature_cols`) work unchanged — the `_has_candle_feats` guard prevents injection.

### In the live bot (`src/bots/pipeline_bot.py :: on_tick()`)

```python
# Step 4 — standard pipeline path
candle_sig = None
if self._candle_feature_inject and self.candle_pipe is not None:
    try:
        candle_sig = self.candle_pipe.predict(ohlcv)
    except Exception:
        pass
sig = self.pipe.predict(ohlcv, candle_signal=candle_sig)
```

`_candle_feature_inject` is True only when `--candle-feature-dir` is passed and flip_mode is NOT `candle_predictor` or `candle_trail` (those modes use the candle model for trading decisions directly, not feature injection).

---

## 6. Adding a New Feature

1. Add indicator function to `src/features/indicators.py`
2. Add column to `src/features/feature_pipeline.py::build()` — append after existing columns
3. Update `_FEATURE_NAMES` list if it exists in that file
4. Delete WF cache: `rm -rf data/models/wf_cache_*`
5. Run comparison: `scripts/compare_candle_features.py --symbol EURUSD`
6. If improvement confirmed, retrain champion: `scripts/retrain_champion.py`

Do NOT add features derived from OHLCV alone — the Saturation Principle (22 confirmed experiments) shows enc8 has absorbed all OHLCV signal. Only features from genuinely different sources (model outputs, external data, deterministic regime rules) can improve enc8.

---

## 7. Walk-Forward Setup

```text
Expanding window — each fold extends training by 1 test period:

Fold 1:  |──180d train──|──30d test──|
Fold 2:  |────210d train────|──30d test──|
Fold 3:  |───────240d train───────|──30d test──|
...
Fold 13: |──────────────────540d train──────────────────|──30d test──|

13 folds total on 60,000 M15 bars (~2.4 years of data)
```

WF OOS concatenation: each fold's OOS predictions are concatenated chronologically. No bar appears in both train and test across the entire run. Final Sharpe/MaxDD is computed on the full concatenated OOS series.

---

## 8. OOS Candle Signal Generation (build_candle_features.py)

The candle parquets in `data/features/` are generated from the 13 WF fold models stored in `data/models/wf_cache_candle2_{SYMBOL}/`. Each fold model predicts only its OOS window. This ensures no lookahead — a bar's candle probability is always predicted by a model that never trained on that bar.

Regenerate after any candle model retrain:

```bash
conda run -n envmt5 python scripts/build_candle_features.py
```

Then retrain the hybrid v2 champion (picks up the new parquet automatically):

```bash
conda run -n envmt5 python scripts/retrain_champion.py \
    --data data/EURUSD_M15.csv --out data/models/pipeline_EURUSD_v2
```

---

## 9. Flip Modes Reference

| Mode | Description | Uses candle? |
| ---- | ----------- | ------------ |
| `always` | Open every signal, no exit filter | No |
| `hedge_exit` | Hedge losing trade then exit both at signal flip | No |
| `hedge_loss` | Hedge only when trade is at loss | No |
| `trailing_hedge` | Hedge with trailing SL on the hedge leg | No |
| `candle_predictor` | 1-bar CatBoost, force-close after 1 bar, SL/TP | Tier 1 only |
| `candle_trail` | 1-bar CatBoost + trailing SL + conf-tier hold | Tier 1 only |
| `lock` | Lock in profit at fixed threshold | No |
| `ratio_hedge` | Variable ratio hedge sizing | No |
| `partial_close` | Close partial position at profit targets | No |
| `zone_recovery` | Zone-based recovery grid | No |

For hybrid v2, use any non-candle flip mode and pass `--candle-feature-dir`. The candle model runs as a feature source, not a trade manager.

---

## 10. Config.yaml Structure

```yaml
pipeline:
  model_type: xgboost
  encoder_mode: supervised
  encoder_latent_dim: 8
  encoder_epochs: 30
  threshold: 0.40
  sl_pips: 30.0
  tp_pips: 60.0
  data_path: data/EURUSD_M15.csv
  artifacts_dir: data/models/pipeline
  backtest:
    pip_size: 0.0001     # bt_pip_size — MUST match symbol (USDJPY = 0.01)

risk_manager:
  max_risk_pct: 0.02
  min_risk_pct: 0.005
```
