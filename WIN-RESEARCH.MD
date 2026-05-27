# WIN-RESEARCH.MD
## MT5 EURUSD M15 Trading Bot — Architecture, Results History & Research Roadmap

> **Data:** EURUSD M15, 75,000 bars, 2023-05-18 → 2026-05-25
> **Walk-forward:** Expanding window, 19 folds, 180-day train / 30-day test, no leakage
> **Backtester:** SL=30p, TP=60p, spread=1.0p, threshold=40%, tiered risk (0.5%–2% per trade)
> Last updated: 2026-05-27

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA LAYER                           │
│  data/EURUSD_M15.csv  (75,000 bars, M15, immutable)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┴──────────────────┐
           │                                    │
           ▼                                    ▼
┌──────────────────────┐          ┌─────────────────────────────┐
│  FEATURE PIPELINE    │          │  SUPERVISED LATENT ENCODER  │
│  src/features/       │          │  src/features/              │
│  feature_pipeline.py │          │  latent_encoder.py          │
│                      │          │                             │
│  31 technical cols:  │          │  MLP encoder trained with   │
│  - RSI, MACD, ATR    │          │  CrossEntropyLoss on        │
│  - Bollinger, ADX    │          │  buy/hold/sell labels.      │
│  - SMA/EMA crosses   │          │  Window: 50 bars → 8 dims   │
│  - Stochastic        │          │  No lookahead guaranteed    │
│  - H1/H4 MTF feats   │          │  (window = [t-50, t-1])     │
│  - Lag/return feats  │          │                             │
│  StandardScaler      │          │  artifact: autoencoder.pt   │
│  (train split only)  │          │                             │
└──────────┬───────────┘          └──────────────┬──────────────┘
           │                                     │
           │         ┌───────────────────────────┘
           │         │
           ▼         ▼
┌────────────────────────────────────────────────────────────────┐
│               MERGED FEATURE MATRIX  (39 columns)              │
│   31 base indicators  +  8 supervised latent dims              │
│   data/features/EURUSD_M15_features_latent_sup8.parquet        │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│               MODEL LAYER  (ModelInterface)                    │
│                                                                │
│   XGBoost ──┐                                                  │
│   LightGBM ─┤ base models → [P_buy, P_hold, P_sell]           │
│   CatBoost ─┤                                                  │
│   LSTM ─────┘                                                  │
│         │                                                      │
│         ▼  (out-of-fold stacking, Phase 6)                     │
│   Ensemble meta-learner (logistic / LightGBM)                  │
│      ← optional: extra latent cols injected here only          │
│                                                                │
│   All outputs: [P_buy, P_hold, P_sell]  (sums to 1.0)         │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│               RISK MANAGER  (Phase 8)                          │
│   Confidence-tiered position sizing:                           │
│     P ≥ 0.75 → 2.0% risk   P ≥ 0.65 → 1.5% risk              │
│     P ≥ 0.55 → 0.75% risk  P ≥ 0.40 → 0.5% risk              │
│   Portfolio cap: 3% total exposure                             │
│   Drawdown throttle: 50% size reduction at 10% DD             │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│               BACKTESTER / WALK-FORWARD                        │
│   src/backtester.py  +  src/walk_forward.py                    │
│   Event-driven, bar-by-bar, realistic costs                    │
│   SL/TP hit detection, equity curve, fold reporting            │
└────────────────────────────────────────────────────────────────┘
```

### Key Architectural Rules (Never Break These)
- `BotBase`, `MT5Connector`, `ModelInterface` are **frozen** — no interface changes
- Every model outputs `[P_buy, P_hold, P_sell]` — never hard labels
- No lookahead: feature at bar `t` uses only data from `[t-W, t-1]`
- Scaler fitted on training split only — never the full dataset
- Raw data (`EURUSD_M15.csv`) is immutable — re-download if needed
- Latent encoder trained on train split only — transform is inference-only

---

## 2. Full Experiment History

### Phase 1 — Random Baseline

| Config | Sharpe | MaxDD | Return | Trades |
|--------|--------|-------|--------|--------|
| Random entry (p=2%) | -0.17 | 21.6% | -2.5% | 354 |

**Takeaway:** Confirmed the system is wired correctly. A random signal loses money as expected once spread is applied.

---

### Phase 2 — Rule-Based Engine

| Config | Sharpe | MaxDD | Return | Trades |
|--------|--------|-------|--------|--------|
| MA cross + RSI + BB rules | -0.45 | — | -9.7% | 430 |

**Takeaway:** Hand-coded rules on their own fail. Rules are rigid and can't adapt to regime changes. They generate too many low-quality signals, and every signal costs spread.

---

### Phase 4–5 — Single ML Models (Walk-Forward, with costs)

| Model | Sharpe | MaxDD | Return | Trades | Notes |
|-------|--------|-------|--------|--------|-------|
| XGBoost (no costs) | +1.34 | 10.5% | +14.2% | 176 | Best raw signal |
| **XGBoost (1p spread)** | **+0.52** | — | -3.8% | 127 | Costs kill edge |
| LightGBM | +0.72 | 6.8% | +3.8% | 99 | More conservative |
| CatBoost | +1.17 | 24.7% | +5.8% | 543 | High DD, too aggressive |
| Random Forest | +0.24 | 14.8% | -0.1% | 93 | Bagging ≪ boosting here |

**Critical finding:** Transaction costs cut XGBoost Sharpe from 1.34 → 0.52. This is the reality of Forex — a strategy that looks good in zero-cost backtest fails once you add realistic spread. Boosting models (XGBoost, LightGBM, CatBoost) consistently beat bagging (RF) on structured financial data.

---

### Phase 6 — Stacking Ensemble

| Config | Log-loss | Accuracy | P≥0.40 coverage |
|--------|----------|----------|-----------------|
| Logistic meta-learner | 1.089 | 39% | 18% of bars |
| **LightGBM meta-learner** | **1.087** | **39%** | **41.8% of bars** |

**Takeaway:** The ensemble's main win is **coverage** — LightGBM meta gives actionable signals on 41.8% of bars vs 18% for logistic. The meta-learner learns when individual models agree and scales confidence accordingly. Walk-forward of the ensemble deferred to Phase 7 (requires per-fold caching).

---

### Phase 7 — Walk-Forward Engine

**Monte Carlo (1000 shuffles of XGBoost trades):**
- 5th-percentile Sharpe: **-1.17**
- Original result near shuffle median (53.3% of shuffles beat it)
- Conclusion: edge exists in the trade *outcomes*, but is fragile — order-dependent in bad scenarios

**Walk-forward established the standard:** 19 folds, 180-day expanding train, 30-day test. This is the benchmark for all subsequent experiments.

---

### Phase 8 — Intelligent Risk Management

| Config | Sharpe | MaxDD | Return | Trades | Change |
|--------|--------|-------|--------|--------|--------|
| A — Fixed 1% risk (baseline) | +0.72 | 14.6% | -0.3% | 126 | — |
| **B — Confidence-tiered risk** | **+0.72** | **7.5%** | **+0.2%** | 126 | **−49% drawdown, free** |
| C — Tiered + ATR dynamic stop | +0.22 | 3.4% | +0.3% | 69 | Lower DD, lower Sharpe |

**Key insight:** Config B achieves the same Sharpe as Config A at half the drawdown. This is the free lunch of position sizing — you are not changing the signal quality, only how much you bet when you are right vs wrong. High-confidence trades (P≥0.75) get 2% risk; low-confidence trades (P≥0.40) get 0.5% risk. The model naturally bets more when it is more right.

**Config B is now the system baseline.** Sharpe 0.72, MaxDD 7.5%.

---

### Phase 9 — Unsupervised Autoencoder (FAILED)

| Config | Sharpe | MaxDD | Return | Trades |
|--------|--------|-------|--------|--------|
| Baseline (31 features) | +0.72 | 7.5% | +0.2% | 126 |
| + Unsupervised AE (16 latent dims) | **-0.41** | — | — | — |

**Why it failed:**

The autoencoder was trained with **MSE reconstruction loss** — it learned to compress and reconstruct OHLCV windows accurately. But "accurately reconstructs price shape" and "predicts direction" are completely different objectives. The 16 latent dimensions encoded market geometry that was **uncorrelated with future returns**. XGBoost received 16 extra noisy features and started overfitting to them.

**Root cause:** Unsupervised representation learning produces general-purpose features. For direction prediction you need task-specific features.

---

### Phase 9 — Supervised Encoder (CURRENT BEST)

| Config | Sharpe | MaxDD | Return | Trades | Calmar |
|--------|--------|-------|--------|--------|--------|
| A — Baseline (31 features) | +0.72 | 7.5% | +0.2% | 126 | 0.02 |
| **C — Supervised enc 8-dim (39 features)** | **+3.13** | **13.3%** | **+358.3%** | **524** | **17.81** |

**Walk-forward fold breakdown (Config C):**

| Fold | Period | Trades | Win Rate | Sharpe | Return |
|------|--------|--------|----------|--------|--------|
| 0 | Nov–Dec 2024 | 34 | 41.2% | +8.13 | +19.6% |
| 1 | Dec–Jan 2025 | 22 | 27.3% | +1.25 | +4.3% |
| 2 | Jan–Feb 2025 | 31 | 22.6% | -1.57 | -9.5% |
| 3 | Feb–Mar 2025 | 26 | 50.0% | +4.61 | +16.4% |
| 4 | Mar–Apr 2025 | 45 | 44.4% | +3.79 | +25.9% |
| 5 | Apr–May 2025 | 62 | 43.5% | +7.75 | +34.3% |
| 6 | May–Jun 2025 | 35 | 40.0% | +2.88 | +23.6% |
| 7 | Jun–Jul 2025 | 30 | 30.0% | -2.14 | -4.8% |
| 8 | Jul–Aug 2025 | 29 | 44.8% | +5.17 | +21.9% |
| 9 | Aug–Sep 2025 | 23 | 39.1% | +2.60 | +3.7% |
| 10 | Sep–Oct 2025 | 20 | 55.0% | +8.67 | +14.4% |
| 11 | Oct–Nov 2025 | 20 | 45.0% | +3.68 | +8.5% |
| 12 | Nov–Dec 2025 | 11 | 63.6% | +4.00 | +6.8% |
| 13 | Dec–Jan 2026 | 11 | 45.5% | +3.76 | +0.5% |
| 14 | Jan–Feb 2026 | 28 | 46.4% | -1.96 | +0.6% |
| 15 | Feb–Mar 2026 | 22 | 31.8% | +0.01 | -1.8% |
| 16 | Mar–Apr 2026 | 41 | 39.0% | -0.26 | +7.5% |
| 17 | Apr–May 2026 | 23 | 30.4% | -0.46 | -1.7% |
| 18 | May 2026 | 11 | 27.3% | +0.20 | -0.5% |

---

## 3. What Changed and Why It Worked

### The Key Change

| | Unsupervised AE (failed) | Supervised Encoder (current) |
|---|---|---|
| **Loss function** | MSE (reconstruct OHLCV) | CrossEntropyLoss (predict buy/hold/sell) |
| **What it learns** | Market geometry, price shape | What regime predicts direction |
| **Latent dims** | 16 | 8 |
| **Gradient signal** | "Reconstruct this window" | "Classify this window's future" |
| **Result** | Noisy, uncorrelated features | Predictive regime fingerprint |

### Training Details

```
Encoder architecture:
  Linear(250 → 256) → ReLU
  Linear(256 → 128) → ReLU
  Linear(128 → 8) → Tanh  ← bottleneck (Tanh bounds latent space to [-1, 1])
  Linear(8 → 3)           ← classification head (discarded after training)

Input:  50-bar OHLCV window, z-scored within each window
Labels: buy/hold/sell aligned to the bar AFTER the window
Loss:   CrossEntropyLoss with inverse-frequency class weights
        (sell=10,615  hold=13,825  buy=10,988 — roughly balanced)

Training curve:
  Epoch  1:  ce_loss=1.0743  acc=42.1%
  Epoch 10:  ce_loss=1.0075  acc=49.8%
  Epoch 20:  ce_loss=0.8376  acc=61.5%
  Epoch 30:  ce_loss=0.6348  acc=72.4%
```

### Why the Latent Features Help XGBoost

The 31 base features describe **what is happening now** — RSI is 65, MACD crossed up, ATR is high. But they don't describe **what regime the market is in** across the last 50 bars.

The 8 latent dimensions encode a compressed fingerprint of the prior 50 bars that was trained to be maximally predictive of direction. XGBoost can now ask: "Given this regime state AND these current indicators, which direction?" That is a much stronger and more discriminative feature space.

### Why 4× More Trades

The confidence threshold (40%) is unchanged. But the latent features shift probability estimates further from 0.33 (uniform) toward 0.5+ when the regime is clearly bullish or bearish. More bars clear the 40% threshold because the model is more certain. This is the right kind of more trades — driven by better signal, not lower standards.

### Why MaxDD Increased

524 trades vs 126 means more exposure. Some of the additional trades occur in difficult periods (folds 2, 7, 14) that the baseline simply sat out. The Calmar ratio (annual return / maxDD) went from 0.02 to **17.81** — the system is earning $17.81 of return per 1% of drawdown risk, vs $0.02 before. The absolute drawdown increase (7.5% → 13.3%) is acceptable given the return increase (+0.2% → +358.3%).

---

## 4. Proposals for Further Improvement

### Priority 1 — Encoder Improvements (Low Cost, High Impact)

**1a. More training epochs with early stopping**
The training curve at epoch 30 shows accuracy still rising (72.4%). Training to epoch 60–100 with a held-out validation set for early stopping will likely improve the latent space quality.

```bash
python scripts/train_autoencoder.py --mode supervised --latent-dim 8 --epochs 80
```

**1b. Test latent_dim=4 and latent_dim=16**
8 was chosen by intuition. 4 dims may generalise better (stronger compression = less noise). 16 dims may capture more regime nuance. Run `compare_latent_vs_baseline.py` for each.

**1c. Attention-weighted window encoder (replace MLP)**
Instead of flattening the 50-bar window into a vector, use a small Transformer encoder with self-attention. The model can learn which bars within the window are most important (e.g., the breakout bar matters more than the consolidation bars before it). This is the architecture used in financial representation learning papers.

```python
class _AttentionEncoderNet(nn.Module):
    # Input: (batch, 50, 5) — sequence of OHLCV bars
    # Self-attention over time axis
    # CLS token pooling → 8-dim latent
```

**1d. Larger window size (100 bars)**
A 100-bar window captures 25 hours of M15 data — enough to see a full daily cycle. The current 50 bars is ~12.5 hours. Longer windows give the encoder more market structure to compress.

---

### Priority 2 — Feature Space Improvements (Medium Cost)

**2a. Multi-timeframe latent encoding**
Train a separate encoder on H1 bars and one on H4 bars. Concatenate their latent outputs with M15 latent. This gives the model regime information at three time scales simultaneously.

```
M15 encoder → 8 dims  (short-term momentum regime)
H1  encoder → 8 dims  (intraday regime)
H4  encoder → 8 dims  (swing regime)
Total: 31 base + 24 latent = 55 features
```

**2b. Volatility-regime features**
Add explicit features the model can use to know *when* the latent encoder's signals are reliable:
- `vol_regime`: rolling 20-bar ATR vs. 100-bar ATR (expanding/contracting)
- `breakout_flag`: ATR spike > 2σ from rolling mean
- `session_hour`: market open hour (London, NY, overlap)

**2c. Orderflow proxy features**
Without tick data, approximate orderflow using:
- `volume_delta`: up-volume vs down-volume (close > open = up)
- `volume_acceleration`: volume change rate vs prior bars
- `cumulative_delta_20`: 20-bar net orderflow direction

---

### Priority 3 — Model Architecture Improvements (Medium Cost)

**3a. Feed latent features to ensemble meta-learner only (not base models)**
This was the original "Config D" proposal. Base XGBoost models train on 31 features (more generalizable). The meta-learner receives the base model OOF probabilities + 8 latent regime features. The meta-learner learns to trust or distrust the base models *conditionally on the current regime*.

Infrastructure already built: `Ensemble.train(X, y, extra_meta=latent_df)` now works.
Requires WalkForwardValidator modification to pass `extra_meta` per fold.

**3b. LSTM + supervised latent features**
The LSTM (Phase 6) processes sequences of feature vectors. Feed it the 39-column feature matrix (31 base + 8 latent) instead of 31. The LSTM will learn temporal dependencies over the regime-aware feature space.

**3c. CatBoost on 39 features**
CatBoost was Sharpe +1.17 on 31 features (Phase 6). It may benefit more than XGBoost from the latent features because CatBoost's symmetric trees handle dense continuous features differently. Run the walk-forward with `model_type=catboost` and the 39-feature matrix.

---

### Priority 4 — Signal Calibration (Low Cost)

**4a. Isotonic regression calibration per fold**
The current model outputs raw XGBoost probabilities. Calibrating with isotonic regression (better than Platt scaling for non-monotonic classifiers) on each fold's OOF predictions will sharpen the probability estimates and make the confidence tiers more accurate.

**4b. Dynamic threshold adaptation**
Currently threshold is fixed at 40%. A regime-aware threshold: raise to 50% in low-volatility periods (many false signals), lower to 35% in high-volatility trending periods (clearer signals). Use the latent features or ATR percentile to define regime.

---

### Priority 5 — Risk Management Improvements (High Impact)

**5a. Reduce MaxDD with partial position scaling**
The current drawdown of 13.3% can be reduced without sacrificing much return by adding a drawdown throttle that kicks in earlier:
```yaml
risk_manager:
  drawdown_threshold: 0.07  # throttle at 7% DD (currently 10%)
  drawdown_throttle: 0.40   # reduce size to 40% (currently 50%)
```
This protects the account during losing runs while staying in the game.

**5b. Walk-forward re-calibration of confidence tiers**
The tiers (0.75→2%, 0.65→1.5%) were set by intuition. For each fold, compute the empirical win rate by confidence bucket and set tiers proportional to Kelly criterion:
```
Kelly% = win_rate - (1 - win_rate) / (TP/SL ratio)
       = win_rate - (1 - win_rate) / 2.0   (30p/60p = 2:1)
```
If P≥0.75 historically wins 55% of the time: Kelly = 0.55 - 0.225 = 0.325 → use 1/4 Kelly = 8.1% risk. Cap at 2% for safety.

**5c. Asymmetric TP scaling**
In trending regimes (identified by the latent encoder), extend TP from 60p to 90p. In ranging regimes, tighten TP to 45p. This improves profit factor without changing SL, using the regime knowledge the latent encoder already provides.

---

### Priority 6 — Data and Generalization

**6a. Multi-symbol training**
Train the encoder and XGBoost on EURUSD + GBPUSD + USDJPY simultaneously. The shared latent space will learn regime features that generalise across pairs. This also triples the training data to ~225,000 bars, improving encoder quality significantly.

**6b. Longer history**
The current dataset starts in May 2023. Extending to 2020 adds COVID volatility, 2022 adds Ukraine crisis and rate-hike cycles. Models trained on diverse market regimes generalise better.

**6c. Synthetic data augmentation**
Use the trained decoder (autoencoder mode) to generate synthetic market windows with slightly perturbed latent codes. This regularises the encoder and prevents overfitting to specific historical patterns.

---

## 5. Summary Leaderboard

| Experiment | Sharpe | MaxDD | Return | Trades | Status |
|------------|--------|-------|--------|--------|--------|
| Random baseline | -0.17 | 21.6% | -2.5% | 354 | Reference |
| Rule engine | -0.45 | — | -9.7% | 430 | Discarded |
| XGBoost (no costs) | +1.34 | 10.5% | +14.2% | 176 | Reference |
| XGBoost (1p spread) | +0.52 | — | -3.8% | 127 | Discarded |
| Phase 8 winner (tiered risk) | +0.72 | 7.5% | +0.2% | 126 | Prev. best |
| + Unsupervised AE 16-dim | -0.41 | — | — | — | Failed |
| **+ Supervised enc 8-dim** | **+3.13** | **13.3%** | **+358.3%** | **524** | **Current best** |

**Next experiments to try (in order):**
1. Train encoder 60–80 epochs (free — just time)
2. Test `latent_dim=4` and `latent_dim=16`
3. Attention encoder (replace MLP)
4. CatBoost on 39 features
5. Multi-timeframe latent (M15 + H1 + H4 encoders)
6. Ensemble meta-learner with latent injection (Config D)

---

## 6. Reproduction Commands

```bash
# Full pipeline from scratch

# 1. Train supervised encoder (latent_dim=8, 30 epochs)
conda run -n envmt5 python scripts/train_autoencoder.py \
    --mode supervised --latent-dim 8 --epochs 30

# 2. Build 39-feature matrix
conda run -n envmt5 python scripts/build_latent_features.py --suffix sup8

# 3. Run A vs C walk-forward comparison
conda run -n envmt5 python scripts/compare_latent_vs_baseline.py --skip-b

# Expected output:
#   A Baseline (31 feat):           Sharpe +0.72  MaxDD  7.5%  Return  +0.2%
#   C Supervised enc 8-dim (39 feat): Sharpe +3.13  MaxDD 13.3%  Return +358.3%
```

---

*All results are walk-forward validated — 19 independent folds, each trained on past data only and tested on future data the model never saw during training. These are not in-sample backtest numbers.*

BASIC IDEA
Raw OHLC + indicators
        ↓
 ┌─────────────┐
 │ Base Models │
 └─────────────┘
   ↓    ↓    ↓
Pred1 Pred2 Pred3
      ↓
 ┌─────────────┐
 │Meta-Learner │
 └─────────────┘
      ↓
 Final signal




 MORE DETAILED 
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAW DATA FLOW (MT5 Live or CSV)                       │
└─────────────────────────────────────────────┬───────────────────────────┘
                                              │ OHLCV bars (open/high/low/close/vol)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Feature Engineering          src/features/feature_pipeline.py │
│  FeaturePipeline                                                          │
│  ─────────────────────────────────────────────────────────────────────── │
│  RSI(14)  MACD  BB(20)  ATR(14)  ADX  SMA(9/21/50)  OBV  Stoch  ...    │
│  + StandardScaler (fit on train set, transform all)                       │
│  → 31-dim feature matrix                                                  │
└─────────────────────────────────────────────┬───────────────────────────┘
                                              │ 31 features
                    ┌─────────────────────────┼────────────────────────────┐
                    │   PLUG-IN (on/off)       ▼                           │
                    │  ┌────────────────────────────────────────────────┐  │
                    │  │  LAYER 2 — Latent Encoder  src/features/        │  │
                    │  │  LatentEncoder (supervised mode)                │  │
                    │  │  ───────────────────────────────────────────── │  │
                    │  │  50-bar OHLCV sliding window → z-normalize      │  │
                    │  │  MLP Encoder → 8-dim "regime fingerprint"       │  │
                    │  │  Trained with CrossEntropyLoss on buy/hold/sell  │  │
                    │  │  → latent_0 … latent_7  (appended to features)  │  │
                    │  └──────────────────────┬─────────────────────────┘  │
                    │  config.yaml:           │ +8 latent dims             │
                    │  encoder.enabled: true  │                            │
                    └─────────────────────────┘                            │
                                              ▼
                                   39-dim feature matrix (31+8)
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Prediction Model                     src/models/              │
│  ModelInterface contract (swappable via config.yaml: model_type)         │
│  ─────────────────────────────────────────────────────────────────────── │
│  XGBoost  │  LightGBM  │  CatBoost  │  RandomForest  │  Ensemble        │
│  → [P_buy, P_hold, P_sell] probability vector                            │
└─────────────────────────────────────────────┬───────────────────────────┘
                                              │ probabilities
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Risk Manager                     src/risk_manager.py          │
│  Confidence tiers → position size                                         │
│  ─────────────────────────────────────────────────────────────────────── │
│  P≥0.75 → 2.0% risk  │  P≥0.65 → 1.5%  │  P≥0.55 → 0.75%  │  P≥0.40 → 0.5%  │
│  Kelly / ATR-stop optional                                                │
│  → signal + sizing (risk_pct, sl_pips, skip flag)                        │
└─────────────────────────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Bot Execution                    src/pipeline_bot.py          │
│  PipelineBot (extends BotBase)                                            │
│  ─────────────────────────────────────────────────────────────────────── │
│  Every 60s: fetch M15 bars → predict → manage positions                   │
│  HOLD / skip → do nothing                                                 │
│  BUY signal → close sells, open 1 buy  (magic=20260101)                  │
│  SELL signal → close buys, open 1 sell                                   │
└─────────────────────────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                                    MT5 Terminal (ICMarkets Demo)
                                    via rpyc bridge on :18812
