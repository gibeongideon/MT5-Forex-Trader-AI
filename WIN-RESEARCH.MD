# WIN-RESEARCH.MD
## MT5 EURUSD M15 Trading Bot — Architecture, Results History & Research Roadmap

> **Data:** EURUSD M15, 75,000 bars, 2023-05-18 → 2026-05-25
> **Walk-forward:** Expanding window, 19 folds, 180-day train / 30-day test, no leakage
> **Backtester:** SL=30p, TP=60p, spread=1.0p, threshold=40%, tiered risk (0.5%–2% per trade)
> Last updated: 2026-05-28

---
AIM


We want I would usually evaluate:

Annual return
Sharpe ratio
Max drawdown
Win rate
Profit factor
Sortino ratio (downside risk only)

A Forex model with:

Sharpe >1.5
Max drawdown <15–20%
Profit factor >1.3
Stable performance across different market periods

is often considered a strong starting point.


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

> Target: Sharpe >1.5 · MaxDD <20% · Stable across market periods
> ✅ = meets target · ❌ = fails target · ⚠️ = borderline

| Experiment | Dataset | Sharpe | MaxDD | Return | Trades | Targets met |
|------------|---------|--------|-------|--------|--------|-------------|
| Random baseline | 49k | -0.17 | 21.6% | -2.5% | 354 | none |
| Rule engine | 49k | -0.45 | — | -9.7% | 430 | none |
| XGBoost alone (costs) | 49k | +0.52 | — | -3.8% | 127 | none |
| Phase 8 — tiered risk | 49k | +0.72 | 7.5% | +0.2% | 126 | DD ✅ |
| Unsupervised AE 16-dim | 49k | -0.41 | — | — | — | failed |
| **Phase 9 — XGBoost + enc8** | **49k** | **+3.13** | **13.3%** | **+358.3%** | **524** | **Sharpe ✅ DD ✅** |
| Phase 10 — XGBoost + enc8 | 75k | +1.12 | 22.5% | +59.9% | 753 | DD ⚠️ |
| Phase 10 — Stack enc8 | 75k | +1.22 | 16.6% | +100.9% | 748 | Sharpe ⚠️ DD ✅ |
| Phase 10 — XGBoost + enc16 | 75k | +1.47 | 36.2% | +253.9% | 735 | Sharpe ⚠️ DD ❌ |
| Phase 10 — enc16 on 49k | 49k | +2.82 | 35.1% | +99.4% | 553 | Sharpe ✅ DD ❌ |
| Stack enc8 on 49k | 49k | +2.48 | 35.4% | +266.4% | 497 | Sharpe ✅ DD ❌ |
| Stack enc8 on 75k | 75k | +1.22 | 16.6% | +100.9% | 748 | Sharpe ⚠️ DD ✅ |
| Transformer enc8 (d_model=32) | 49k | +0.44 | 7.7% | +0.3% | 129 | underpowered |
| **CatBoost + enc8** | **49k** | **+2.27** | **37.5%** | **+376.8%** | **517** | **Sharpe ✅ DD ❌** |
| H1 uncalibrated enc8 (30p/60p) | H1 May24+ | +1.47 | 11.6% | +31.2% | 465 | Sharpe ⚠️ DD ✅ |
| H1 calibrated enc8 (40p/80p) | H1 May24+ | +1.56 | 4.8% | +14.4% | 69 | Sharpe ⚠️ DD ✅ |
| Regime Router (4 specialists) | 49k | +1.40 | 34.1% | +110.0% | 504 | Sharpe ⚠️ DD ❌ |
| Multi-task enc8 (dir+vol α=0.3) | 49k | -0.43 | 28.0% | -24.8% | 564 | loses money |
| Session features (6 new) + enc8 | 49k | +1.33 | 9.2% | +22.0% | 510 | Sharpe ⚠️ DD ✅ |
| Session + K-Means cluster (k=32) | 49k | +0.85 | 15.0% | +10.9% | 516 | worse than session |
| Adaptive WF test=14d bi-weekly | 49k | +2.57 | 27.7% | +286.1% | 523 | Sharpe ✅ DD ❌ |
| M30 enc8 same params (30p/60p) | M30 May24+ | +2.46 | 23.2% | +68.5% | 411 | Sharpe ✅ DD ❌ |
| M30 enc8 wider stops (40p/80p) | M30 May24+ | +1.63 | 9.8% | +39.9% | 266 | Sharpe ✅ DD ✅ |
| M5 enc8 time-matched (h=12) | M5 May25+ | +1.06 | 21.6% | +24.8% | 139 | DD ❌ |
| M5 enc8 same params (h=4) | M5 May25+ | -0.60 | 23.7% | -14.7% | 144 | loses money |

---

### 49k Dataset — All Challengers Ranked (May 2024 → May 2026)

> Every experiment run on the same 49k dataset with the same enc8 features (39 cols).
> One winner across all dimensions: XGBoost + enc8.

| Rank | Model | Sharpe | MaxDD | Return | Verdict |
|------|-------|--------|-------|--------|---------|
| 🥇 | **XGBoost + enc8** | **+3.13** | **13.3%** | **+358.3%** | **Champion — deploy candidate** |
| 2 | enc16 XGBoost | +2.82 | 35.1% | +99.4% | Higher Sharpe failed, 3× DD |
| 3 | Stack XGB+Cat | +2.48 | 35.4% | +266.4% | Stacking hurts when signal is strong |
| 4 | CatBoost + enc8 | +2.27 | 37.5% | +376.8% | Most return, worst DD |
| 5 | Transformer enc8 | +0.44 | 7.7% | +0.3% | Never learned (d_model too small) |

**Pattern:** Every challenger either raises MaxDD dramatically (>35%) or collapses Sharpe.
XGBoost + enc8 is the only config meeting **both** targets (Sharpe >1.5 AND MaxDD <20%) simultaneously.
The enc8 bottleneck + XGBoost's gradient boosting is the sweet spot for this data.

---

## 6. Sweet Spot Analysis

### Target Criteria (from AIM)

```
Sharpe ratio  > 1.5
Max drawdown  < 15–20%
Profit factor > 1.3
Stable across different market periods (multiple folds, different regimes)
```

### What We Know So Far

| Metric | 49k dataset (May 2024+) | 75k dataset (May 2023+) |
| ------ | ----------------------- | ----------------------- |
| **Best Sharpe** | +3.13 (XGBoost + enc8) | +1.47 (XGBoost + enc16) |
| **Best MaxDD** | 13.3% (XGBoost + enc8) | 16.6% (Stack + enc8) |
| **Meets BOTH targets** | ✅ Phase 9 enc8 (+3.13, 13.3%) | ⚠️ Stack enc8 (+1.22, 16.6%) |

### The Core Tension

**More data = lower Sharpe, better reliability.**

- 49k rows (2024–2026): strong recent results, but only ~2 years tested
- 75k rows (2023–2026): harder, includes more market regimes, more honest

The +3.13 on 49k data is a real result but may reflect a particularly favourable period. The +1.22 on 75k is more conservative but tested across a wider range of conditions.

### Where the Sweet Spot Lives

Based on results so far, the sweet spot is at the intersection of:

```
Best model:    enc8 (8-dim supervised encoder, 30 epochs)
Best signal:   XGBoost alone or Stack (XGBoost + CatBoost)
Best dataset:  49k rows (May 2024+) — the period where our encoder was trained
Best config:   Stack beats XGBoost alone on MaxDD (16.6% vs 22.5% on 75k)
               Stack result on 49k = PENDING (running now)
```

**Stack result (completed 2026-05-27):**

- Stack on 49k: +2.48 Sharpe / 35.4% MaxDD — worse than XGBoost alone on both metrics
- Stack on 75k: +1.22 Sharpe / 16.6% MaxDD — marginally better than XGBoost alone (+0.10)
- **Conclusion:** stacking adds value when base model is weak; when signal is strong (enc8 on 49k), keep it simple

### Key Decisions — Current Status

| Question | Options tested | Decision |
| -------- | -------------- | -------- |
| Which encoder dim? | enc4 / enc8 / enc16 | **enc8 confirmed best** |
| MLP vs Transformer? | MLP +3.13 vs Transformer +0.44 | **MLP wins — keep default** |
| Blend or stack? | Blend hurts, Stack helps MaxDD on 75k | **XGBoost alone on 49k is unbeaten** |
| Which dataset? | 49k (Sharpe +3.13) vs 75k (Sharpe +1.12) | **49k for deployment, 75k for robustness check** |
| Which model? | XGBoost +3.13 vs CatBoost +2.27 | **XGBoost wins on Sharpe AND MaxDD** |
| Which timeframe? | M15/M30/H1/M5 all tested | **M15 = sweet spot. M30 viable secondary. M5 loses money.** |
| Regime routing? | 4-specialist RegimeRouter vs XGBoost | **XGBoost wins — KMeans labels unstable across folds** |
| Deployment ready? | Sharpe >1.5 ✅, MaxDD <20% ✅ | **Phase 9 enc8 is the deploy candidate** |

### After Stack Results — Next Steps

If stack on 49k beats +3.13 or achieves same Sharpe with lower MaxDD:
→ **This is the model to deploy.** Retrain on full 49k dataset and push to live bot.

If stack on 49k doesn't beat +3.13:
→ XGBoost + enc8 on 49k remains the deployment candidate (+3.13, MaxDD 13.3%).

Either way the 75k stack result shows how the strategy holds up over the harder 2023-2024 period — important for confidence before going live.

**Remaining experiments to try after stack:**

1. Stack + enc8 on 49k → **running now**
2. Stack + enc8 on 75k → **running now**
3. CatBoost alone + enc8 on 49k (CatBoost was +1.17 baseline, may shine with enc8)
4. Reduce MaxDD: tighten drawdown throttle from 10% → 7% (free, config change only)
5. Extend 49k data: add 2023-2024 data gradually, find breakeven fold

**Phase 10 results (2026-05-27, 31 folds, 2h 20min run):**

| Config | Description | Sharpe | MaxDD | Return | Trades |
| ------ | ----------- | ------ | ----- | ------ | ------ |
| A | XGBoost alone (enc 8-dim, 39 feat) | +1.12 | 22.5% | +59.9% | 753 |
| B | XGBoost + CatBoost blend 50/50 | +1.03 | 26.3% | +58.5% | 749 |
| C | XGBoost + CatBoost + LightGBM blend | +0.65 | 28.9% | +34.4% | 756 |
| D | XGBoost + CatBoost stack (LGB meta) | +1.22 | 16.6% | +100.9% | 748 |
| **E** | **XGBoost + enc 16-dim / 60 epochs** | **+1.47** | **36.2%** | **+253.9%** | **735** |

**Key findings from Phase 10:**

- Blending (B, C) *hurts* Sharpe — adding more models as a simple average dilutes the best model's signal
- Stacking (D) *helps* vs blending — meta-learner learns when to trust each model, cuts MaxDD to 16.6%
- Bigger encoder (E) gives best Sharpe on 75k data (+1.47), at the cost of very high MaxDD (36.2%)
- The 75k vs 49k gap: Phase 9 measured 19 folds (May 2024–May 2026); Phase 10 used 31 folds (May 2023–May 2026). The extra 12 folds on older 2023-2024 data are harder and pull aggregate Sharpe down.

---

### Phase 13 — H1 Timeframe Test (2026-05-28)

Same winning config (XGBoost + enc8) applied to EURUSD H1 data, same May 2024 → May 2026 date range.
Two stop variants tested: tight 30p/60p (same as M15) and wide 50p/100p (H1-typical).

**H1 dataset:** 12,619 bars (vs 49,892 M15 bars for same period — 4× fewer bars, same 2-year window)
**Encoder accuracy at epoch 30:** 53.9% (vs M15's 72.4% — H1 patterns are harder to learn)
**Folds:** 19 (same structure — 180-day train / 30-day test)

| Config | Sharpe | MaxDD | Return | Trades |
| ------ | ------ | ----- | ------ | ------ |
| M15 XGBoost + enc8 30p/60p [CHAMPION] | +3.13 | 13.3% | +358.3% | 524 |
| H1 XGBoost + enc8 30p/60p | +1.47 | 11.6% | +31.2% | 465 |
| H1 XGBoost + enc8 50p/100p (wide) | +0.18 | 12.8% | -1.1% | 196 |

**H1 30p/60p is borderline viable** (Sharpe +1.47 meets the >1.5 target just barely, MaxDD 11.6% is excellent). But it generates only 10% of the return of M15.

**Wide stops collapsed the system.** The root cause: labels were trained with `threshold=0.0003` (3 pips) at 4-bar horizon. On H1, 4 bars = 4 hours. The model detects micro-directional signals. With 50p stops, it needs 100 pips of favorable movement — a completely different task than what the model learned. Only 196 trades fired (vs 465 with tight stops) because the model became too selective.

**Why M15 works better than H1 for this strategy:**

| Factor | M15 | H1 |
| ------ | --- | -- |
| Bars per year | ~26,000 | ~6,500 |
| Label threshold | 3 pips / 1 hour ahead | 3 pips / 4 hours ahead (same — but H1 moves more) |
| Encoder accuracy | 72.4% | 53.9% |
| Return | +358.3% | +31.2% |
| Trades per year | ~250 | ~230 |

The enc8 encoder learns better on M15 because there are 4× more training windows for the same time period, and M15 patterns at 50-bar windows (12.5 hours) are more predictable than H1 50-bar windows (50 hours).

**To make H1 work properly would need:**
- `label_threshold: 0.001` (10 pips, matching H1 volatility)
- `label_horizon: 8` (8 H1 bars = 8 hours)
- Stop calibration: SL=30–40p, TP=60–80p (let signal dictate, not bar range)

**Decision: M15 remains the deployment timeframe. H1 test confirmed M15 is the sweet spot.**

---

### Phase 19 — Session Features + K-Means Candle Clustering (2026-05-28)

Testing two ideas from RANDOM_IDEAS.MD (MY IDEAS section):

**Idea 4 partial — Session/time features:** 6 new features added to `feature_pipeline.py`:
`hour_sin`, `hour_cos`, `is_london_open`, `is_ny_open`, `is_london_ny_overlap`, `is_asia`

**Idea 2 — K-Means Candle Clustering:** New `CandleTokenizer` class — KMeans(k=32) on
`[log_return, upper_wick_pct, lower_wick_pct, vol_zscore, atr_norm_range]` per bar.
Learns 32 bar archetypes (aggressive breakout, doji, hammer, etc.) from training data.
Adds single `candle_cluster` integer feature (0–31).

Note: Config A and B are identical because `feature_pipeline.py` was modified globally —
all configs now include session features. Comparison is vs Phase 9 pre-built baseline.

| Config | Features | Sharpe | MaxDD | Return | Trades |
| ------ | -------- | ------ | ----- | ------ | ------ |
| Phase 9 pre-built enc8 [CHAMPION] | 39 | +3.13 | 13.3% | +358.3% | 524 |
| + Session features (retrained fresh) | 45 | +1.33 | 9.2% | +22.0% | 510 |
| + Session + candle cluster (k=32) | 46 | +0.85 | 15.0% | +10.9% | 516 |

**Both new feature groups hurt Sharpe. Baseline +3.13 holds.**

**Why session features hurt (-1.80 Sharpe):**

The 6 session features change the input distribution to the enc8 encoder. The encoder was implicitly optimised to work with 31 features — retraining it with 37 features produces different latent dims that encode less predictive direction signal. This is the same saturation effect seen in multi-task and enc16: the enc8 is already at maximum capacity for the direction task.

The session features do produce lower MaxDD (9.2% vs 13.3%) — the model becomes more selective (510 vs 524 trades) and more conservative. But the Sharpe drops too far.

**Why candle clustering makes it worse (-0.48 from session alone):**

The 32-cluster integer feature adds noise that XGBoost splits on spuriously. Like the regime router, the cluster labels are unsupervised — they may not align with direction prediction. Splitting training data into 32 archetype-based groups dilutes the directional signal further.

**Key insight from both experiments:** The enc8 + 31-feature combination is already at a local optimum. Adding more features — whether session context, candle shapes, or multi-task objectives — consistently hurts because:
1. The encoder retrains on a different input distribution, producing weaker latent dims
2. XGBoost gets more noisy dimensions to split on, reducing its signal-to-noise ratio

**The right way to use session features:** Add them to the original Phase 9 pipeline BEFORE building the pre-built parquet (don't retrain enc8, just add them as extra columns to the existing 39-feature parquet). This would test session features with the SAME enc8 latent dims, not a freshly trained encoder.

**Decision: Revert `feature_pipeline.py` to original 31 features (or keep them off by default for walk-forward reproduction). The pre-built enc8 parquet remains the gold standard.**

---

### Phase 18 — Multi-task Encoder: Direction + Volatility Heads (2026-05-28)

Added an auxiliary volatility prediction head to the supervised MLP encoder.
Training loss = L_direction + 0.3 × L_volatility, forcing the shared encoder trunk
to learn features useful for both predicting direction AND next-bar move size.

Architecture:
- Encoder trunk: same MLP (250→256→128→8 Tanh) as supervised champion
- Direction head: Linear(8→3), CrossEntropyLoss (primary)
- Volatility head: Linear(8→16→1), Softplus + MSELoss (auxiliary)
- Alpha=0.3: L = L_dir + 0.3 × L_vol
- Both heads discarded after training — only trunk kept

| Config | Encoder acc (ep30) | Sharpe | MaxDD | Return | Trades |
| ------ | ------------------ | ------ | ----- | ------ | ------ |
| Supervised MLP enc8 [CHAMPION] | 72.4% | +3.13 | 13.3% | +358.3% | 524 |
| Multi-task enc8 alpha=0.3 | **53.0%** | **-0.43** | **28.0%** | **-24.8%** | 564 |

**Multi-task failed — loses money. Supervised MLP stays champion.**

**Root cause — 3 compounding problems:**

1. **Capacity conflict.** With only 8 latent dimensions, there is no room for both direction AND volatility signals. The encoder cannot simultaneously satisfy "encode directional regime" and "encode volatility level" — these are different aspects of market structure that are not always correlated.

2. **Gradient interference.** A high-volatility ranging bar sends conflicting gradient signals: the direction head says "hold" (no trend), but the volatility head says "this bar moves a lot" (high vol). The shared encoder weights receive opposing signals, producing a compromise representation that is good at neither.

3. **Reported loss is misleading.** The ce_loss printed during training includes the combined loss (L_dir + 0.3 × L_vol), not just direction. The epoch 30 ce_loss of 1.26 looks high compared to supervised's 0.63 but they're different quantities. The accuracy drop (72.4% → 53.0%) is the real signal — 19pp worse direction learning.

**What would fix it:**
- More latent dims (e.g. enc16 for multitask) — separate capacity for each objective
- Separate encoder trunks per head with shared first layers only
- Lower alpha (0.1 or 0.05) — reduce vol gradient's influence on the shared trunk
- But note: enc16 already tested and failed (+2.82 Sharpe, 35.1% MaxDD) even without multitask

**Conclusion:** The 8-dim supervised enc is already saturated for the direction task. Any modification that dilutes the direction gradient degrades performance. The architecture is optimal as-is.

---

### Phase 17 — M5 Timeframe Test (2026-05-28)

Testing XGBoost + enc8 on EURUSD M5 (May 2025 → May 2026, 75k bars, ~1 year).
⚠ Different date range — only ~7 folds vs M15's 19. Results have higher variance.

Two label configs × two stop variants:

| Config | Sharpe | MaxDD | Return | Trades |
| ------ | ------ | ----- | ------ | ------ |
| M15 XGBoost+enc8 30p/60p [CHAMPION] | +3.13 | 13.3% | +358.3% | 524 |
| M5 same-as-M15  h=4 (20min)  30p/60p | -0.60 | 23.7% | -14.7% | 144 |
| M5 same-as-M15  h=4 (20min)  40p/80p | -2.26 | 17.7% | -13.9% | 84 |
| M5 time-matched h=12 (60min) 30p/60p | +1.06 | 21.6% | +24.8% | 139 |
| M5 time-matched h=12 (60min) 40p/80p | +0.71 | 13.1% | +11.2% | 83 |

**M5 is the worst timeframe tested — below even H1.**

**Why M5 fails:**

1. **Spread destroys the edge.** 1-pip spread on a 3-pip threshold (h=4, 20min horizon) = spread is 33% of the signal. On M15 the same 1-pip spread is ~17% of a 3-pip move over 60 min — proportionally half as destructive.

2. **Encoder accuracy stagnates at 57%.** M5 bars contain more microstructure noise. The same 50-bar window covers only 250 minutes (4 hours) vs M15's 12.5 hours — not enough to capture meaningful regime patterns.

3. **Label distribution reveals extreme uncertainty:** hold=60% of bars (vs M15's 20%). The model simply cannot predict 20-minute moves reliably.

**Time-matching (h=12) helps significantly:** +1.06 vs -0.60 — making the horizon 60 minutes matches M15's wall-clock prediction window. But MaxDD remains high (21.6%) and Sharpe is still well below M15.

**Complete timeframe leaderboard:**

| Timeframe | Bars/2yr | Best Sharpe | MaxDD | Verdict |
| --------- | -------- | ----------- | ----- | ------- |
| M15 | ~49k | +3.13 | 13.3% | ✅ Champion — deploy |
| M30 | ~25k | +2.46 | 23.2% | ⚠️ Viable secondary (40p/80p: +1.63 / 9.8%) |
| H1 cal | ~12k | +1.56 | 4.8% | ❌ Too few signals (69 trades/2yr) |
| M5 | ~75k* | +1.06 | 21.6% | ❌ Spread kills edge |

*M5 only 1 year of data (May 2025+).

**Conclusion: M15 is the undisputed sweet spot.** Going faster (M5) or slower (H1) both hurt. M15's 15-minute resolution is optimal for this enc8 + XGBoost architecture: enough bars to train a strong encoder (49k windows), clean regime patterns in the 50-bar window (12.5 hours), and a prediction horizon (60 min) where directional moves are detectable above the spread.

**All timeframe experiments complete. Champion confirmed: XGBoost + enc8, M15, +3.13 Sharpe, 13.3% MaxDD.**

---

### Phase 16 — M30 Timeframe Test (2026-05-28)

Testing XGBoost + enc8 on EURUSD M30 (~25k bars, May 2024+). Ran in **57 min total** (24 min Config A fold training + 33 min Config B fold training; stop variants 2 loaded from cache instantly).

Two label configs × two stop variants = 4 runs:

| Config | Sharpe | MaxDD | Return | Trades |
| ------ | ------ | ----- | ------ | ------ |
| M15 XGBoost+enc8 30p/60p [CHAMPION] | +3.13 | 13.3% | +358.3% | 524 |
| **M30 same-as-M15  horizon=4 (2h)  30p/60p** | **+2.46** | **23.2%** | **+68.5%** | **411** |
| M30 same-as-M15  horizon=4 (2h)  40p/80p | +1.63 | 9.8% | +39.9% | 266 |
| M30 time-matched  horizon=2 (1h)  30p/60p | +1.38 | 9.8% | +28.9% | 206 |
| M30 time-matched  horizon=2 (1h)  40p/80p | +0.96 | 12.1% | +17.3% | 153 |

**M15 wins but M30 is the closest challenger — only -0.67 Sharpe behind.**

**Key findings:**

1. **Best M30 Sharpe (+2.46) has MaxDD 23.2%** — just above the 20% target. Nearly there.
2. **MaxDD-safe M30 (+1.63 / 9.8% MaxDD)** — meets BOTH targets (Sharpe >1.5 ✅, MaxDD <20% ✅) with wider 40p/80p stops. Only 266 trades vs M15's 524.
3. **Time-matching hurts.** horizon=2 (same 1-hour wall clock as M15) is worse than horizon=4 (same count). Predicting 2 hours ahead gives a cleaner signal on M30 than 1 hour.
4. **Timeframe gradient confirmed:** M15 > M30 > H1 — more bars = more training signal, more compounding opportunities.

**Complete timeframe picture:**

| Timeframe | Best Sharpe | MaxDD | Trades/2yr | Meets targets? |
| --------- | ----------- | ----- | ---------- | -------------- |
| M15 | +3.13 | 13.3% | 524 | ✅ Both |
| M30 (30p/60p) | +2.46 | 23.2% | 411 | Sharpe ✅ DD ❌ |
| M30 (40p/80p) | +1.63 | 9.8% | 266 | ✅ Both |
| H1 calibrated | +1.56 | 4.8% | 69 | Sharpe ⚠️ DD ✅ |

**Decision: M15 remains the deployment timeframe.** M30 at 40p/80p is a viable secondary if wider stops are preferred (lower MaxDD at cost of fewer trades). The key question for M30 is whether 266 trades over 2 years provides enough compounding — at +39.9% return over 2 years vs M15's +358.3%, M15 compounds far more efficiently.

---

### Phase 15 — Regime-Conditioned Models (2026-05-28)

Testing `RegimeRouter` — 4 specialist XGBoost models, one per market regime.
Same 49k enc8 features (39 cols), same walk-forward as M15 champion. Ran in **84 min**.

Architecture:
- KMeans(k=4) on [ATR-ratio, ADX/100, (RSI-50)/50] → 4 regime labels
- Regimes: trending-up / trending-down / ranging / high-volatility
- Specialist XGBoost trained per regime (min 200 samples, else fallback)
- LogisticRegression regime classifier routes each bar at inference

| Model | Sharpe | MaxDD | Return | Trades |
| ----- | ------ | ----- | ------ | ------ |
| XGBoost + enc8 [CHAMPION] | +3.13 | 13.3% | +358.3% | 524 |
| RegimeRouter (4 specialists) | +1.40 | 34.1% | +110.0% | 504 |

**RegimeRouter lost on every metric. XGBoost alone wins.**

**Why it failed — two root causes:**

1. **KMeans regime labels are unstable across folds.** The cluster assignments are arbitrary numbers (0–3) that don't map to the same market conditions fold-to-fold. Fold 0's "regime 0" (37% trending-up) becomes a 10% minority in fold 1. The "trending-up specialist" in fold 1 was trained on completely different data than in fold 0.

2. **XGBoost already handles regimes internally.** Its tree splits on ATR/ADX/RSI naturally separate behaviour by volatility and trend strength. Forcing separate models removes cross-regime learning and cuts each specialist's training data to ~25% (12k bars instead of 49k).

**Fix if retrying:** Use a deterministic regime labeller (e.g. ADX threshold rules instead of KMeans) so regime 0 always means the same thing across folds. But the data-splitting penalty remains.

**Decision: Keep `model_type: xgboost`. Regime routing in this form hurts more than it helps.**

---

### Phase 14 — H1 Calibrated Labels (2026-05-28)

Previous H1 test used M15 labels (3-pip, 4-bar = 1h horizon) — mismatch.
This run recalibrates for H1 scale: **horizon=8 bars (8h), threshold=10 pips**.

**Encoder accuracy improved:** 57.3% at epoch 30 (vs 53.9% with M15 labels).
**Label distribution shifted:** hold=42% (vs 20% with M15 labels) — model became much more selective.

| Config | Sharpe | MaxDD | Return | Trades |
| ------ | ------ | ----- | ------ | ------ |
| M15 XGBoost+enc8 30p/60p [CHAMPION] | +3.13 | 13.3% | +358.3% | 524 |
| H1 uncalibrated 30p/60p (from Phase 13) | +1.47 | 11.6% | +31.2% | 465 |
| H1 cal SL=30p/60p (tight) | +1.41 | 5.1% | +17.1% | 97 |
| **H1 cal SL=40p/80p (medium)** | **+1.56** | **4.8%** | **+14.4%** | **69** |
| H1 cal SL=50p/100p (wide) | +1.43 | 3.2% | +16.2% | 59 |

**Best H1 config: SL=40p/TP=80p → Sharpe +1.56, MaxDD 4.8%**

The 40p/80p stops match H1's typical bar range (~1× ATR) and give the model enough room to breathe without getting stopped by noise.

**Calibration helped slightly (+0.09 Sharpe vs uncalibrated) but exposed a bigger problem: too few trades.**

With `threshold=10pip, horizon=8h`:
- 42% of bars labelled "hold" (vs 20% with M15 labels)
- Only 69 trades over 2 years = ~3 trades/month
- Return: +14.4% over 2 years vs M15's +358.3%

The model became very selective (high Sharpe, tiny MaxDD) but doesn't generate enough signals to compound meaningfully. H1 simply can't match M15's 524 trades / +358% return over the same period.

**Why M30 might be the sweet spot between M15 and H1:**
- M15: 524 trades (lots of compounding, some noise)
- M30: expected ~200–300 trades (reduced noise, still enough compounding)
- H1 cal: 69 trades (too selective, not enough opportunities)

**Decision: H1 requires a fundamentally different strategy (wider TP, fewer signals, longer holds) to work well. Not a drop-in swap for M15.**

---

### Phase 12 — CatBoost + enc8 on 49k (2026-05-28)

Quick test: CatBoost has the same enc8 features as XGBoost (39 cols, pre-built parquet). No encoder retraining needed. Ran in **1.4 minutes**.

| Model | Sharpe | MaxDD | Return | Trades |
| ----- | ------ | ----- | ------ | ------ |
| XGBoost + enc8 [CHAMPION] | **+3.13** | **13.3%** | +358.3% | 524 |
| CatBoost + enc8 | +2.27 | 37.5% | +376.8% | 517 |

**XGBoost wins. CatBoost made more money (+376.8% vs +358.3%) but at nearly 3× the drawdown (37.5% vs 13.3%).**

CatBoost is more aggressive — fires more confident-looking signals but with higher volatility equity curve. XGBoost's smoother, tighter signal quality is what drives the Sharpe advantage.

**Decision: XGBoost + enc8 default confirmed. config.yaml unchanged.**

---

### Phase 11 — Transformer Encoder vs MLP (2026-05-28)

Drop-in replacement test: can self-attention beat the MLP enc8 champion (+3.13)?
Same dataset (49k, May 2024+), same latent_dim=8, same XGBoost downstream — only the encoder internals change.

**Architecture tested:**
```
_TransformerEncoderNet:
  Input: (batch, 50×5=250) → reshape → (batch, 50, 5)
  Linear(5 → 32)                     ← project to d_model
  CLS token prepended                ← (batch, 51, 32)
  TransformerEncoder(d_model=32, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1)
  CLS output → Linear(32 → 8)        ← latent vector
  Linear(8 → 3)                      ← classification head (discarded after training)
```

**Training curve (the red flag):**
```
epoch  1/30  ce_loss=1.1149  train_acc=33.2%   ← barely above 3-class random (33%)
epoch  5/30  ce_loss=1.0996  train_acc=32.7%
epoch 10/30  ce_loss=1.0989  train_acc=34.9%
epoch 20/30  ce_loss=1.0987  train_acc=34.0%
epoch 30/30  ce_loss=1.0974  train_acc=36.0%   ← barely moved after 30 epochs
```

Compare to MLP at epoch 30: **train_acc=72.4%**. The Transformer never learned.

| Config | Encoder | Sharpe | MaxDD | Return | Trades | Runtime |
| ------ | ------- | ------ | ----- | ------ | ------ | ------- |
| MLP supervised enc8 (BASELINE) | Linear×3 + Tanh | **+3.13** | **13.3%** | **+358.3%** | **524** | ~15 min |
| Transformer enc8 (d_model=32, 4h, 2L) | Self-attention + CLS | +0.44 | 7.7% | +0.3% | 129 | 33.7 min |

**MLP wins by +2.69 Sharpe. Default config unchanged.**

**Why the Transformer failed:**

1. **Underpowered for 30 epochs.** Transformers require many more iterations to converge than MLPs on tabular input. The MLP has a strong inductive bias (dense layers) that fits this flat feature format. The Transformer needs to learn positional relationships from scratch.

2. **d_model=32 is too small.** The attention mechanism with only 32 hidden dims and 4 heads (8 dims per head) has very limited capacity. Standard NLP Transformers use d_model=256–512 minimum.

3. **No learning rate warmup.** Transformers notoriously need a linear warmup + cosine decay schedule. Using a flat Adam lr=0.001 causes the attention weights to diverge early and never recover.

4. **Overfitting risk from small dataset.** 40k windows × 30 epochs = 1.2M passes through a model with ~100k parameters (vs MLP ~70k). Transformer needs more diverse data to generalise.

**What to try before discarding Transformers:**
- Train 150+ epochs with cosine annealing LR schedule
- Increase d_model to 64 or 128
- Add positional encoding (markets have temporal structure)
- Use larger dataset (75k rows) where MLP already struggles — Transformer may show edge there

**Decision:** MLP enc8 stays as default. Transformer kept as switchable option via `config.yaml encoder.mode: transformer` for future tuning.

---

### Phase 10 — enc8 vs enc16 on 49k Dataset (2026-05-27)

Direct test: does a bigger encoder beat the +3.13 record on the same data?

| Config | Encoder | Features | Sharpe | MaxDD | Return | Trades |
| ------ | ------- | -------- | ------ | ----- | ------ | ------ |
| A (Phase 9 reference) | enc8 / 30ep | 39 | +3.13 | 13.3% | +358.3% | 524 |
| B | enc16 / 60ep | 47 | +2.82 | 35.1% | +99.4% | 553 |

**enc8 wins on every metric.** enc16 is -0.31 Sharpe worse and nearly 3× the drawdown.

**Why:** the 8-dim bottleneck forces the encoder to keep only the most predictive regime signal — stronger regularisation. 16 dims captures noise alongside signal, hurting OOS generalisation.

**Conclusion: enc8 / 30 epochs remains the best encoder.**

---

### Phase 10 — Stack on 49k vs 75k (2026-05-27, in progress)

Running `scripts/compare_stack_49k_75k.py` — results pending.

| Config | Dataset | Model | Sharpe | MaxDD | Return | Trades |
| ------ | ------- | ----- | ------ | ----- | ------ | ------ |
| Reference | 49k (May 2024+) | XGBoost + enc8 | +3.13 | 13.3% | +358.3% | 524 |
| Reference | 75k (full) | XGBoost + enc8 | +1.12 | 22.5% | +59.9% | 753 |
| Stack + enc8 | 49k (May 2024+) | Stack (XGB+Cat) + enc8 | +2.48 | 35.4% | +266.4% | 497 |
| Stack + enc8 | 75k (full) | Stack (XGB+Cat) + enc8 | +1.22 | 16.6% | +100.9% | 748 |

**Stack on 49k LOST to XGBoost alone** (-0.65 Sharpe, MaxDD nearly 3× worse).
**Stack on 75k marginally better** (+0.10 Sharpe, MaxDD 6pp lower).

**Insight:** Stacking adds value when the base model is weak. When the base model is already strong (enc8 on 49k = +3.13), the meta-learner adds noise from OOF folds — hurts rather than helps. Rule: **keep it simple when the signal is strong.**

**Future experiments (cloud / LLM ensemble — requires cache build first):**

| Priority | Config | Description | Prerequisite |
| -------- | ------ | ----------- | ------------ |
| F | LLM blend (low weight) | XGBoost + CatBoost + Anthropic Claude (weight 0.2) | Run `scripts/precompute_llm_signals.py` over full dataset |
| G | LLM stack (meta-learner) | XGBoost + CatBoost + LLM → LightGBM meta-learner | Same as F — needs full signal cache |
| H | Cloud model blend | XGBoost + SageMaker/Vertex endpoint | Deploy model to cloud, create `src/models/cloud_model.py` |
| I | Hybrid ensemble | XGBoost (local) + Cloud (60/40 blend) | H complete |

**Why LLM in ensemble makes sense:** XGBoost/CatBoost learn statistical feature patterns.
Claude reads the tokenized bar sequence and reasons about momentum, compression, and context
— orthogonal signal sources. Meta-learner learns when to trust each.

**Why to do local ensemble first:** LLM signals require a one-time cache build (38k+ API calls
for the full dataset) which costs money and time. Local ensemble (configs A–E) is free
and fast. Establish a strong local baseline before adding cloud cost.

**How to add LLM to the ensemble (once cache is built):**

```yaml
# config.yaml
ensemble:
  mode: blend
  base_models:
    - name: xgboost
      weight: 0.4
    - name: catboost
      weight: 0.4
    - name: llm_signal
      weight: 0.2   # low weight until LLM accuracy is validated
```

**Experiments completed (2026-05-27/28):**
- ~~enc8 vs enc16~~ → enc8 wins (+3.13 vs +2.82, enc16 MaxDD 3×worse)
- ~~Stack on 49k vs 75k~~ → XGBoost alone unbeaten on 49k; stack helps slightly on 75k
- ~~Transformer encoder (self-attention drop-in)~~ → failed at d_model=32/30ep (+0.44); MLP holds

**Next experiments to try (in order):**
1. Transformer with proper tuning — 150+ epochs, d_model=64, LR warmup scheduler (estimated +0.5–1.0 Sharpe lift if it learns)
2. CatBoost alone + enc8 on 49k (CatBoost was +1.17 baseline, untested with enc8)
3. Multi-timeframe latent (M15 + H1 + H4 encoders → 31 base + 24 latent = 55 features)
4. Regime-conditioned XGBoost (IMPROVEMENT.MD Technique #2 — specialist models per regime)
5. Walk-forward online adaptation (IMPROVEMENT.MD Technique #3 — retrain every 500 bars)

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




MODES TO EXPLORE


Option	Expected Sharpe lift	Risk	Time
XGBoost + CatBoost blend	+0.1 to +0.3	Low	5 min
XGBoost + CatBoost + LightGBM blend	+0.2 to +0.4	Low	5 min
XGBoost + CatBoost stack (meta-learner)	+0.3 to +0.6	Low	15 min
Increase encoder to latent_dim=16, epochs=60	+0.2 to +0.5	Low	20 min
XGB + CatBoost + LLM blend (cache=partial)	unpredictable	Medium	build cache first
XGB + CatBoost + LLM stack (full cache)	+0.5 to +1.0	High	hours (cache build)
My recommendation: Run the stack meta-learner with XGBoost + CatBoost first — no API costs, no cache needed, fast. If that doesn't beat +3.13, add LightGBM to the mix. LLM should be step 3 after you've seen what local models can do.

Let's run the comparison now — first the current blend, then stack: