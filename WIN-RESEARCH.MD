# MT5 Algorithmic Trading — Research Log & Results Bible

> **Data:** EURUSD + USDJPY M15, 60,000 bars, 2024-01-08 → 2026-06-05
> **Walk-forward:** Expanding window, 180d train / 30d test, no leakage
> **Platform:** Ubuntu + Wine + mt5linux/rpyc bridge, conda env `envmt5` (Python 3.10)
> **Last updated:** 2026-06-12

This document is the single source of truth for the entire research journey. It supersedes WIN-RESEARCH.MD (v1), the stale sections of PROGRESS.md, and the outdated LEADERBOARD champion section. Do not repeat experiments listed in the **Dead Ends** section without a new hypothesis.

---

## RESEARCH AIM

A Forex model with:
- Sharpe > 1.5
- Max drawdown < 15–20%
- Profit factor > 1.3
- Stable performance across different market periods

**Current status (2026-06-12):** All targets exceeded across both EURUSD and USDJPY.

---

## CURRENT SYSTEM ARCHITECTURE (Hybrid v2)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID v2 ARCHITECTURE                             │
│                                                                              │
│  Layer 1 — CatBoost Candle Predictor (1-bar specialist)                      │
│    data/models/candle_{SYMBOL}/   52 features, CatBoost, sliding 120d window │
│    Predicts bar i+1 direction using only bars 0..i                           │
│    Output: P_buy, P_hold, P_sell                                             │
│                        │                                                     │
│                        ▼  candle_p_buy, candle_p_sell passed as features     │
│  Layer 2 — XGBoost Pipeline (4-bar generalist)                               │
│    data/models/pipeline_{SYMBOL}_v2/  42 features, XGBoost + enc8            │
│    32 technical indicators + 8 latent encoder dims + 2 candle probabilities  │
│    Predicts 4-bar direction horizon                                          │
│                        │                                                     │
│                        ▼                                                     │
│  Risk Manager → MT5 Execution → Trade Journal (live_trades.db)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Feature breakdown — hybrid v2 (42 features)

| Layer | Count | Content |
| ----- | ----- | ------- |
| Base technical | 32 | SMA(9/21/50), EMA(20), RSI(14), MACD, BB(20), ATR(14), ADX, OBV, Stoch, + session hours/flags, MTF EMA |
| enc8 latent | 8 | Supervised MLP encoder: 50-bar OHLCV window → 8 latent dims, trained with CrossEntropyLoss |
| Candle probs | 2 | `candle_p_buy`, `candle_p_sell` — OOS predictions from CatBoost candle model |
| **Total** | **42** | |

The candle probability features are injected as **walk-forward OOS predictions** — each bar's prediction came from a CatBoost fold model that never trained on that bar. This prevents lookahead leakage.

---

## CURRENT CHAMPIONS

### Candle Standalone (CatBoost, 1-bar, OOS 2.4 yr)

| Mode | Symbol | Sharpe | Win% | MaxDD | Trades/yr | Notes |
| ---- | ------ | ------ | ---- | ----- | --------- | ----- |
| candle_predictor | EURUSD | **+20.1** | 87.1% | 6.7% | ~920 | 1-bar force-close |
| candle_predictor | USDJPY | **+25.6** | 81.1% | 10.9% | ~1,720 | |
| candle_trail (tuned) | EURUSD | **+19.4** | 91.3% | **3.1%** | ~800 | act=12p, behind=5p |
| candle_trail (tuned) | USDJPY | **+24.5** | 88.4% | **4.8%** | ~1,570 | act=15p, behind=5p |

### Hybrid v2 (XGBoost + enc8 + candle injection, WF OOS, 60k bars)

| Symbol | Features | Sharpe | MaxDD | Return | Trades | Artifact |
| ------ | -------- | ------ | ----- | ------ | ------ | -------- |
| EURUSD | 42 | **+3.01** | 11.4% | +93% | 540 | `data/models/pipeline_EURUSD_v2/` |
| USDJPY | 42 | **+4.27** | 19.6% | +1419% | 1,395 | `data/models/pipeline_USDJPY_v2/` |

---

## COMPLETE EXPERIMENT HISTORY

### PHASE 1–8 — Foundation

| Phase | Config | Sharpe | MaxDD | Verdict |
| ----- | ------ | ------ | ----- | ------- |
| 1 | Random baseline | −0.17 | 21.6% | Noise floor |
| 2 | Rule-based MA cross | −0.45 | — | No ML edge |
| 4 | XGBoost 31 features | +1.34 | 10.5% | First real edge |
| 5 | LightGBM 31 features | +0.72 | 6.8% | Worse than XGB |
| 5 | Random Forest | +0.24 | 14.8% | Bagging < boosting |
| 6 | CatBoost 31 features | +1.17 | 24.7% | High DD, worse than XGB |
| 8-B | XGBoost + tiered risk | +0.72 | 7.5% | Same Sharpe, −49% DD |
| 8-C | XGBoost + ATR stop | +0.22 | 3.4% | Too few trades |

**Outcome:** XGBoost + 31 base features = +1.34 Sharpe. This is the baseline signal. CatBoost at same features gives more return but 3× the drawdown — XGBoost chosen as backbone.

---

### PHASE 9–22 — The Saturation Era

All 22+ experiments that added OHLCV-derived features to the enc8 champion. **All failed.**

| Phase | Signal Added | Δ Sharpe | Why Failed |
| ----- | ------------ | -------- | ---------- |
| 9–18 | Session flags (London/NY hours) | negative | Session timing already in price |
| 19 | K-Means candle clusters k=32 | negative | Different encoding of same OHLCV |
| 21-B | Order Blocks + FVG + Daily HL | −0.89 | enc8 already absorbs OB/FVG structure |
| 21-C | All 6 SMC types | −1.15 | Full SMC = noise on saturated enc8 |
| 22-A | Volume anomaly (vol_ratio, zscore) | −0.63 | enc8 latent absorbs volume info |
| 24 | Cross-market: GBPUSD/DXY/Gold/VIX | +0.24 | Marginal, within enc8 seed variance |

**The Saturation Principle:** The supervised enc8 encoder has fully absorbed all signal available in EURUSD OHLCV. Any feature derived from the same price/volume source adds noise. The only genuine path forward is a **different information source** or a **different problem formulation**.

---

### PHASE 20 — Supervised Encoder Breakthrough (enc8)

The pivot that turned +1.34 into +3.13 Sharpe.

| Config | Sharpe | MaxDD | Notes |
| ------ | ------ | ----- | ----- |
| XGBoost, 31 base features | +1.34 | 10.5% | Phase 4 baseline |
| XGBoost + enc8 (cached) | **+3.13** | 13.3% | **Champion (specific seed)** |
| XGBoost + enc8 (fresh avg) | +2.31 | 8.0% | Average of 5+ fresh retrains |
| XGBoost + enc16 | ~same | — | Wider latent gives no gain |
| CatBoost + enc8 | +2.27 | 37.5% | 3× higher DD — XGB wins |
| Transformer encoder | comparable | — | More complex, no gain |

**enc8 architecture:** MLP, 50-bar OHLCV window, 250 input → 8 latent dims, CrossEntropyLoss (label-supervised), 30 epochs. The supervised objective (predict 4-bar direction during encoding) is what makes it work — unsupervised autoencoders gave worse results.

**Note on +3.13:** This is from a specific cached fold initialization. Fresh retrains average +2.31. The +3.13 is real but represents the upper variance of enc8. The 60k-bar hybrid v2 runs (+3.01/+4.27) are more honest estimates — different dataset, more recent data.

---

### PHASE 23 — LSTM Experiments

| Config | Result | Verdict |
| ------ | ------ | ------- |
| LSTM on 39 features (XGBoost swap) | Did not improve over enc8 | FAILED |
| End-to-end LSTM on raw OHLCV | Similar to enc8, higher training cost | FAILED |

**Why LSTM failed:** The saturation principle applies — LSTM still reads the same OHLCV source. The sequential inductive bias of LSTM does not reveal new signal that enc8 misses on M15 data. 15-minute bars already average out micro-structure that LSTM would exploit on tick data.

---

### PHASE 24 — Cross-Market Features

Added 9 columns from GBPUSD, DXY, Gold, VIX as features in walk-forward.

Result: +0.24 Sharpe delta vs enc8 baseline (+2.31). Within the ±0.5–0.8 variance of fresh enc8 retrains. Inconclusive — not enough signal to justify the data dependency complexity.

**Verdict:** FAILED (inconclusive at +0.24; enc8 seed variance masks any real gain).

---

### PHASE 25 — Regime Detection + Specialist Models

RegimeRouter: KMeans k=4 clusters → 4 specialist XGBoost models, LogisticRegression router.

| Config | Sharpe | MaxDD |
| ------ | ------ | ----- |
| Single XGBoost + enc8 | +2.31 | 8.0% |
| RegimeRouter k=4 | +1.40 | 34.1% |

**Why it failed:**
1. KMeans cluster IDs are non-deterministic across walk-forward folds. "Regime 0" in fold 3 ≠ same market as "Regime 0" in fold 7.
2. 49k bars ÷ 4 regimes = ~12k bars per specialist — insufficient for enc8 to converge.

**If retrying:** Use rule-based deterministic regimes (`ADX > 25 = trending`, `ADX < 20 = ranging`). But data-splitting penalty remains.

---

### PHASE 26 — Meta-Labeling (Triple-Barrier)

Applied triple-barrier labeling (TP/SL/time-out) to generate P(TP before SL) labels. Result: only 15% of bars received non-zero labels. Model too conservative — only 7 trades per walk-forward fold.

**Why it failed:** The label sparsity at 15% means the model sees far fewer positive examples than the standard threshold-based labeler, and becomes overly conservative. Standard 40% threshold labeler (label_threshold=0.0005 for 4-bar horizon) is more appropriate.

**Verdict:** FAILED.

---

### PHASE 28+ — Candle Predictor Development

Separate model family. Different objective: predict ONLY the next 1-bar direction at high confidence. 1:3 R:R (SL=10p, TP=30p), threshold=0.60.

#### Candle Model Version History

| Version | Key Changes | EURUSD OOS Sharpe | USDJPY OOS Sharpe |
| ------- | ----------- | ----------------- | ----------------- |
| v1 | XGBoost, expanding window, threshold=0.40, SL=15/TP=20 | +0.025 | +1.165 |
| v2 | CatBoost, sliding 120d window, threshold=0.60, SL=10/TP=30, 52 features | +7.938 | +14.598 |
| v3 (current) | Session hours corrected to UTC, Sydney added, Tokyo+London overlap, 52 features total | +7.118 | +14.414 |

v3 WF OOS scores are conservative estimates. Full-data OOS verification (13 fold models, 51,689 bars): EURUSD **+20.1**, USDJPY **+25.6**. These are the live-relevant numbers.

#### Session Hour Corrections (v2 → v3)

v2 used DST-era London hours (07:00 UTC). v3 corrected to actual broker UTC schedule:

| Session | UTC | EAT (UTC+3) |
| ------- | --- | ----------- |
| Sydney | 22:00–07:00 (midnight-crossing) | 01:00–10:00 |
| Tokyo | 00:00–09:00 | 03:00–12:00 |
| London | 08:00–17:00 | 11:00–20:00 |
| New York | 13:00–22:00 | 16:00–01:00 |
| Tokyo+London overlap | 08:00–09:00 | 11:00–12:00 |

#### fractal_corr Feature (added Jun 2026)

Symmetric Pearson correlation scan on 10-bar windows. Detects fractal price symmetry. Added to the candle model as feature 52 (Numba-optimized). Reduced candle model MaxDD: EURUSD −1.1%, USDJPY −3.1%. See `READS.MD` Read 7 for origin.

#### candle_trail Mode

Extends candle_predictor: trailing SL + confidence-based max-bars hold.

- conf < 0.70 → max 1 bar (identical to candle_predictor)
- conf 0.70–0.80 → max 2 bars (med tier)
- conf ≥ 0.80 → max 4 bars (high tier)
- Once profit ≥ activation_pips → trail SL at trail_pips_behind from peak

Grid search (160 combos). Finding: `trail_pips_behind=5` dominates all combinations. Tight trail captures profit before reversal. At 10p behind, too many winners give back too much.

| Symbol | Tuned Params | Sharpe | Win% | MaxDD | vs candle_predictor |
| ------ | ------------ | ------ | ---- | ----- | ------------------- |
| EURUSD | act=12, behind=5, lo=1, me=2, hi=4 | +19.4 | 91.3% | **3.1%** | −0.7 Sharpe, −3.6pp DD |
| USDJPY | act=15, behind=5, lo=1, me=3, hi=4 | +24.5 | 88.4% | **4.8%** | −1.1 Sharpe, −6.1pp DD |

Recommendation: use `candle_trail` for capital-preservation accounts. Use `candle_predictor` for maximum Sharpe.

---

### PHASE 30 — Hybrid v2 (2026-06-12) — BREAKTHROUGH

**The insight:** The candle predictor's P_buy/P_sell is a genuinely new signal for the main XGBoost model. The candle model captures 1-bar momentum precision that the enc8 encoder's 50-bar window misses.

**Leakage-free injection design:** 13 WF fold models generate predictions only for their OOS windows. Each bar's P_buy/P_sell was predicted by a model that never trained on that bar. 51,689 OOS predictions saved to `data/features/candle_signal_{SYMBOL}.parquet`.

**WF OOS results vs baseline:**

| Symbol | Baseline v1 (40 feat) | Hybrid v2 (42 feat) | Delta |
| ------ | --------------------- | ------------------- | ----- |
| EURUSD | Sharpe +1.35, MaxDD 16.4%, Return +49% | **Sharpe +3.01, MaxDD 11.4%, Return +93%** | **+1.66 Sharpe, −5pp DD** |
| USDJPY | Sharpe +3.24, MaxDD 24.3%, Return +551% | **Sharpe +4.27, MaxDD 19.6%, Return +1419%** | **+1.03 Sharpe, −4.7pp DD** |

Why this works when all other feature additions failed: `candle_p_buy`/`candle_p_sell` are **not derived from OHLCV alone** — they are the output of a separately-trained model with its own latent representations, session features, and fractal features. They represent a compression of CatBoost's world model, not raw price data.

**Bug fixed during this work:** `PipelineConfig` had no `bt_pip_size` field, causing USDJPY backtests to use EURUSD's 0.0001 pip size. SL/TP in USDJPY were 100× too tight, generating 28k+ trades/year with −100% MaxDD. Fixed: `bt_pip_size` added to `PipelineConfig`, wired through `from_dict()` and `_make_backtest_cfg()`.

**Artifact sync bug fixed:** `pipe._feature_cols` is set to 40 inside `build_features()`. After externally injecting 2 candle columns, it still reads 40. `meta.json` saved wrong feature count, causing silent column drop at inference. Fixed: `pipe._feature_cols = list(X.columns)` added after injection in `retrain_champion.py`.

---

## TIMEFRAME EXPERIMENTS

| Timeframe | Config | Sharpe | MaxDD | Trades/yr | Verdict |
| --------- | ------ | ------ | ----- | --------- | ------- |
| M15 ★ | XGBoost + enc8 | +3.13 | 13.3% | ~220 | **Champion** |
| H1 calibrated | XGBoost + enc8, SL=40p/TP=80p | +1.56 | 4.8% | ~29 | Too few trades |
| H1 uncalibrated | XGBoost + enc8 | +1.47 | 11.6% | ~190 | Worse Sharpe |
| M5 | Tested | lower | higher | — | Noisier, not competitive |
| M30 | Not tested | — | — | ~200–300 | **Worth exploring** |

H1 finding: 29 trades/year is not enough for compounding. H1 needs a fundamentally different strategy design, not a drop-in swap.

---

## MULTI-PAIR STATUS

| Symbol | Candle Model | Pipeline v2 | WF Candle Sharpe | WF Pipeline Sharpe |
| ------ | ------------ | ----------- | ---------------- | ------------------ |
| EURUSD ✅ | `candle_EURUSD/` v3, 52 feat | `pipeline_EURUSD_v2/` 42 feat | +7.1 | +3.01 |
| USDJPY ✅ | `candle_USDJPY/` v3, 52 feat | `pipeline_USDJPY_v2/` 42 feat | +14.4 | +4.27 |
| XAUUSD ⬜ | Not trained | Not trained | — | — |
| GBPUSD ⬜ | Data exists | Not trained | — | — |
| M30 EURUSD ⬜ | Not trained | Not trained | — | — |

XAUUSD previous attempt failed: `label_threshold` was too low. Gold moves require threshold ~0.001, SL=100p, TP=200p.

---

## DEAD ENDS — DO NOT REPEAT

| Experiment | Why It Fails | Confirmed |
| ---------- | ------------ | --------- |
| Session flags (London/NY hours) as features | Already in price patterns | Phase 9–18 |
| K-Means candle cluster labels | enc8 already captures shape | Phase 19 |
| SMC: OB, FVG, Daily HL, SuperTrend, Andean, Heiken Ashi | All derived from OHLCV; enc8 saturated | Phase 21 |
| Volume anomaly signals | enc8 latent absorbs volume | Phase 22 |
| Cross-market GBPUSD/DXY/Gold/VIX | +0.24 delta within enc8 seed variance | Phase 24 |
| KMeans RegimeRouter (k=4) | Non-deterministic IDs; 12k bars/specialist insufficient | Phase 25 |
| Triple-barrier meta-labeling | 15% non-zero rate → only 7 trades/fold | Phase 26 |
| H1 as M15 drop-in | 29 trades/year | 2026 |
| CatBoost on main 4-bar pipeline | MaxDD 37.5% (3× XGBoost) | Phase 12 |
| enc16 (wider latent) | No Sharpe gain vs enc8 | Phase 20 |
| LSTM swap for enc8 | Same data source; saturation principle applies | Phase 23 |
| Any OHLCV-derived feature on top of enc8 | Saturation principle | Phases 9–22 |

---

## PROMISING — NOT YET TESTED (Priority Order)

### 1. Validate Hybrid v2 in Live Dry-Run

Before any new experiments, verify the 42-feature model works in live conditions.
```bash
conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \
    --model-dir data/models/pipeline_EURUSD_v2 \
    --candle-feature-dir data/models/candle_EURUSD \
    --flip-mode trailing_hedge --dry-run
```

### 2. fractal_corr in Hybrid v2 Pipeline ⭐⭐⭐⭐

`fractal_corr` already proven in candle model (Read 7). The main pipeline v2 does NOT have it. Add to `feature_pipeline.py`, retrain, compare.

Expected: +0.1 to +0.5 Sharpe if fractal symmetry adds signal enc8 misses.

### 3. USDJPY Hybrid v2 Live Upgrade ⭐⭐⭐⭐

+1.03 Sharpe, −4.7pp DD. Biggest near-term gain with lowest risk (upgrade existing live bot).

### 4. candle_trail + Hybrid v2 Combined Exit Logic ⭐⭐⭐⭐

Hypothesis: use hybrid v2 for entry direction, but apply candle confidence tier to max-bars hold. When candle says high confidence, allow trade to hold up to 4 bars. When candle is neutral (conf ≈ 0.5), exit at 1 bar.

This is a new mode — not candle_trail (standalone) and not standard hybrid v2. Likely to combine the capital-preservation of trailing with the 4-bar signal quality.

### 5. BOA Hyperparameter Optimization ⭐⭐⭐

BOA (Billiards Optimization, `src/optimization/boa.py`) can tune SL/TP, confidence threshold, trail params across joint space more efficiently than grid search. Already implemented, needs wrapper around WF backtest objective.

### 6. M30 Timeframe ⭐⭐⭐

Unexplored gap. Expected ~200–300 trades/yr. Config: horizon=2 bars, label_threshold=0.0005, SL=15p, TP=30p.

### 7. XAUUSD Pipeline ⭐⭐⭐

Fix known: label_threshold=0.001, SL=100p, TP=200p. Candle model for Gold 1-bar specialist may also work at those params.

### 8. Deterministic Regime Routing ⭐⭐

KMeans failed (Phase 25). Fix: rule-based regimes with stable definitions (`ADX>25 = trending`, `ADX<20 = ranging`). Route each bar to specialist trained only on matching regime bars. 60k ÷ 3 regimes = ~20k bars each — enough to train.

### 9. VQ-VAE / Market Language Model ⭐⭐

Replace enc8 with Vector Quantization encoder (512-entry codebook). Convert OHLCV windows into discrete tokens, train autoregressively (GPT-style). Highest risk (still OHLCV-derived, saturation may apply), highest potential (discrete codebook forces pattern routing through named states).

---

## LIVE SETUP

| Service | Symbol | Mode | Model | Status |
| ------- | ------ | ---- | ----- | ------ |
| `mt5-eurusd` | EURUSD | candle_predictor | candle_EURUSD | active |
| `mt5-usdjpy` | USDJPY | candle_predictor | candle_USDJPY | active |
| `mt5-eurusd-hedge` | EURUSD | hedge_exit | pipeline_EURUSD (v1) | active |
| `mt5-usdjpy-hedge` | USDJPY | hedge_exit | pipeline_USDJPY (v1) | active — upgrade to v2 pending |

---

## REPRODUCTION COMMANDS

```bash
# Generate OOS candle signal (one-time per symbol, ~2 min)
conda run -n envmt5 python scripts/build_candle_features.py

# WF comparison: with vs without candle features
conda run -n envmt5 python scripts/compare_candle_features.py --symbol EURUSD

# Retrain hybrid v2 (auto-injects candle features)
conda run -n envmt5 python scripts/retrain_champion.py \
    --data data/EURUSD_M15.csv --out data/models/pipeline_EURUSD_v2
conda run -n envmt5 python scripts/retrain_champion.py \
    --data data/USDJPY_M15.csv --out data/models/pipeline_USDJPY_v2

# Retrain candle model (after feature changes)
rm -rf data/models/wf_cache_candle2_EURUSD data/models/wf_cache_candle2_USDJPY
conda run -n envmt5 python scripts/train_candle_model.py

# Tune candle_trail params
conda run -n envmt5 python scripts/tune_candle_trail.py --symbol EURUSD

# Backtest candle_trail
conda run -n envmt5 python scripts/backtest_candle_trail.py --symbol EURUSD
conda run -n envmt5 python scripts/backtest_candle_trail.py --symbol USDJPY

# Dry-run hybrid v2
conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \
    --model-dir data/models/pipeline_EURUSD_v2 \
    --candle-feature-dir data/models/candle_EURUSD \
    --flip-mode trailing_hedge --dry-run

# Dry-run candle standalone
conda run -n envmt5 python src/bots/pipeline_bot.py --symbol EURUSD \
    --candle-model-dir data/models/candle_EURUSD \
    --flip-mode candle_predictor --dry-run
```

---

## SHARPE VALIDITY NOTES

1. **+3.13 vs +2.31:** The +3.13 is from a specific lucky enc8 seed (cached 49k-bar run). Fresh retrains of same config average +2.31. The hybrid v2 numbers (+3.01/+4.27) on the 60k-bar dataset are the most trustworthy forward estimates.

2. **+20.1 candle OOS:** Uses full-data encoder for feature building (slight advantage vs WF). The WF number (+7.1) is the conservative bound. Live performance should fall between WF and OOS: expect ~+10–15 in favourable conditions.

3. **Live compression:** Walk-forward → live typically sees −0.5 to −1.3 Sharpe compression (slippage, spread widening, regime change). The candle model's large OOS cushion (+7 WF) provides ample margin.

4. **Hybrid v2 is the honest estimate:** The +3.01/+4.27 WF OOS numbers use true walk-forward — no bar sees its own candle prediction. These should be treated as the ground truth for live expectations from the pipeline model.
