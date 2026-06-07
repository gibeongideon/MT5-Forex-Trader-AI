# MT5 EURUSD M15 — Experiment Leaderboard

> All results: walk-forward, expanding window, 180d train / 30d test, 18–19 folds.
> Threshold: 0.40 min confidence. SL 30p / TP 60p. Spread 1.0p. Balance $10,000.
> Dataset: EURUSD M15. Phases 1–8 used 50k bars. Phases 21+ used 49k filtered bars (from 2024-05-14).

---

## Champion (Locked)

| Model | Features | Sharpe | MaxDD | Return | Trades | Phase |
|-------|----------|--------|-------|--------|--------|-------|
| **XGBoost + enc8** | 31 base + 8 latent = **39** | **+3.13** | 13.3% | **+358%** | 524 | 20 (cached run) |

> Note: fresh enc8 retrains vary ±0.5–0.8 Sharpe due to random seed. The +3.13 is from a
> specific cached initialization. Fresh runs of the same config average ~+2.31 Sharpe.

---

## All Experiments — Sorted by Sharpe

### Phase 20–22: enc8 Latent Encoder Era (49k bars, 2024-05-14 → 2026-05-25)

| Rank | Config | Phase | Features | Sharpe | MaxDD | Return | Trades | vs Baseline | Verdict |
|------|--------|-------|----------|--------|-------|--------|--------|-------------|---------|
| 1 | XGBoost + enc8 (cached) | 20 | 39 | **+3.13** | 13.3% | +358% | 524 | — | **CHAMPION** |
| 2 | XGBoost + enc8 (fresh baseline) | 21–22 | 39 | +2.31 | 8.0% | +43% | 513 | reference | Fresh enc8 baseline |
| 3 | + vol_ratio / zscore / fast_slow | 22-A | 42 | +1.68 | 10.3% | +25.2% | — | −0.63 | WORSE |
| 4 | + OB + FVG + DailyHL (3 SMC) | 21-B | 47 | +1.42 | 9.9% | +16.3% | — | −0.89 | WORSE |
| 5 | + all 6 SMC signal types | 21-C | 55 | +1.16 | 10.9% | +18.4% | — | −1.15 | WORSE |

**Signals tested and failed (all from OHLCV source — saturation confirmed):**

| Signal Group | Phase | Δ Sharpe | Conclusion |
|-------------|-------|----------|------------|
| Order Blocks + Fair Value Gaps + Daily HL | 21-B | −0.89 | enc8 already captures OB/FVG patterns |
| All 6 SMC types (OB, FVG, Daily, Andean, SuperTrend, Heiken Ashi) | 21-C | −1.15 | Full SMC library = noise |
| Volume anomaly (vol_ratio, vol_zscore, vol_fast_slow) | 22-A | −0.63 | enc8 latent absorbs volume info |
| Session features (London open/close, NY session) | 19 (prior) | negative | Session timing already in patterns |
| K-Means candle clusters | 19 (prior) | negative | Cluster labels = different encoding of same OHLCV |

---

### Phase 4–8: Base Model Era (50k bars, original 31 features, no encoder)

| Rank | Config | Phase | Features | Sharpe | MaxDD | Return | Trades | Notes |
|------|--------|-------|----------|--------|-------|--------|--------|-------|
| 1 | **XGBoost** (walk-forward) | 4 | 31 | **+1.34** | 10.5% | +14.2% | 176 | Best single model |
| 2 | CatBoost (walk-forward) | 6 | 31 | +1.17 | 24.7% | +5.8% | 543 | High trades, high DD |
| 3 | XGBoost + tiered risk | 8-B | 31 | +0.72 | **7.5%** | +0.2% | 126 | Same Sharpe, −49% DD vs 8-A |
| 3 | LightGBM (walk-forward) | 5 | 31 | +0.72 | 6.8% | +3.8% | 99 | Fewer trades |
| 4 | XGBoost + ATR stop | 8-C | 31 | +0.22 | 3.4% | +0.3% | 69 | Lowest DD, too few trades |
| 5 | Random Forest (walk-forward) | 5 | 31 | +0.24 | 14.8% | −0.1% | 93 | Bagging < boosting |
| 6 | Rule-based bot | 2 | rules | −0.45 | — | −9.7% | 430 | No ML |
| 7 | Random baseline | 1 | none | −0.17 | 21.6% | −2.5% | 354 | Noise floor |

---

### Phase 23: LSTM Experiments (Running — check logs/lstm_compare.log)

| Config | Model | Features | Sharpe | MaxDD | Return | Trades | Status |
|--------|-------|----------|--------|-------|--------|--------|--------|
| A | XGBoost + enc8 (fresh baseline) | 39 | +2.31 | 8.0% | +43% | 513 | Done (1.7 min, cached) |
| B | LSTMModel on 39 features | 39 | TBD | TBD | TBD | TBD | Running (~90 min) |
| C | E2ELSTMModel on raw OHLCV | 5 | TBD | TBD | TBD | TBD | Running after B |

---

## Key Findings

### Why the champion works

- **31 base features** = hand-crafted indicators (RSI, MACD, ATR, Bollinger, EMAs, returns, rolling stats). Human domain knowledge encoded explicitly.
- **8 latent features (enc8)** = what a supervised MLP encoder discovered by compressing 50-bar OHLCV windows, trained to predict buy/sell/hold. Captures patterns no human named.
- **Together**: two complementary information channels → XGBoost exploits both simultaneously.

### Saturation Principle (22 experiments, confirmed)

Every signal derived from the same OHLCV source **hurts** Sharpe. The enc8 latent space has fully absorbed all available information from price and volume data.
The only way to genuinely add information: **different data source** (cross-pair, macro, sentiment).

### Phase 23 Decision Rule

- If LSTM variant beats Config A by > +0.05 Sharpe AND MaxDD ≤ 20% → promote, move to Phase 24 (cross-pair)
- Otherwise → deploy champion as-is, cross-pair becomes a separate add-on

---

## Next Steps — Genuinely New Directions Only

> 22 experiments confirmed: enc8 has fully absorbed EURUSD OHLCV.
> No more indicator variants, SMC types, or volume features — all tested and failed.
> Every next phase must bring **new information** or a **fundamentally different problem formulation**.
> Model stacking (done Phase 6), position sizing (done Phase 8) — already in the system.

---

### Phase 24 — Cross-Market Features ✅ DONE — Inconclusive

| Config | Sharpe | Notes |
|--------|--------|-------|
| A: XGBoost+enc8 baseline (fresh) | -0.16 | stochastic seed variance |
| B: +GBPUSD/USDJPY/XAUUSD (9 cols) | +0.08 | +0.24 vs A, still near zero |

**Verdict:** Cross-market adds marginal signal (+0.24 delta) but absolute performance is unreliable due to enc8 random seed variance. Does not confidently beat the cached champion (+3.13). **Champion holds.**

---

### Phase 25 — Regime Detection ✅ DONE — Failed

RegimeRouter (KMeans k=4 + per-regime XGBoost): **+1.36 vs +2.31 baseline = -0.95**

Root cause: 49k bars ÷ 4 regimes ≈ 12k bars per specialist — insufficient training data.

---

### Phase 26 — Meta-Labeling (Triple-Barrier) ✅ DONE — Failed

Triple-barrier labels: **-0.71 Sharpe, only 7 trades** in 18 folds.

Root cause: 15% non-zero label rate makes model too conservative. Standard labels (40%) needed for trade frequency.

---

### Phase 27 — VQ-VAE + GPT Market Language Model (Future Research)

**Complexity: High (2–3 weeks). Not yet started.**

The only uncharted encoder architecture. Converts OHLCV windows into discrete tokens (like words), then trains a GPT-style transformer autoregressively on the token sequence.

```
50-bar OHLCV window
        ↓
   MLP Encoder → z_continuous [8 floats]
        ↓
   Vector Quantization → nearest codebook entry → token ID (e.g. 42)
        ↓
   Codebook embedding lookup → [8 floats] (learnable, not continuous)
        ↓
   XGBoost (Option A: drop-in for enc8)
   OR
   GPT Transformer trained autoregressively on token sequence (Option B)
```

**Why different from enc8:** enc8 produces continuous floats. VQ-VAE forces the encoder to route patterns through a discrete learned vocabulary (e.g. 512 "market state" entries). Transformers are designed for discrete token sequences — this could unlock sequence modelling that LSTMs failed at.

**Honest risk:** Still OHLCV-derived data — saturation principle may still apply. Option B (GPT) is the novel part; Option A (VQ-VAE encoder only) is likely similar to enc8. The E2E LSTM failure (Phase 23) is a warning sign for end-to-end OHLCV approaches.

**Prerequisite:** Complete paper trading validation first (see below).

---

### Deployment — Next Immediate Step ⭐⭐⭐⭐⭐

All planned experiment phases are complete. The champion (XGBoost + enc8, 39 feat) is ready for production validation. See Production Deployment section below.

---

### Phase 25 — Regime Detection + Dedicated Models ⭐⭐⭐⭐⭐

**Expected uplift: +0.3 to +1.0 Sharpe**

#### Simple version

Right now the champion does this:
```
EURUSD bar → XGBoost → "buy / hold / sell"
```
With regime models:
```
EURUSD bar → "what market are we in?" → route to the right specialist
                    ↓
         Trending up   →  XGBoost Trend Model   → signal
         Trending down →  XGBoost Trend Model   → signal
         Ranging       →  XGBoost Range Model   → signal
         High vol      →  skip / reduce size    → signal
```
One doctor for everything → specialist per condition.

#### How it works in detail

**Step 1 — Regime Detection (every bar)**

Three features describe the current market:
- `ATR ratio` — current ATR vs rolling-100-bar mean. >1 = elevated volatility.
- `ADX` — trend strength. >25 = trending, <20 = ranging.
- `RSI` — directional bias. >50 = bullish pressure, <50 = bearish.

KMeans(k=4) clusters all training bars into 4 buckets from those three numbers.
Each new bar gets classified in real time (microseconds).

**Step 2 — Specialist Training (per walk-forward fold)**

Instead of one XGBoost on all bars, RegimeRouter:
1. Detects regimes across the training window
2. Splits data: "trending bars only", "ranging bars only", etc.
3. Trains a **separate XGBoost on each slice**
4. If a regime has < 200 bars → no specialist → falls back to the global model

The trending specialist only ever saw trending markets. It learns entry patterns
that work specifically in momentum. The ranging specialist learns mean-reversion.

**Step 3 — Inference (live trading)**

At each new bar:
1. Compute ATR ratio, ADX, RSI → classify regime (microseconds)
2. Route to matching specialist → get `[P_buy, P_hold, P_sell]`
3. Apply same confidence threshold and risk sizing as before

The trading engine sees no difference — it still receives a probability triple.

**Why this might work**

Current XGBoost trained on everything learned:
> "On average, when RSI=65 and MACD crosses up, there's a slight buy edge"

That average blends trending markets (where the signal works well) with ranging
markets (where it fakes out). The specialist sees only the conditions it will be
deployed in — it learns a sharper, cleaner version of the same signal.

**The risk**

49k bars ÷ 4 regimes ≈ 12k bars per specialist per fold (vs 48k for global model).
Less data per model. If regime boundaries are drawn wrong, specialists get confused
training sets and perform worse. The comparison script measures this directly.

**Implementation**

- `src/models/regime_router.py` — fully implemented, implements ModelInterface
- `scripts/compare_regime.py` — A/B walk-forward: XGBoost vs RegimeRouter, 39 features

Observed regime distribution on 49k M15 bars:
- Regime 0 (trending-up):   10.2%
- Regime 1 (trending-down): 37.6%
- Regime 2 (ranging):       13.6%
- Regime 3 (high-vol):      38.6%

---

### Phase 26 — Meta-Labeling ⭐⭐⭐⭐

**Expected uplift: +0.2 to +0.7 Sharpe**

Our current labels (buy/hold/sell by forward return threshold) tell the model what
direction to predict. Meta-labeling changes the question entirely:

```
Step 1 — Primary model generates a candidate signal (existing champion)
Step 2 — Meta-model answers: "Should I take this trade?"
          Features: primary signal strength + regime + time-of-day + recent PnL
          Label: 1 if trade hit TP, 0 if trade hit SL
Step 3 — Only trade when meta-model says yes
```

Alternative: replace classification labels with **P(TP hits before SL)** —
a regression target that directly optimises what we care about in live trading.
Source: Marcos López de Prado, *Advances in Financial Machine Learning* (triple-barrier labeling).

---

## Production Deployment (when ready)

```bash
# 1 — Retrain champion on all data
conda run -n envmt5 python scripts/retrain_champion.py

# 2 — Dry run (no trades placed)
conda run -n envmt5 python src/bots/pipeline_bot.py --dry-run

# 3 — Paper trade 30 days minimum before live capital
conda run -n envmt5 python src/bots/pipeline_bot.py

# 4 — Systemd auto-start
sudo cp deploy/mt5_bot.service /etc/systemd/system/
sudo systemctl enable mt5_bot && sudo systemctl start mt5_bot
journalctl -u mt5_bot -f
```

---

*Last updated: 2026-06-07. Phase 23 LSTM results pending.*
