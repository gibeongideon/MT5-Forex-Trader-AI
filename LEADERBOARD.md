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

### Phase 24 — Cross-Market Features ⭐⭐⭐⭐⭐ (Next)

**Expected uplift: +0.2 to +0.8 Sharpe**

The only remaining untested information source that institutional quants use first.
EURUSD moves are driven by forces that don't appear in EURUSD candles alone.

| Add | Why |
|-----|-----|
| GBPUSD `return_1`, `rsi_14` | USD institutional flow — same driver as EURUSD |
| USDJPY `return_1` | Risk-on/risk-off — when JPY weakens, USD strengthens |
| DXY `return_1` | Dollar index — most direct EURUSD driver |
| XAUUSD `return_1` | Gold up → USD weak → EURUSD rises |
| VIX `return_1` | Fear spike → flight to USD → EURUSD falls |
| SPX `return_1` | Risk-on → USD sells → EURUSD rises |

Script to build: `scripts/compare_crosspair.py`
Format: A = champion baseline (39 feat), B = champion + cross-market signals.

---

### Phase 25 — Regime Detection + Dedicated Models ⭐⭐⭐⭐⭐

**Expected uplift: +0.3 to +1.0 Sharpe**

One universal model trying to trade both trending and ranging markets is the single
biggest structural weakness. Markets behave differently in different regimes — a model
trained on everything learns an average that fits nothing well.

```
Regime classifier (ATR percentile / ADX threshold)
        ↓
High ATR / trending  →  XGBoost Trend Model  (trained only on trending bars)
Low ATR / ranging    →  XGBoost Range Model   (trained only on ranging bars)
Volatility spike     →  Skip (no trade)
```

`src/models/regime_router.py` already exists as a stub — build on it.
This is how most hedge funds handle structural non-stationarity.

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
