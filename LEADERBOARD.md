# MT5 M15 — Experiment Leaderboard

> All pipeline results: walk-forward, expanding window, 180d train / 30d test.
> All candle results: OOS verified, 13 fold models, 51,689 bars, 2024-05-08 → 2026-06-05.
> Dataset: EURUSD + USDJPY M15, 60,000 bars (2024-01-08 → 2026-06-05) unless noted.
> Last updated: 2026-06-12

---

## Current Champions

### Candle Standalone (CatBoost 1-bar, OOS)

| Model | Symbol | Sharpe | Win% | MaxDD | Return | Artifact |
| ----- | ------ | ------ | ---- | ----- | ------ | -------- |
| candle_predictor | EURUSD | **+20.1** | 87.1% | 6.7% | +840% | `candle_EURUSD/` |
| candle_predictor | USDJPY | **+25.6** | 81.1% | 10.9% | +2,100% | `candle_USDJPY/` |
| candle_trail (tuned) | EURUSD | **+19.4** | 91.3% | **3.1%** | +780% | `candle_EURUSD/` |
| candle_trail (tuned) | USDJPY | **+24.5** | 88.4% | **4.8%** | +1,900% | `candle_USDJPY/` |

### Hybrid v2 (XGBoost + enc8 + candle injection, WF OOS) ★ CHAMPION PIPELINE

| Symbol | Features | Sharpe | MaxDD | Return | Trades | Artifact |
| ------ | -------- | ------ | ----- | ------ | ------ | -------- |
| EURUSD | 42 | **+3.01** | 11.4% | +93% | 540 | `pipeline_EURUSD_v2/` |
| USDJPY | 42 | **+4.27** | 19.6% | +1419% | 1,395 | `pipeline_USDJPY_v2/` |

---

## Pipeline Leaderboard — Full History

### 60k-bar era (2024-01-08 → 2026-06-05, EURUSD + USDJPY)

| Rank | Config | Symbol | Phase | Features | Sharpe | MaxDD | Return | Verdict |
| ---- | ------ | ------ | ----- | -------- | ------ | ----- | ------ | ------- |
| 1 | **Hybrid v2: XGBoost + enc8 + candle** | USDJPY | 30 | 42 | **+4.27** | 19.6% | +1419% | CHAMPION |
| 2 | **Hybrid v2: XGBoost + enc8 + candle** | EURUSD | 30 | 42 | **+3.01** | 11.4% | +93% | CHAMPION |
| 3 | XGBoost + enc8 baseline v1 | USDJPY | 20 | 40 | +3.24 | 24.3% | +551% | superseded by v2 |
| 4 | XGBoost + enc8 baseline v1 | EURUSD | 20 | 40 | +1.35 | 16.4% | +49% | superseded by v2 |

### 49k-bar era (2024-05-14 → 2026-05-25, EURUSD only)

| Rank | Config | Phase | Features | Sharpe | MaxDD | Return | Trades | vs Baseline | Verdict |
| ---- | ------ | ----- | -------- | ------ | ----- | ------ | ------ | ----------- | ------- |
| 1 | XGBoost + enc8 (cached seed) | 20 | 39 | **+3.13** | 13.3% | +358% | 524 | — | Prev champion |
| 2 | XGBoost + enc8 (fresh baseline) | 21–22 | 39 | +2.31 | 8.0% | +43% | 513 | reference | enc8 average |
| 3 | CatBoost + enc8 | 12 | 39 | +2.27 | 37.5% | +376% | 517 | −0.04 Sharpe | WORSE (3× DD) |
| 4 | + Cross-market (GBPUSD/DXY/Gold/VIX) | 24 | 48 | +2.55 | 9.3% | +64% | — | +0.24 | INCONCLUSIVE |
| 5 | + vol_ratio / zscore / fast_slow | 22-A | 42 | +1.68 | 10.3% | +25% | — | −0.63 | FAILED |
| 6 | + OB + FVG + DailyHL (3 SMC) | 21-B | 47 | +1.42 | 9.9% | +16% | — | −0.89 | FAILED |
| 7 | RegimeRouter KMeans k=4 | 25 | 39 | +1.40 | 34.1% | +110% | — | −0.91 | FAILED |
| 8 | + all 6 SMC types | 21-C | 55 | +1.16 | 10.9% | +18% | — | −1.15 | FAILED |
| 9 | LSTM on 39 features | 23 | 39 | <+2.31 | — | — | — | negative | FAILED |
| 10 | E2E LSTM raw OHLCV | 23 | — | <+2.31 | — | — | — | negative | FAILED |
| 11 | Triple-barrier meta-labeling | 26 | 39 | −0.71 | — | — | ~7 | — | FAILED |

### Base era (Phases 1–8, 50k bars, EURUSD)

| Rank | Config | Phase | Features | Sharpe | MaxDD | Return | Verdict |
| ---- | ------ | ----- | -------- | ------ | ----- | ------ | ------- |
| 1 | XGBoost + tiered risk | 8-B | 31 | +0.72 | 7.5% | +0.2% | Best before enc8 |
| 2 | XGBoost 31 features | 4 | 31 | +1.34 | 10.5% | +14.2% | First real edge |
| 3 | CatBoost 31 features | 6 | 31 | +1.17 | 24.7% | +5.8% | High DD |
| 4 | LightGBM 31 features | 5 | 31 | +0.72 | 6.8% | +3.8% | Worse |
| 5 | Random Forest | 5 | 31 | +0.24 | 14.8% | −0.1% | Weak |
| 6 | XGBoost + ATR stop | 8-C | 31 | +0.22 | 3.4% | +0.3% | Too few trades |
| 7 | Rule-based MA cross | 2 | — | −0.45 | — | −9.7% | No ML edge |
| 8 | Random baseline | 1 | 0 | −0.17 | 21.6% | −2.5% | Noise floor |

---

## Candle Model Version Leaderboard

| Version | EURUSD WF Sharpe | USDJPY WF Sharpe | Notes |
| ------- | --------------- | --------------- | ----- |
| v1 | +0.025 | +1.165 | XGBoost, threshold=0.40, SL=15/TP=20 — FAILED for EURUSD |
| v2 | +7.938 | +14.598 | CatBoost, threshold=0.60, SL=10/TP=30 — session hours slightly wrong |
| v3 (current) | +7.118 | +14.414 | Session UTC corrected, Sydney+TKLon overlap added, 52 features |

Full-data OOS verification (51,689 bars): EURUSD +20.1, USDJPY +25.6

candle_trail tuning results (grid search 160 combos):

| Symbol | act | behind | lo | me | hi | Sharpe | Win% | MaxDD | vs candle_predictor DD |
| ------ | --- | ------ | -- | -- | -- | ------ | ---- | ----- | ---------------------- |
| EURUSD | 12 | 5 | 1 | 2 | 4 | +19.4 | 91.3% | 3.1% | −53% |
| USDJPY | 15 | 5 | 1 | 3 | 4 | +24.5 | 88.4% | 4.8% | −56% |

---

## Flip Mode Reference (Pipeline, in-sample, 60k M15 EURUSD)

| Mode | Trades | Win% | Sharpe | MaxDD | Notes |
| ---- | ------ | ---- | ------ | ----- | ----- |
| always | 7,636 | 61% | +12.97 | 4.9% | |
| hedge_loss | 4,013 | 71% | +7.47 | 4.9% | fewest trades, highest win% |
| hedge_exit | 6,033 | 61% | +10.12 | 5.3% | |
| trailing_hedge | 5,635 | 69% | +9.23 | 3.2% | lowest DD |
| lock | TBD | — | — | — | |
| ratio_hedge | TBD | — | — | — | |
| partial_close | TBD | — | — | — | |
| zone_recovery | TBD | — | — | — | |

---

## Key Findings (Do Not Re-test)

**Saturation Principle:** enc8 has absorbed all EURUSD OHLCV signal. All 22+ experiments adding OHLCV-derived features to enc8 hurt Sharpe. The only path to improvement is a genuinely different information source.

**Hybrid v2 works because:** `candle_p_buy`/`candle_p_sell` are outputs of a separately-trained CatBoost model with its own latent representations — not raw OHLCV. This is structurally new information for XGBoost.

**XGBoost > CatBoost for 4-bar pipeline:** CatBoost delivers more raw return but at 3× the MaxDD. Not worth the DD cost.

**M15 > H1 for this architecture:** H1 = 29 trades/year. Compounding requires at least 200+ trades/year.

**KMeans regimes are unstable:** Cluster IDs are non-deterministic across walk-forward folds. Only deterministic rule-based regimes (ADX thresholds) would be stable enough to route correctly.
