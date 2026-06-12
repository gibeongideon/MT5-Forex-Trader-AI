# MT5 Trading Bot — Build Progress

> **Purpose:** Phase status tracking. New sessions: read WIN-RESEARCH.MD for full context before starting work.
> Last updated: 2026-06-12

---

## Current Status

| Phase | Name | Status | Result |
| ----- | ---- | ------ | ------ |
| 1 | Random Baseline + Logging Infrastructure | ✅ COMPLETE | Sharpe −0.17 (noise floor confirmed) |
| 2 | Composable Indicator Library + Rule Engine | ✅ COMPLETE | Sharpe −0.45 (no ML edge) |
| 3 | Feature Engineering Pipeline | ✅ COMPLETE | Foundation for all phases |
| 4 | XGBoost with Calibrated Probabilities | ✅ COMPLETE | Sharpe +1.34 (first real edge) |
| 5 | Pluggable Model Registry | ✅ COMPLETE | LightGBM +0.72, RF +0.24 — XGBoost wins |
| 6 | Signal Stacking & Meta-Learning | ✅ COMPLETE | CatBoost +1.17 but MaxDD 24.7% — too risky |
| 7 | Robust Backtesting & Walk-Forward | ✅ COMPLETE | WF infrastructure, bt_pip_size bug fixed Jun 2026 |
| 8 | Intelligent Risk Management | ✅ COMPLETE | Tiered risk; ATR stop too conservative |
| 9–19 | Feature experiments (session, K-Means, SMC ×6, volume) | ✅ ALL FAILED | Saturation principle confirmed — 22 experiments |
| 20 | Supervised Latent Encoder (enc8) | ✅ COMPLETE | Sharpe +3.13 (cached), +2.31 (avg) — BREAKTHROUGH |
| 21 | SMC/ICT signal comparison (6 types) | ✅ COMPLETE | All hurt Sharpe — saturation confirmed |
| 22 | Production hardening + volume anomaly | ✅ COMPLETE | Volume hurt Sharpe |
| 23 | LSTM experiments | ✅ FAILED | Both LSTM variants worse than enc8 |
| 24 | Cross-market features (GBPUSD/DXY/Gold/VIX) | ✅ INCONCLUSIVE | +0.24 delta within enc8 seed variance |
| 25 | Regime detection + KMeans routing | ✅ FAILED | Non-deterministic cluster IDs; insufficient data per specialist |
| 26 | Meta-labeling (triple-barrier) | ✅ FAILED | 15% non-zero labels → only 7 trades/fold |
| 27 | USDJPY model + multi-pair expansion | ✅ COMPLETE | USDJPY WF Sharpe +3.24 baseline; now +4.27 hybrid v2 |
| 28 | CatBoost Candle Predictor (1-bar, 52 feat) | ✅ COMPLETE | EURUSD +20.1, USDJPY +25.6 (OOS) |
| 29 | candle_trail mode + tuning | ✅ COMPLETE | EURUSD +19.4/3.1% DD, USDJPY +24.5/4.8% DD (tuned) |
| 30 | **Hybrid v2: candle features → XGBoost** | ✅ **CHAMPION** | EURUSD +3.01 (+1.66), USDJPY +4.27 (+1.03) — **BREAKTHROUGH** |

---

## Active Priorities

| # | Task | Est. Effort | Why Now |
| - | ---- | ----------- | ------- |
| 1 | Dry-run hybrid v2 EURUSD + USDJPY | 0.5 day | Validate before live deploy |
| 2 | Upgrade mt5-usdjpy-hedge to pipeline_USDJPY_v2 | 1 day | +1.03 Sharpe, −4.7pp DD waiting |
| 3 | fractal_corr feature in main pipeline v2 | 2 days | Proven in candle model; structurally different from OHLCV |
| 4 | M30 timeframe experiment | 3 days | Unexplored; expected 200–300 trades/yr |
| 5 | XAUUSD candle + pipeline (fixed threshold) | 2 days | Previous attempt failed due to wrong label_threshold |
| 6 | candle_trail + hybrid v2 combined exit mode | 3 days | New mode: hybrid entry + candle confidence for hold duration |

---

## Champion Summary (All-Time)

| Category | Champion | Sharpe | Notes |
| -------- | -------- | ------ | ----- |
| Pipeline EURUSD | Hybrid v2, 42 feat, XGBoost+enc8+candle | +3.01 | WF OOS, 60k bars |
| Pipeline USDJPY | Hybrid v2, 42 feat, XGBoost+enc8+candle | +4.27 | WF OOS, 60k bars |
| Candle EURUSD | candle_predictor (CatBoost v3, 52 feat) | +20.1 | Full OOS verified |
| Candle EURUSD low-DD | candle_trail tuned | +19.4 | MaxDD 3.1% |
| Candle USDJPY | candle_predictor (CatBoost v3, 52 feat) | +25.6 | Full OOS verified |
| Candle USDJPY low-DD | candle_trail tuned | +24.5 | MaxDD 4.8% |

---

## Artifact Directory

| Path | Contents | Last Trained |
| ---- | -------- | ------------ |
| `data/models/pipeline_EURUSD_v2/` | 42-feat XGBoost+enc8+candle, EURUSD | 2026-06-12 |
| `data/models/pipeline_USDJPY_v2/` | 42-feat XGBoost+enc8+candle, USDJPY | 2026-06-12 |
| `data/models/pipeline_EURUSD/` | 40-feat XGBoost+enc8 baseline v1, EURUSD | 2026-05-xx |
| `data/models/pipeline_USDJPY/` | 40-feat XGBoost+enc8 baseline v1, USDJPY | 2026-05-xx |
| `data/models/candle_EURUSD/` | CatBoost candle v3, 52 feat, EURUSD | 2026-06-xx |
| `data/models/candle_USDJPY/` | CatBoost candle v3, 52 feat, USDJPY | 2026-06-xx |
| `data/features/candle_signal_EURUSD.parquet` | 51,689 OOS candle predictions | 2026-06-12 |
| `data/features/candle_signal_USDJPY.parquet` | 51,689 OOS candle predictions | 2026-06-12 |
| `data/models/wf_cache_candle2_EURUSD/` | 13 fold candle models (build_candle_features.py source) | 2026-06-xx |
| `data/models/wf_cache_candle2_USDJPY/` | 13 fold candle models | 2026-06-xx |
