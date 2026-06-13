# Honest Re-Validation of Live Champions — Phase 1 Verdict

> Generated 2026-06-13. Leak-free walk-forward re-test of the deployed models.
> Supersedes the performance claims in `WIN-RESEARCH.MD` and `PROGRESS.md`.

## TL;DR

**Every headline Sharpe in `WIN-RESEARCH.MD` is inflated by two independent
look-ahead leaks. Once both are removed, all live champions are negative-to-flat.**

| Model | Claimed (WIN-RESEARCH) | Honest (leak-free WF) | Verdict |
| ----- | ---------------------- | --------------------- | ------- |
| Candle EURUSD (CatBoost, 52 feat) | +7.1 WF / +20.1 "OOS", 87% win | **−0.84 Sharpe, 45.1% win, 102 tr** | RETIRE / rework |
| Candle USDJPY (CatBoost, 52 feat) | +14.4 WF / +25.6 "OOS", 81% win | **−1.31 Sharpe, 40.2% win, 495 tr** | RETIRE / rework |
| Hybrid v2 EURUSD (XGB+enc8+candle) | +3.01 Sharpe | negative (proxy −6.8, see below) | RETIRE / rework |
| Hybrid v2 USDJPY (XGB+enc8+candle) | +4.27 Sharpe | negative (proxy −8.8, see below) | RETIRE / rework |

The win rate collapse (87% → 45%) and Sharpe collapse (+7 → −0.8) are the
signature of removed look-ahead, not noise.

---

## The two leaks

### Leak #1 — Encoder look-ahead
The enc8 MLP encoder is fit on the first 80% of ALL data
(`scripts/train_candle_model.py:179`, `train_frac=0.80`). The sliding-120d
walk-forward then reuses that single encoder for every fold, so early
out-of-sample windows fall **inside** the encoder's training set. The encoder
has effectively seen the future for those folds.

`scripts/verify_candle_oos.py` argues "the WF number is the conservative bound."
That reasoning is **backwards** — it verifies *with the full-data encoder*, which
is the leak itself. Verified > WF because the full-data encoder leaks even more.

### Leak #2 — Multi-timeframe EMA look-ahead (found this session)
`scripts/train_candle_model.py:_add_extra_features` (lines 93–109) builds 1H/4H
EMA features:
```python
close_1h   = df["close"].resample("1h").last().ffill()   # bin labeled by LEFT edge
ema_1h_m15 = ema.reindex(df.index, method="ffill")        # ffilled back onto M15
```
`resample("1h")` labels each bin by its **left edge** (10:00) but the value is the
**last** M15 close in the bin (10:45). After `ffill`, the **10:15 bar is assigned
the 10:45 close — 30 minutes in the future**. The 4H features leak up to **3h45**
ahead. Confirmed empirically: at 2024-01-11 00:15, `ema_1h_ratio` uses the
00:45 close.

This matters most for the candle model's **1-bar horizon** — peeking 30 min ahead
is almost the entire prediction target.

---

## Evidence

### Candle model — full clean WF (per-fold encoder + fixed MTF)
`scripts/backtest_candle_clean_wf.py` (sliding 120d/60d, CatBoost, thr=0.60,
SL=10p/TP=30p, 1-bar force-close):

| Symbol | Sharpe | Win% | Profit factor | MaxDD | Trades | vs claimed |
| ------ | ------ | ---- | ------------- | ----- | ------ | ---------- |
| EURUSD | **−0.836** | 45.1% | 0.77 | 12.6% | 102 | +7.118 → −0.836 (−112%) |
| USDJPY | **−1.314** | 40.2% | 0.83 | 42.9% | 495 | +14.414 → −1.314 (−109%) |

### MTF-leak ablation (encoder OFF, isolates leak #2)
`scripts/audit_live_champions.py` — EURUSD, same WF, CatBoost:

| Config | Sharpe | Win% | Trades | Exit mix |
| ------ | ------ | ---- | ------ | -------- |
| MTF **leaky** (as shipped) | −12.11 | 11.4% | 220 | TP 1% / SL 61% / force-close 37% |
| MTF **fixed** (no look-ahead) | n/a | 33.3% | **3** | — |

Removing the MTF look-ahead collapses trades from **220 → 3**: the leak was
manufacturing false model confidence (the model only clears the 0.60 threshold
because it can see near-future price). The 11.4% win rate with encoder off is the
known anti-signal — the candle edge depends entirely on the (leaky) encoder.

### Pipeline / hybrid v2 — honest proxies (all negative)
No single clean run reproduces hybrid v2 exactly, but every leak-free analog of
its components is negative:

| Clean configuration | EURUSD | USDJPY |
| ------------------- | ------ | ------ |
| Champion-label XGBoost, no encoder, temporal-calibrated (23 folds) | −6.77 | −8.84 |
| Per-fold fresh encoder (120d) | −10.58 | −15.48 |
| Option A expanding per-fold encoder | −22.0 | −28.2 |
| Option B fixed-60% encoder | −4.40 | −5.10 |
| Pre-train + fine-tune encoder | −11.31 | −8.72 |

Hybrid v2 = XGBoost + enc8 + candle-probability injection. Both enc8 and the
candle probabilities are now shown to be leak-dependent, so the +3.01/+4.27
claims do not survive a clean test. A fully-clean hybrid reconstruction is the
one remaining confirmation, but the direction is not in doubt.

---

## What is actually real

- A **weak, regime-dependent directional signal** exists: clean champion-label
  folds range +5 to +9 Sharpe (trending) vs −12 to −15 (choppy). The model
  **overtrades in chop** — that is the core problem to solve, not "no edge."
- The win-rate/Sharpe paradox (44–46% win yet negative Sharpe) is an
  **exit-vs-label decoupling bug**: 4-bar direction label, but TP=60p needs >4
  bars while SL=30p hits fast → tiny wins, full-size losses.

## Recommendation per live service (`WIN-RESEARCH.MD` §LIVE SETUP, all demo)

| Service | Model | Action |
| ------- | ----- | ------ |
| mt5-eurusd | candle_predictor EURUSD | **Stop / do not trust** — honest Sharpe −0.84 |
| mt5-usdjpy | candle_predictor USDJPY | **Stop / do not trust** — honest Sharpe −1.31 |
| mt5-eurusd-hedge | pipeline_EURUSD v1 | **Stop / do not trust** — leak-dependent |
| mt5-usdjpy-hedge | pipeline_USDJPY v1 (v2 upgrade pending) | **Hold upgrade** — v2 gain was leakage |

All four run on a **demo** account, so there is no capital loss — but they must
not be promoted to live, and the WIN-RESEARCH champion claims should be retracted.

## Next steps (plan Phases 2–3)
1. Fix `_add_extra_features` MTF look-ahead in the production code (shift higher-TF
   series by one completed bin) before any retraining — the fix is in
   `scripts/audit_live_champions.py:_mtf_emas(fix_lookahead=True)`.
2. Expand history (Dukascopy) for a true post-2026-06-05 holdout.
3. Build the clean meta-labeling system (matched exits + vol-scaling) — the path
   to an honest, tradeable edge.
