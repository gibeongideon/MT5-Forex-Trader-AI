# Candle-Color (next-bar direction) Prediction Study — Results (2026-06-16)

Goal: predict whether the **next candle is GREEN or RED** (close[t+1] > close[t]) at higher
timeframes (H1/H4/D1) and judge it as a **tradeable edge net of real cost**, for EURUSD,
GBPUSD, USDJPY, XAUUSD. Script: `scripts/candle_color_backtest.py`.

Method (leak-free): deep Dukascopy M15 (real per-bar spread) resampled in-memory to the TF →
validated engineered feature set (`_build_X`, encoder **off**) → `TemporalCalibratedXGBoost`
(temporal-holdout isotonic calibration) → expanding walk-forward → trade 1-bar hold (enter at
bar close in predicted color when P ≥ threshold, exit next bar close) net of **real spread +
0.5pip commission** → discover/confirm split (2022-01-01) + block-bootstrap 95% CI + threshold
sweep (0.50 / 0.55 / 0.60).

**GO gate:** confirm net Sharpe ≥ +0.5 with bootstrap CI lower bound > 0 AND positive both halves.

## Verdict: NO-GO — next-candle color is not a cost-surviving tradeable edge

| TF | Instrument | Best FULL Sharpe | Best CONFIRM Sharpe | Hit % | Note |
|----|-----------|------------------|---------------------|-------|------|
| H1 | EURUSD | −8.10 (thr .55) | — | 38% | heavy spread drag, many trades |
| H4 | EURUSD | −1.84 | −3.74 | ~44% | negative all thresholds |
| H4 | GBPUSD | −5.02 | −5.03 | ~41% | negative |
| H4 | USDJPY | −0.63 | +1.68 (n=132) | ~47% | lone +ve is noise, CI [−4.4,+7.6] |
| H4 | XAUUSD | −2.50 | −1.88 | ~42% | negative |
| D1 | EURUSD | −1.18 | −0.89 | ~46-48% | negative |
| D1 | GBPUSD | −0.57 | −0.70 | ~48% | negative |
| D1 | USDJPY | −0.29 | +0.33 (n=1131) | ~50% | flat, CI [−0.56,+1.25] — not sig |
| D1 | XAUUSD | −0.24 | +0.42 (n=285) | ~50% | flat, CI [−1.25,+2.62] — not sig |

### What the numbers say
- **Hit rate is a coin flip:** ~38–50% across the board (≈50% at D1, below 50% intraday). The
  calibrated model has **no real directional skill** on next-bar color.
- **Cost dominates intraday:** H1/H4 are strongly negative because the spread is large relative
  to one-bar moves and the strategy trades constantly. The more it trades, the more it loses.
- **D1 is the least-bad** (spread is small vs a daily bar) — it converges to ~flat (Sharpe ~0,
  hit ~50%). The two "positive" confirm cells (USDJPY D1 +0.33, XAUUSD D1 +0.42) have **CIs that
  straddle zero** — they are noise, not signal, and FULL-period Sharpe is still negative.
- **Nothing clears the GO gate.** No (TF, instrument, threshold) shows confirm Sharpe ≥ +0.5
  with CI lower bound > 0. Raising the threshold (0.60) shrinks trades to a handful and widens
  CIs without producing a significant positive — i.e. no high-confidence sub-population edge.

## Conclusion
Predicting next-candle color is the **next-bar directional sign** problem in disguise, and it
fails for the same reason every prior single-instrument directional test failed: at the bar
horizon FX/Gold is ≈ a random walk, so a calibrated classifier lands at ~50% and the spread
turns that into a loss. (This is consistent with the earlier finding that the original
"candle predictor" near-100% accuracy was leakage — net of cost and leak-free it is a coin flip.)

**The project's one validated edge remains the daily CTA momentum portfolio**
(`data/CTA_FINDINGS.md`: combined + cluster-risk, net Sharpe +0.65, CI [+0.18,+1.12]). The edge
is **structural diversification at the daily horizon across a broad universe**, not per-instrument
directional/candle prediction. Directional single-name forecasting is now exhaustively dead:
M15/H1/H4/D1 direction, session-gating, intraday time-of-day, and candle-color all NO-GO.
