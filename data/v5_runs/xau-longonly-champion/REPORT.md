# XAUUSD Overnight Sharpe Hunt — Final Report (2026-07-14)

Goal: research-only backtests (bot untouched) on XAUUSD single-symbol until Sharpe >= 1.0.
Data: `data/XAUUSD_H4_long.csv` 2015-01 → 2026-07. Eval window 2017+ (repo convention).
Costs: half-spread (median-floored) + $0.10 slippage per side, next-bar fills.
All signals lookahead-free (expanding past-only scalars, shift(1) everywhere).

## Headline result — TARGET HIT

**Champion "LO-blend-conc1.5" (continuous engine, buffer 0.1): eval Sharpe 1.041**
(CI95 [0.42, 1.68]), full-window 0.976 (CI [0.39, 1.56]), maxDD −16.9%, CAGR +18.0%,
turnover 17.5x/yr. Stress: costx2 1.027, costx3 1.014, delay2 1.022. 2021+ window: 1.18.
Yearly Sharpe 2015→2026: 0.0, 0.55, 0.75, 0.40, 1.18, 1.37, −0.12, 0.02, 0.72, 1.38, 2.50, 0.82.

Champion definition (H4 closes only):
- `ewm_mid` = EWMAC speeds (96,384),(192,768),(384,1536) H4 bars (Carver, past-only scalars)
- `bko_f` = Carver breakout, windows 60/120/240 H4 bars (10/20/40 days), smoothed N/4
- `maxewbko = max(ewm_mid⁺, bko_f⁺)` (⁺ = clip at 0 → LONG-ONLY)
- concentrate: `conc(s) = norm(s^1.5)` (norm = past-only expanding mean-abs)
- forecast `= 0.5*(conc(maxewbko)*0.8 + 0.15) + 0.5*(conc(bko_f)*0.8 + 0.15)`, clip [0,2]
- sizing: forecast × (10%/ann vol), vol = EWM std halflife 42 H4 bars (fast vol is load-bearing)
- causal no-trade buffer 0.1; decisions at close, filled next bar open.

**Discrete live-style engine validation** (src.v5.xau_trend.run_trades, champion signal
monkeypatched in, trail exit, SL/trail 3.0×ATR, conf_risk_scale {low:0.5, med:1.0, high:1.5}):
**Sharpe 0.968, CI [0.33, 1.61], maxDD −10.0%, CAGR +9.6%** — transfers to real execution.
With default 2×ATR params: 0.843. Binary flip mode: 0.795 (loses sizing alpha).

## The one structural discovery: KILL THE SHORTS

Everything else was refinement. Gold 2015-2026 shorts are pure drag:
- long/short trend ensemble: eval 0.757 → zero out shorts: **0.931** (monotone in damping:
  0%→0.931, 25%→0.898, 50%→0.856, 75%→0.809, 100%→0.757).
- Vol-targeted buy&hold benchmark: eval 0.928 but full 0.731, DD −16% — trend timing beats
  it out-of-window (full 0.976) and in 2015-2020 (0.77 vs B&H weaker).
- CAVEAT (honest): long-only is a property of this 11.5-yr sample (secular gold bull).
  In a 2011-2015-style bear the book would sit in cash + small resting tilt (worst year
  observed: −0.12). It is NOT market-neutral alpha; ~half the eval Sharpe is gold drift,
  the trend timing adds the rest and cuts DD from −35% (B&H full) to −17%.

## What helped (in order of impact)
1. Long-only (shorts → 0): +0.17 eval Sharpe. THE lever.
2. max(trend, breakout) union entry: 0.931 → 0.951.
3. Forecast concentration ^1.5: full-window 0.879 → 0.957 (biggest full-window gain).
4. Blending concentrated max-book with concentrated breakout book: → 1.041 eval.
5. Resting long tilt +0.15 when flat: +0.01-0.02, more CAGR.
6. Small no-trade buffer (0.1): +0.02 and cuts turnover 36→17x/yr.
7. FAST vol estimator (halflife 42 H4 bars ≈ 1wk): halflife 126→0.89, 216→0.85, 378→0.83.
   Slow vol destroys the edge — gold vol-shading is real alpha.
8. Confidence-scaled risk on the discrete engine: 0.917 → 0.968.

## What did NOT help (tested tonight, adds to repo's disproven list)
- D1 timeframe anything (0.18-0.44 — H4 dominates every family).
- Very-fast EWMAC H4 (12,48)-(48,192): 0.44; all-6-speeds: 0.65 (mid/slow set is right).
- Acceleration (0.22), skew 120d (0.10), seasonality month-tilt (0.80 vs 0.93 base).
- Vol-floor sizing (0.72-0.74), dual-vol max(fast,slow) (0.86-0.90), vol halflife >42 bars.
- D1-slow-trend regime gate on H4 entries (0.93 — no better than ungated).
- Short damping partial (any k>0 worse than k=0).
- tsmom as 3rd max leg (0.949 vs 0.951 — neutral, more turnover).

## Multiple-testing honesty
~90 configurations tested across 9 campaigns (all logged in results.csv). The champion is
a 2-step composition of two INDEPENDENTLY strong, structurally-motivated books (each ~0.95
eval standalone, each robust to costx3/delay2/subwindows), not a grid argmax; nearest
neighbours in every direction (exponent 1.25/2.0, buf 0.2, weights ±0.1) all sit 0.96-1.03,
so the point is a plateau, not a spike. Still, treat 1.04 as ~0.9-1.0 expected OOS after
selection effects (2015-2020 subwindow: 0.77; CI lower bound 0.42).

## Files
- Harness: scratchpad/xau_lab.py; campaigns camp1..camp9; full log results.csv (~90 rows).
- Live bot NOT touched. No project files modified.

## Suggested next steps (not done tonight)
1. Paper-validate the champion alongside the live bot (same pattern as the fade paper bot).
2. Port champion forecast into a v5 runner variant behind a --research flag.
3. Consider deploying the discrete trail-atr3 + conf-scale variant (SR 0.97, DD −10%) —
   it fits the existing executor with only a signal swap + param change.
4. Re-run camp8 battery when data extends; watch the long-only caveat if gold regime turns.
