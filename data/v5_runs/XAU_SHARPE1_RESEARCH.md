# XAUUSD Sharpe ≥ 1 Overnight Research — 2026-07-14

Goal: improve the single-symbol XAUUSD trend bot's net Sharpe to ≥ 1.0 via
backtest research only (live bot untouched). Evaluation matches the repo
convention: daily-resampled equity, ×√252, primary window 2017+, costs =
time-varying spread column + 1 pip slippage, next-bar execution.

## Baseline (current live engine, freshest data)

`scripts/v5_xau_backtest.py` grid: trail/confidence **0.37**, flip/always
**0.47**, DD 32–43%. (Lower than the remembered 0.63 — data now runs to
2026-07-13.)

## Result: target hit — TWO independent configs ≥ 1.0

### PRIMARY CHAMPION — "H4 long-flat blend-conc1.5" (best overall)

Forecast = 0.5·[conc(max(EWMAC-mid⁺, BKO⁺), 1.5)·0.8 + 0.15] +
0.5·[conc(BKO⁺, 1.5)·0.8 + 0.15], clipped [0,2], where ⁺ = clip at 0
(long-flat), BKO = Carver breakout 10/20/40d on H4, EWMAC-mid =
(16,64),(32,128),(64,256) daily-equivalent, conc(s,p) = renormalized s^p
(concentrates size into strong signals), +0.15 = small permanent long floor.
Vol-target 10% (hl 42d), buffer 0.1, next-bar execution, full costs.
(Built in an earlier segment of tonight's session; battery re-verified.)

| window | Sharpe | CI95 |
|---|---|---|
| 2017+ (primary) | **+1.041** | [+0.42, +1.68] |
| full 2015+ | +0.976 | [+0.39, +1.56] |
| 2021+ | +1.180 | [+0.37, +2.04] |
| 2015–2020 | +0.765 | [−0.09, +1.59] |

CAGR 18.0%, maxDD 16.9%, turnover 17.5/yr. Stress: costx2 = 1.027,
costx3 = 1.014, delay2 = 1.022. Yearly Sharpe: worst year −0.12 (2021);
every other year ≥ 0 — far more consistent than any other config tested
(2016 0.55, 2017 0.75, 2018 0.40, 2019 1.18, 2020 1.37, 2021 −0.12,
2022 0.02, 2023 0.72, 2024 1.38, 2025 2.50, 2026 0.82).

### RUNNER-UP CHAMPION — "H1 long-flat trend ensemble" (simplest recipe)

Independently reached ≥1.0 with a plainer construction (no power
transform, no floor), all costs included:

| window | Sharpe | CI95 (block bootstrap) |
|---|---|---|
| 2017+ (primary) | **+1.017** | [+0.39, +1.66] |
| full 2015+ | +0.933 | [+0.35, +1.51] |
| 2021+ | +1.310 | [+0.54, +2.15] |
| 2023+ | +1.837 | [+0.82, +2.77] |

CAGR 14.4% at 10% vol target, maxDD 18.9%, turnover 22/yr, avg leverage
0.55, in market 91% of the time, long-only exposure.

### Recipe

- Data: XAUUSD H1 bars.
- Forecast = 0.6·EWMAC(fast: daily-span (4,16),(8,32),(16,64) sampled on
  H1) + 0.2·EWMAC(mid: (16,64),(32,128),(64,256) daily) +
  0.2·Breakout(Carver channel, 10/20/40 days), each scaled to |1|≈avg by
  causal expanding scalar, clipped ±2.
- **Long-flat: negative forecasts → 0 (no shorts).**
- Sizing: vol-target 10% ann. (EWM std halflife 21 days), position =
  fc × (0.10/vol), capped 8×.
- No-trade buffer: 0.35 × average-size band (turnover 22/yr).
- Execution: decide on close, trade next bar open, cost = half-spread+$0.10.

### Robustness battery (eval Sharpe)

- costs ×2 = 0.994; ×3 = 0.971 (buffering makes it cost-insensitive)
- execution delay +1 bar = 1.037; +2 bars = 1.019
- vol-target 15% = 1.019 (DD 27%); buffer 0.25 = 1.003; vol-HL 42d = 0.997
- plateau, not spike: every weight variant tested (w532/w5221/w4222…)
  scored eval 0.98–1.02, full 0.88–0.93
- 2008+ D1 check (incl. 2011–15 bear): long-flat 0.47 full vs long-short
  0.34; in the 2011-09..2015-12 bear window long-flat lost −14% vs −29%
  buy-hold (it de-risks, doesn't die)
- Deflated Sharpe (192 configs tried tonight, per-day units): DSR = 0.91
  vs expected-max benchmark 0.58 ann. Caveat: trials are highly correlated
  variants, so true deflation is milder; conventional 0.95 bar not met on
  the raw number — flagged honestly.
- Yearly: 2016 +0.55, 2017 −0.24, 2018 −0.08, 2019 +0.66, 2020 +1.39,
  2021 −0.99, 2022 +0.58, 2023 +1.21, 2024 +1.30, 2025 +3.09, 2026 +0.92.
  The edge concentrates in gold bull years; 2017/18/21 were flat-to-negative.

### Why it works (and the honest caveat)

Three compounding effects, in order of importance:
1. **No shorts.** Long-short control of the same book: 0.64. Shorting gold
   2015–2026 was pure drag (secular bull + positive drift). This is the
   single biggest lever found tonight (+0.2–0.3 Sharpe).
2. **H1 sampling of daily-speed signals** beats H4/D1 sampling of the same
   speeds (faster vol/trend response, same turnover after buffering):
   H4 best 0.76 → H1 0.99 for the identical ensemble.
3. **Signal-family diversification** (fast+mid EWMAC + breakout): +0.03–0.05
   and better full-window/DD than any single family.

Caveat: a long-flat gold book is structurally a *bull-regime harvester*.
It survived 2011–15 (D1 check) by going flat, but its Sharpe premium over
the long-short book exists only while gold's drift is positive. If gold
enters a multi-year bear, expect ~0 (flat) rather than negative — that is
the acceptable failure mode by design.

## Live-bot-compatible discrete version

Binary hysteresis long-flat (BUY when forecast ≥ 0.5, close when ≤ 0.0,
vol-targeted size at entry, buffer 0.5): eval **0.84–0.86**, full 0.78,
maxDD **10%**, turnover 1.7–1.9/yr, essentially cost-immune (costx2 −0.004).
Roughly **2× the current bot's Sharpe at one-third of its drawdown**, in
the exact BUY/close framework the live engine already uses.

## What failed tonight (recorded so we don't retry)

- D1 GOLD (2008+) versions of everything: 0.2–0.55 (H1 sampling is the
  right lattice; 2008–14 also genuinely harder for trend).
- Acceleration signal 0.22; skew signal 0.10; tsmom alone 0.60.
- Overnight-drift overlay: hours 20:00/00:00 carry gold's drift (t≈3.6–3.9,
  holds OOS 2020+), but harvesting it costs 413 turnover/yr → net 0.37
  alone and dilutive as a tilt. Same spread-vs-edge wall as the dead fade.
- Seasonality tilt (Dec–Feb/Aug–Sep): 0.80 < base.
- Vol-floor sizing: 0.71–0.74 < base.
- Long tilt / exposure floor on H4: helped (0.97) but inferior to clean
  long-flat H1 and adds permanent-long risk.

## Files

- Lab + campaign scripts + full 192-row log: session scratchpad
  (`xau_lab.py`, `camp1..10*.py`, `results.csv`); copies below.
- No bot/source files were modified; live bot untouched.

## Recommended next step

1. Primary: paper-trade the **H4 blend-conc1.5** book (VIRTUAL pattern,
   like the fade paper bot) — H4 bars mean it drops straight into the
   existing runner cadence; 17 position-adjustments/yr.
2. Alternative (simplest to implement in the live BUY/close engine): the
   discrete hysteresis long-flat H1 book — BUY when ensemble forecast
   ≥ 0.5, close when ≤ 0, size = 10% vol target (eval 0.85, DD 10%,
   ~2 trades/yr, cost-immune).
3. Both champions are LONG-FLAT. Keep the current long-short book off or
   demoted — shorting gold has cost ~0.3 Sharpe every way it was tested.
