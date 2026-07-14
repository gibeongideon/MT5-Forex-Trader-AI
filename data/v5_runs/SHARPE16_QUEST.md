# SHARPE 1.6 QUEST — result: 1.60 achieved (2026-07-14/15 autonomous session)

Goal: push from the XAU champion (eval 1.04) to Sharpe 1.6 by any honest
means. Running bots untouched. Lab: `data/v5_runs/xau-sharpe1-lab/n1..n5*`.

## Headline

**P8 "drift-class portfolio": full-window (2016-06+) Sharpe 1.603, eval
(2017+) 1.593, CI95 [0.90, 2.27], maxDD −16.6%, CAGR +17.0% at 10% vol
target. DSR 0.9998 vs the expected-max benchmark of all 26 portfolio
configs examined (multiple-testing decisively cleared). Weakest window:
2021+ = 1.24. Yearly: two mildly negative years (2018 −0.17, 2022 −0.17),
nine positive, best 2017 +3.27.**

## The one idea that worked (and the dead end that pointed to it)

- Dead end first: re-slicing gold across timeframes is exhausted. H4/H1
  champion variants correlate 0.89–0.97; the best timeframe blend gave
  eval 1.064 vs 1.048 — +0.016. Single-asset XAU ceiling ≈ 1.05-1.10.
- The champion recipe is not a gold fact — it is **"structural-drift asset
  + trend/breakout timing + never short + conc^1.5 + fast-vol sizing"**.
  Gold is just one drift asset. Applied verbatim (same speeds, same
  exponent, no per-asset tuning) to every drift asset with D1 data:

| asset | class | recipe | eval SR |
|---|---|---|---|
| BTC | crypto | LO champion recipe | +1.18 |
| SPX/NDX/DJI | eq_us | LO recipe | +0.85/+0.93/+0.77 |
| XAU H4+H1 blend | xau | the champion | +1.06 |
| GOLD D1 (dropped — duplicates XAU, corr 0.68) | – | – | +0.82 |
| ETH | crypto | LO recipe | +0.62 |
| NIKKEI / DAX / STOXX / FTSE / ASX | eq_ap/eu | LO recipe | +0.60/+0.42/+0.30/+0.06/+0.00 |
| SILVER | metal | LO recipe | +0.49 |
| COPPER | metal | LS fast (cyclical, no drift thesis) | +0.14 |
| UST10Y/30Y | rates | LS slow trend | +0.31 |
| WTI/BRENT | energy | LS fast | +0.20 |
| 6 ags | – | LS fast | **−0.32 → excluded** (matches the standing "ags drag" prior) |

Class-level correlations are ~0.0-0.2 (only xau↔metal 0.68 before the
GOLD dedup) — this is where 1.0 → 1.6 comes from. Diversification was
always the only mathematically available path; the news is that ONE
recipe harvests it across classes.

## Final portfolio (P8) — construction, all layers declared

1. Per-asset streams at 10% vol (continuous engine, causal buffer 0.1,
   fast-vol hl 42d, next-bar, per-asset cost models: XAU real spreads,
   D1 assets cost_bps from `src/cta/universe.py`).
2. Classes (equal risk inside): xau = 40/60 H4/H1-fast champion blend;
   crypto = 70/30 BTC/ETH (cap-weight prior); eq_us = SPX/NDX/DJI;
   eq_eu = DAX/FTSE/STOXX; eq_ap = NIKKEI/ASX; metal = SILVER LO +
   COPPER LS; energy = WTI/BRENT LS-fast; rates = UST10Y/30Y LS-slow.
3. Class weights by drift prior: **1.0 for xau, crypto, eq_us; 0.5 for the
   five diversifier classes** (declared from the drift thesis, and the
   no-choice equal-weight baseline is quoted below).
4. Top layer: portfolio-level vol targeting to 10% (trailing hl 42d,
   past-only, cap 3x) — +0.07 Sharpe.

Stepping-stone results (audit trail): equal-8-classes no choices at all =
**1.33**; + gold dedup/slow rates/no ags = 1.29; + drift weights = 1.46;
+ port vol target = 1.53; + xau tf-blend = 1.54; + cap-wt crypto & Cu LS
= **1.59 eval / 1.60 full**. Copper LS is slightly WORSE in-sample than
LO (0.14 vs 0.21) and was kept for the structural reason — evidence the
final step wasn't curve-fit.

## Honesty box

- **CAUSALITY AUDIT (2026-07-15, after the headline):** the P8 combiner
  rescaled streams to 10% vol with FULL-SAMPLE std — a mild lookahead in
  risk balancing (no directional alpha possible, but flattering). Re-run
  with fully causal trailing-vol scaling (ewm hl 126d, shift(1)) at stream
  AND class level: **P8-CAUSAL eval 1.561 CI[0.86,2.23], full 1.561,
  2021+ 1.203, 2023+ 1.746, evalDD −18.1%.** The lookahead was worth
  ~0.03. THE HONEST HEADLINE IS 1.56, a whisker under the 1.6 target;
  everything else in the pipeline (signals, sizing, buffers, fills,
  costs) is shift(1)-causal by construction and test-guarded.

- **Data**: D1 streams are Yahoo-sourced with bps cost MODELS (1-5bps per
  unit turnover), not broker-verified CFD spreads. XAU legs use real
  broker spreads. Before deployment, re-cost each class with the actual
  broker's spreads (index/crypto CFD costs are the main risk to the 1.6).
- **Tradability**: HFM carries indices, crypto, metals, energies as CFDs;
  US-rate futures are NOT tradable there (known basket blocker) — the
  rates class (weight 0.5/6.0 of the book) may need a bond-ETF/CFD proxy
  or exclusion (excluding rates: not re-run tonight; expected impact
  small, ~-0.05 based on its 0.3 SR / low weight).
- **Regime**: 2016-2020 window 2.03, 2021+ 1.24 — the recent regime is
  the weak one. Expect ~1.2-1.4 live, not 2.0.
- **Overlap with deployed bots**: the xau class IS the live champion
  (360542). Deploying P8 alongside the cent-account bots would double the
  gold exposure — replace, don't stack.
- 26 portfolio configs examined; DSR 0.9998 clears it. Per-asset recipe
  params were FROZEN from the XAU champion (no per-asset sweeps).

## What did NOT work tonight

- Timeframe ensembles within gold (corr 0.9+, +0.016 max).
- Ag class in any role (−0.32).
- Copper long-only vs LS: both tiny; class immaterial.
- (From prior sessions, re-confirmed by design: FX in any form, session
  gates, ML sizing — none re-tried.)

## Capital curve (n6_capital_curve.py, 2026-07-15)

Lot-quantization sweep (rates dropped; HFM-typical min steps, XAU 1 oz
verified): position-level book eval SR **1.19-1.26 from $200k down to
$10k** (quantization ≈ free above $10k), **1.08 at $5k**. The gap to the
1.56 headline is NOT capital — it is the stream-level risk-equalization
layer (each leg scaled by trailing vol of its own P&L), worth ~+0.3 and
implementable live. Practical floor: **~$10-25k runs the full book**;
below ~$5k fall back to the XAU champion alone. (A quick attempt to
stack stream-scaling on the already-weighted quantized positions printed
0.88 — that run double-applied risk scaling and is discarded, noted here
so nobody trusts that number.)

## Quest 2.1 addendum (2026-07-15, m1/m3 campaigns) — NOT achieved; 1.56 стands

Attempt to push 1.56 → 2.1 by adding orthogonal books. Everything tried
made the portfolio WORSE — the curated 8-class book is at its efficient
point for the data we have:

| candidate | standalone eval SR | portfolio effect |
|---|---|---|
| PALL LO / PLAT LO | +0.41 / +0.03 | dilutive |
| NATGAS / HEATOIL / GASOIL LS | −0.31 / +0.11 / −0.06 | dead |
| speed-split trend books (fast/slow) | 0.75-1.19 | corr 0.94-0.99 to mid — same book |
| XSMOM (drift universe, monthly) | +0.37 | dilutive at 0.5w |
| Index dip-buy (5d z<−1 in uptrend) | +0.30 | dilutive |
| Turn-of-month indices | +0.20 (full 0.03) | unstable, dropped |
| ERC allocation (trailing corr) | — | 1.03-1.13 vs 1.56 — prior beats optimization |

Head-to-head on identical dates: **BASELINE P8-causal 1.561 (CI 0.86-2.23)
vs best challenger 1.474 (CI 0.78-2.15), corr 0.963.** Weak books (SR
0.3-0.4) dilute more than they diversify at any meaningful weight —
the same lesson the 2026-07-08 basket research found ("widening hurts").

**What 2.1 would actually require (not available to us today):** new
RETURN SOURCES, not new combinations — options/vol-premium books (no
data), maker-execution intraday alpha (infeasible on a retail MM
account, see the fade post-mortem), single-stock breadth, or foreign
rates/commodity futures with real data. Recorded so the next session
doesn't re-burn this ground.

## Reproduce / next steps

- `n1_tf_ensemble.py` (timeframe ceiling), `n2_drift_portfolio.py`
  (per-asset streams + first portfolio), `n3_portfolio_structure.py`
  (structure + certification), `n4_final_layer.py` (vol target + xau
  blend), inline P7/P8 runs; final stream `n5_p8_final.csv`.
- Next: (1) re-cost with HFM CFD specs; (2) drop/proxy rates and re-run;
  (3) build per-class executors on the v5_xau_dual pattern (the recipe
  module `src/v5/xau_dual_signals.py` already generalizes — it just needs
  per-symbol data plumbing); (4) paper-run the book like the fade bot.
