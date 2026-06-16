# Single-Instrument Simple-Strategy Study — Results (2026-06-16)

Goal: a **simple, single-instrument** edge using a **different problem formulation** than the
(dead) directional-prediction class, for EURUSD, GBPUSD, USDJPY, XAUUSD. Five formulations,
deep Dukascopy M15→D1 (real per-bar spread, 2015–2026), vol-targeted to 15%, net of real cost,
discover(2015–21)/confirm(2022–26) + block-bootstrap 95% CI + GO gate. **Textbook params only**
(no in-sample tuning). Script: `scripts/single_instrument_strategies.py`.

**GO gate:** confirm net Sharpe ≥ +0.5 AND bootstrap CI lower bound > 0 AND discover Sharpe > 0.

## Verdict: NO-GO — no genuine single-instrument timing/structural edge

Two cells tripped the GO gate (XAUUSD trend, USDJPY carry+filter) — but the **cardinal-rule
audit shows both are disguised buy-and-hold of an asset that rallied hard in 2022–26**, not
timing skill. They do **not beat (gold) or even add anything over (USDJPY) simply holding the
asset** vol-targeted, and both were flat in the discover regime. No deployable edge.

### Full matrix (annualized net Sharpe)

| Instrument | 1.TREND (ewmac) | 3.BREAK (Donchian) | 4.CARRY (rate-diff) | 4b.CARRY+trendfilt | 5.REVERT (z-score) |
|-----------|----------------|--------------------|--------------------|--------------------|--------------------|
| EURUSD | −0.05 / −0.06 | −0.49 / −0.23 | +0.22 / +0.27 | +0.30 / +0.26 | +0.26 / **−0.20** |
| GBPUSD | −0.08 / −0.03 | −0.36 / +0.24 | −0.32 / −0.60 | −0.19 / −0.34 | +0.44 / **+0.14** |
| USDJPY | +0.08 / +0.44 | +0.11 / +0.51 | +0.42 / +1.22 | **+0.57 / +1.12 ✅** | −0.14 / −0.49 |
| XAUUSD | **+0.64 / +1.22 ✅** | +0.12 / +0.23 | — (no yield) | — | −0.41 / −0.65 |

*(cells show full / confirm Sharpe; ✅ = passed GO gate before audit)*

### The audit that kills both GOs (vol-targeted buy-and-hold benchmark)

| Strategy | strat Sharpe (full/conf) | **buy&hold Sharpe (full/conf)** | corr to B&H | position mix |
|----------|--------------------------|---------------------------------|-------------|--------------|
| XAUUSD trend | +0.64 / +1.22 | **+0.82 / +1.37** | +0.55 | long 68% / short 21% / flat 11% |
| USDJPY carry+tf | +0.57 / +1.12 | **+0.57 / +1.22** | +0.75 | long 55% / short 0% / flat 45% |

- **Gold:** plain long-gold beats the trend strategy — the timing *subtracts* value. The result
  is gold's bull market, not skill.
- **USDJPY:** the carry strategy is statistically *identical* to long-USDJPY. Carry accrual and
  the trend filter add nothing over riding the 2022–26 USD/JPY rally; it is never short.
- Both confirm-Sharpes are the **2022–26 macro regime** (USD strength, gold bull). Discover-period
  Sharpes were ~flat (+0.15, +0.09) — there was no such trend then, so there was no "edge" then.

## What IS real (but not a standalone trade)
- **Volatility is genuinely predictable.** Vol-forecast IC (next-day |return|, Spearman) =
  **+0.16 to +0.25** on all four — strongly positive, unlike direction. But conditioning trend
  on it added little (gold +0.48→+0.64; majors ≈ neutral). Useful for *sizing/risk*, not as an
  entry signal.

## What failed outright (NO-GO, leak-free)
- **Breakout (Donchian 20/10):** flat-to-negative, ~60% max DD, high turnover. Worse than EWMAC's
  continuous low-turnover trend.
- **Mean-reversion (z-score):** the classic in-sample mirage — positive discover (+0.58/+0.63),
  flat-to-negative confirm. NO-GO.
- **Single-asset trend on EUR/GBP:** flat (~0). Single-asset trend only "works" on an instrument
  that happens to be trending (gold/JPY) — and then it's just beta to that trend.

## Conclusion
Net of cost and audited against buy-and-hold, **there is no single-instrument timing or carry
edge that adds value** for FX majors or Gold. The apparent winners are directional exposure to
assets that rallied — they don't beat holding the asset and had no edge in the prior regime.

This is exactly *why the diversified CTA portfolio works and a single instrument does not*:
per-asset trend is weak and regime-dependent (≈ beta), but **diversifying weak, low-correlation
trend signals across ~48 instruments** (√N) is what converts it into the significant, ~0-beta
portfolio Sharpe of **+0.65, CI [+0.18,+1.12]** (`data/CTA_FINDINGS.md`). The edge is structural
diversification at the daily horizon — not anything a single instrument can deliver simply.

**Recommendation:** the diversified daily CTA portfolio remains the only validated edge. If a
"simple" path is wanted, the closest honest version is a *small* multi-instrument trend basket
(e.g. Gold + a few low-correlation futures) vol-targeted — not any single name. Volatility
forecasting (+0.16..+0.25 IC) is worth keeping as a sizing/risk overlay, not a signal.
