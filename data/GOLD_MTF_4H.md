# GOLD — do 15M/30M/1H/2H features improve the 4H trend prediction? (2026-06-19)

Target = 4H turning-point/trend direction (trend-scan label). Two models per fold, side-by-side:
**4H-only** (leak-free 4H engineered features, encoder OFF) vs **4H+MTF** (same + lower-timeframe
features — ewmac/return/EMA-ratio/RSI/ATR-momentum/realized-vol from 15M, 30M, 1H, 2H, aggregated
point-in-time to the 4H grid). Reported: OOS ROC-AUC/accuracy of P(up) vs realized trend, and flip
trading (ls/ls_atr vs EWMAC + buy-and-hold). Run on all data (2015–26) and 2022–26 alone.
Code: `src/features/mtf_features.py` (causality-tested), `scripts/gold_mtf_4h.py`.

## Verdict: lower timeframes add NOTHING to the 4H trend forecast

### Prediction quality (the headline)
| period | 4H-only AUC (full/conf) | 4H+MTF AUC (full/conf) | Δ AUC |
|--------|------------------------|------------------------|-------|
| all 2015–26 | 0.503 / 0.510 | 0.503 / 0.510 | **−0.000 / −0.000** |
| 2022–26 | 0.492 / 0.436 | 0.487 / 0.431 | **−0.005 / −0.005** |

- **MTF features change the 4H forecast AUC by essentially zero (slightly negative).** 15M/30M/1H/2H
  structure does not help predict the 4H turning point.
- The 4H trend is **barely predictable at all**: AUC ≈ 0.50 (all data), ≈ 0.49 (2022–26, even mildly
  inverse out-of-sample). The ~0.57–0.60 "accuracy" in 2022–26 is just label imbalance (gold mostly
  rose), not skill.

### Trading (net Sharpe, full / confirm[CI])
| | all 2015–26 | 2022–26 |
|---|---|---|
| Buy-and-hold gold | +0.90 / +1.30 | +1.43 / **+1.58** |
| 4H-only (best ls_atr) | +0.68 / +1.05 | +1.18 / +1.47 |
| 4H+MTF (best ls_atr) | +0.64 / +1.39* | +1.33 / +1.57 |

\* 4H+MTF's higher *confirm* Sharpe comes with **negative discover** (regime-reshuffling), and AUC
shows no real skill — it's long-gold beta + noise, not turning-point alpha. **Nothing beats
buy-and-hold** (+1.30 / +1.58), and nothing clears the GO gate (positive both discover halves AND
> B&H).

## The most important takeaway: the leak guard worked
With strictly point-in-time lower→higher aggregation, MTF features added **0.000 AUC**. The original
+3.14 enc8 champion's MTF-EMA features *did* "help" — because they peeked ahead (resample lookahead).
This clean result is direct evidence that the historical MTF "contribution" was **leakage, not
signal**: done honestly, lower timeframes carry no usable information about the 4H trend direction.

## Add SL/TP to the 4H model (ATR triple-barrier, SL=1×ATR, TP 1.5/2/3×ATR, 6-bar exit)

Replacing the flip exposure with discrete SL/TP trades on **4H alone**, 4H-only vs 4H+MTF:

| period | best avg R (4H-only / 4H+MTF) | discover | reading |
|--------|-------------------------------|----------|---------|
| all 2015–26 | +0.040R / +0.050R (0.55, 1:3) | **negative** | marginal; MTF adds a noise-level +0.01R; no GO |
| 2022–26 alone | +0.153R / +0.141R (0.55, 1:3) | (in-regime only) | **looks great (Sharpe +4) but it's gold-bull beta** |

- **All data:** best expectancy +0.04–0.05R, **negative discover** on every positive-confirm cell;
  the +2.8–4.4 confirm Sharpes are √trades × rally mirages. No cell clears GO. MTF adds nothing.
- **2022–26 alone:** large positive Sharpe (+4) and avg R (+0.15R) — but 2022–26 is **entirely the
  gold bull**, so a long-biased turning-point model with a 1:3 TP just rides it up. There is no
  out-of-regime test here; the all-data run shows the *same* config is **negative in 2016–21**. So
  the 2022-only "edge" is regime/beta, not skill — exactly the cardinal-rule trap (AUC ≈ 0.49 = no
  predictive skill). MTF does not improve it (Δ avg R ≤ 0).

**SL/TP verdict:** adding SL/TP to the 4H model does NOT create an edge — same as the flip version.
Marginal/negative out-of-regime; the only "good" numbers are in-sample to the 2022–26 bull.

## Conclusion
Feeding 15M/30M/1H/2H into the 4H trend prediction does not improve it — not on all data, not in
2022–26. The 4H trend is near-unpredictable (AUC ~0.50) and the apparent gold P&L is buy-and-hold
beta. Consistent with the whole arc: single-instrument directional/turning-point prediction has no
honest edge; gold's only real exposure is holding it / sizing it in the diversified CTA basket.
