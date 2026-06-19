# GOLD (XAUUSD) Trend-Direction / Turning-Point Predictor — Results (2026-06-19)

Goal: classify gold's trend direction (up/down), enter at the turning point, ride it, flip when
the opposite direction is predicted past a threshold. Honest, leak-free: encoder OFF, features
past-only (`_build_X`), forward-looking labels used as TARGETS only. H4, expanding WF (540/90/90),
`TemporalCalibratedXGBoost` → P(up), 3 exposure modes × 3 thresholds, vs **two benchmarks on the
same OOS window/cost**: mechanical EWMAC trend and vol-targeted **buy-and-hold gold**.
Script: `scripts/gold_trend_predictor.py`; labels: `src/features/trend_labels.py`.
GO gate (pre-registered): confirm Sharpe ≥ +0.5, CI lower > 0, positive both discover halves,
**AND beats both EWMAC and buy-and-hold**. OOS 2016-06 → 2026-05 (15,709 H4 bars).

## Verdict: NO genuine turning-point alpha over buy-and-hold gold

The model is a *legitimate trend-follower* — it beats mechanical EWMAC and posts a positive,
significant OOS Sharpe — but it **does not beat simply holding gold**, which rode the 2022–26 bull.

### Benchmarks (the bar to beat)
| | FULL | discover | confirm (CI) | maxDD | turn |
|---|---|---|---|---|---|
| EWMAC trend | +0.36 | −0.10 | +0.76 [−0.19,+1.78] | 5% | 9/yr |
| **Buy-and-hold gold** | **+0.90** | +0.49 | **+1.30 [+0.34,+2.33]** | 25% | 0 |

### Model — trend_scan labels (actively trades)
| mode/thr | FULL | confirm (CI) | win | DD | turn | vs bench |
|---|---|---|---|---|---|---|
| ls 0.55 | +0.58 | +0.97 [−0.02,+2.02] | 51% | 29% | 8/yr | >EWMAC <B&H |
| **ls_atr 0.55** | **+0.68** | **+1.05 [+0.05,+2.09]** | 50% | 25% | 13/yr | >EWMAC <B&H |
| lfs (flat state) | ≤+0.39 | ≤+0.61 | 3–28% | — | high | <EWMAC <B&H |

→ Best active config **ls_atr@0.55: confirm +1.05, CI lower +0.05 > 0, beats EWMAC** — the
strongest *genuinely-trading* single-instrument ML result in the project. **But < buy-and-hold
(+1.30).** lfs (long/flat/short) is poor: sitting flat misses the trend.

### Model — zigzag labels (the "✅GO" is a buy-and-hold mirage)
`ls@0.65` printed conf **+1.32 [+0.36,+2.34]**, FULL +0.96 — nominally beating B&H and tripping
the GO flag. **Audit: turnover = 2/yr** — it stopped flipping and converged to *being long gold
almost the whole time*. Its confirm +1.32 ≈ B&H +1.30 (identical CI, identical 25% DD) → it is
**buy-and-hold in disguise**, not reversal-timing. The +0.02 over B&H is noise. Cardinal rule
caught it: the gate was gamed by degenerating into the long-gold beta it was meant to beat.

## Conclusion
On gold H4, an ML turning-point model **adds value over mechanical EWMAC trend** (ls_atr +1.05 vs
+0.76, CI>0) — the first single-instrument ML signal to clear a real benchmark. But it **cannot
beat passively holding gold**: either it trades and underperforms B&H (trend_scan), or it stops
trading and *becomes* B&H (zigzag). Same lesson as the single-instrument study — **single-asset
gold is dominated by bull-market beta**; reversal-timing alpha isn't enough to beat owning the
trend. To beat B&H you must avoid gold's drawdowns or short profitably, and the model does neither
robustly (discover-half Sharpes are weak; the edge is concentrated in the 2022–26 rally).

**Deployable takeaway:** if you want gold exposure, the honest options are (a) just hold it / size
it in the diversified CTA basket (where gold is one of 5 legs), or (b) ls_atr@0.55 as an *active*
trend-follower that beats EWMAC — but accept it trails buy-and-hold. No standalone gold ML edge
over B&H. (`ls_atr` = long↔short + 3×ATR stop; trend_scan = López de Prado trend-scanning labels.)
