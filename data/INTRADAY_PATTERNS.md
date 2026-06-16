# Intraday Hour-of-Day Pattern Study — Results (2026-06-16)

Goal: find a recurring pattern at specific hours/sessions → entry, separate exit signal, for
EURUSD/GBPUSD/USDJPY/XAUUSD on deep M15 (2015–2026, UTC, real spread). Scripts:
`scripts/intraday_seasonality.py` (Phase A map), `scripts/intraday_pattern_backtest.py`
(Phase B, discover/confirm + bootstrap CI + real spread).

## Verdict: NO-GO — no cost-surviving intraday time-of-day edge

### Phase A (descriptive) found *structure* but it's sub-cost or artifact
- **Hour drift:** a few UTC hours have |t|>3 (e.g. EUR 11/20/22, XAU 22/23) but magnitudes are
  tiny (~0.5–3 bp) vs spreads (~3–14 bp). Net of cost, unprofitable.
- **Entry→exit 24×24 matrix:** the "strongest" cells for ALL instruments cluster at **20–23 UTC
  (the daily rollover)** — wide-spread, illiquid hours.
- **ORB continuation** rates look high (67–87%) but that's largely definitional drift, not net edge.

### Phase B (backtested, net of real spread, OOS) — all fail
- **Pre-registered liquid windows** (London open 07→16, NY 13→21, overlap 13→17, Tokyo 00→08),
  direction from discover: flat-to-negative on confirm AND full (best EUR London −0.06 full /
  −0.31 confirm; rest −0.7 to −2.2). None clear GO (≥+0.5, CI>0).
- **Rollover cells (20→21 UTC):** the Phase-A "edge" was a quote artifact — once real round-trip
  spread is paid it loses **both directions**, −6 to −8 Sharpe (hit 16–30%) on discover AND confirm.
- **Open-range breakout** (range→break→session-close exit): flat-to-negative net everywhere on
  full; the only +ve confirm slices (USDJPY +0.35/+0.19) are −ve on full with CIs through zero = noise.

## Conclusion
Intraday FX/Gold at M15 has **no exploitable, cost-surviving, out-of-sample time-of-day pattern**
— consistent with the prior session study and the M15/H1/H4 directional dead-ends. Apparent
intraday "patterns" are either smaller than the spread or rollover quote artifacts. The
discipline (real spread + discover/confirm + bootstrap CI + rollover audit) correctly caught them.

**The project's one validated edge remains the daily CTA momentum portfolio** (`data/CTA_FINDINGS.md`:
combined + cluster-risk, net Sharpe +0.65, CI [+0.18,+1.12]). Intraday is exhausted; the edge is
at the daily horizon across a diversified universe, not in intraday timing.
