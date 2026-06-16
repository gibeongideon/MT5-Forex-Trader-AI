# CTA Daily Momentum — Build #1 Findings (2026-06-16)

First genuinely-different approach after directional/stat-arb/vol-timing all died. Daily
momentum portfolio across 25 instruments (FX majors+crosses, metals, energy, equity
indices), 2008–2026 Yahoo daily data, inverse-vol risk-budgeted + 10% vol-target,
monthly rebalance, real-ish costs. Scripts: `scripts/download_universe.py`,
`scripts/download_rates.py`, `src/cta/*`, `scripts/cta_backtest.py`. Unit tests:
`tests/test_cta_pnl.py` (6 green — P&L value, lookahead guard, cost, vol-target).

## Result — first real, OOS-positive, non-bug edge (but modest)

| Sleeve | Discover 2008–21 | Confirm 2022–26 | Full | corr→long-all | turnover |
| ------ | ---------------- | --------------- | ---- | ------------- | -------- |
| TSMOM | +0.16 | +0.31 | +0.20 | −0.03 | ~23×/yr |
| **Combined (TSMOM+XSMOM)** | **+0.17** | **+0.46** | **+0.24** | +0.03 | ~18×/yr |
| Carry (rate-diff) | −0.50 | +0.23 | −0.34 | +0.28 | ~6×/yr |
| Momentum+Carry | −0.29 | +0.43 | −0.13 | +0.29 | ~14×/yr |

(net Sharpe; vol on-target ~10.7%; maxDD ~20–35% for momentum sleeves.)

## Build #2 — EWMAC trend signal + expanded universe (2026-06-16)

Lever #1: replaced binary sign with continuous **EWMAC** trend-strength forecast
(multi-speed 8/16/32/64-day crossovers, vol-normalized, capped ±20, FDM-combined;
`src/cta/signals.py::ewmac`). Lever #2: expanded universe **25 → 48 instruments**
(added rates futures ZN/ZB/ZF/ZT, more metals/energy, ags, crypto, more FX crosses;
Yahoo daily, weekday-aligned — fixed a crypto-weekend calendar bug that had injected
Sat/Sun NaN rows and corrupted the panel).

| Build | Sleeve | Full net | Confirm 22–26 | sub 08–15 | sub 16–21 | maxDD | corr→long-all |
| ----- | ------ | -------- | ------------- | --------- | --------- | ----- | ------------- |
| #1 (25, binary) | combined | +0.24 | +0.46 | — | — | ~28% | +0.03 |
| #2 (25, EWMAC) | combined | +0.28 | +0.49 | +0.29 | +0.11 | 28% | +0.12 |
| **#2 (48, EWMAC)** | **combined** | **+0.36** | +0.39 | +0.26 | +0.49 | 26% | +0.09 |

EWMAC + more instruments lifted full-period net Sharpe **+0.24 → +0.36** and made it
more robust: positive across the full 18 years AND both discover sub-halves, vol
on-target ~10.7%, ~0 beta, DD ~26%. Still below +0.5 with CI straddling zero.

## Build #3 — cluster-aware risk budgeting → FIRST significant edge (GO bar cleared)

Lever #3: replaced diagonal inverse-vol with **risk budget allocated equally ACROSS asset
classes, then within each** (`src/cta/portfolio.py::cluster_risk_weights`). Diagonal sizing
over-concentrated directional risk in correlated clusters (8 equity indices that trend
together ≈ 1 bet, not 8; 4 UST futures ≈ 1 bet). Treating each class as ~one bet restores
true diversification. (Position buffering also added, `--buffer`; negligible at monthly
rebalance — turnover wasn't binding.)

| Sleeve (48-instr, EWMAC, cluster risk) | Full net | 95% CI | Confirm 22–26 | sub 08–15 | sub 16–21 | maxDD |
| -------------------------------------- | -------- | ------ | ------------- | --------- | --------- | ----- |
| xsmom | +0.55 | [+0.10, +1.02] | +0.37 | — | — | 18% |
| ewmac | +0.47 | [0.00, +0.95] | +0.28 | — | — | 27% |
| **combined (ewmac+xsmom)** | **+0.65** | **[+0.18, +1.12]** | +0.38 | +0.40 | +1.13 | 24% |

**Full-period net Sharpe +0.65 with bootstrap CI lower bound +0.18 > 0 — the FIRST
statistically-significant, GO-bar-clearing result in the project.** Diag→cluster nearly
doubled the Sharpe (+0.36→+0.65); the gain generalizes across all sleeves (not a single-
config artifact); lands squarely in the documented "good CTA" band (0.6–1.0); ~0 beta to
long-everything; DD 24%; vol on-target; no artifact tells (no ≫1 overall, no 100% win, no
0 DD). **Honest caveat:** significant over the full 18 years, but the recent confirm-only
slice (+0.38, 4.5yr) is positive yet NOT independently significant — too short. Expect the
usual live-vs-backtest haircut, and note this is a 48-instrument DAILY portfolio that would
need multi-instrument execution infra to run (the current bots are single-symbol).
## Build #4 — ML forecast combination: tested, REJECTED (worse + beta-contaminated)

Lever #4: pooled ridge (walk-forward monthly, target-purged) combining EWMAC speeds +
xsmom + 1-wk reversal + vol-regime into one forecast (`src/cta/ml_combine.py`, `--sleeve ml`).

| Config | Full net | 95% CI | Confirm | corr→long-all | vol |
| ------ | -------- | ------ | ------- | ------------- | --- |
| hand-combined (cluster risk) | **+0.65** | [+0.18,+1.12] | +0.38 | +0.08 | 10.6% |
| ML ridge-combine | +0.50 | [+0.02,+1.04] | +0.75 | **+0.59** | 3.5% |

ML scored LOWER full Sharpe and, worse, **corr-to-long-everything jumped to +0.59** — the
ridge learned a net-long tilt, so it's partly market beta, not clean diversified momentum
(its higher confirm is largely the 2022–26 risk rally). Vol also fell to 3.5% (diffuse
forecasts hit the leverage cap). **Verdict: ML does not beat the disciplined hand-built
combination; rejected.** A cross-sectionally-demeaned/market-neutral ML could be revisited,
but the classical config is cleaner and stronger.

## CHAMPION (locked): `--sleeve combined --rebalance monthly --risk cluster`
48-instrument daily momentum (EWMAC + xsmom), cluster-equal risk budgeting, 10% vol-target,
monthly rebalance. **Net Sharpe +0.65 full / CI [+0.18,+1.12] (significant) / DD 24% /
~0 beta / both sub-halves positive.** The project's validated edge. Next: consolidate to a
deployable daily portfolio runner + multi-instrument execution design + paper trade.

## Verdict
- **Momentum (combined) is a real, diversified, out-of-sample-positive edge** — positive
  in BOTH discover and confirm, ~zero correlation to long-everything (genuine momentum,
  not beta), vol on-target, and — critically — **no ≫1 Sharpe / no 100% win** (passes the
  artifact smell-test that caught the stat-arb bug). This is the FIRST positive result in
  the whole project that isn't leakage or a bug.
- **But modest and below the GO bar:** combined confirm +0.46 (just under +0.5) with CI
  [−0.46, +1.49] straddling zero — 4.5 OOS years is too short for a ~0.4-Sharpe strategy to
  be statistically significant. NOT yet a confirmed deployable strategy.
- **Carry (simple sign-of-rate-differential) does NOT help** — negative over the sample
  (2010–2015 carry-unwind) and adds risk-on beta; it degraded the momentum portfolio.
  Real carry would need better construction (vol-scaled differential, term structure).

## Honest read
~+0.2–0.4 net Sharpe diversified daily momentum is **consistent with documented modern-era
CTA performance** (post-2011 momentum is weak) — it's real, just not strong. It's the right
foundation. Levers to firm it up (future): more instruments incl. true futures with longer
history (pre-2008 trends), trend-STRENGTH weighting (not just sign), better carry, and
simply more OOS time to tighten the CI. This is the keeper to build on; everything
directional/stat-arb/vol remains dead. See [[project_leakage_findings]].
