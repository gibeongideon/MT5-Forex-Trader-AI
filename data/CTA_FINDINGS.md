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
