# CTA Building-Block Ablation — Marginal Contribution (2026-06-18)

Each block toggled in isolation, locked champion config held fixed, on the 5-symbol basket
(GOLD,UST10Y,SPX,WTI,EURUSD) and the full 48-instrument universe. Net Sharpe (cost included),
discover/confirm + bootstrap CI + maxDD + turnover. Engine: `scripts/cta_backtest.py` (regime &
toggles added); driver: `scripts/cta_ablation.py`; raw table: `data/cta_ablation.csv`.
Regression guard held: `regime=none` reproduces the locked +0.746 exactly.

## Ranking by marginal contribution (biggest lever first)

| # | Block | Effect on FULL net Sharpe | Verdict |
|---|-------|---------------------------|---------|
| 1 | **Risk budgeting** | equal **+0.20** → diag **+0.45** → cluster **+0.65** | **Dominant lever (+0.45).** Cluster-aware (equal risk per asset class) is the single biggest contributor. |
| 2 | **Cross-sectional momentum** | ewmac-only +0.49 → combined **+0.65** | **+0.16 marginal.** A genuine, additive, diversifying return stream (xsmom-only also has the lowest DD, 18%). |
| 3 | **Time-series momentum (representation)** | tsmom +0.52 → ewmac +0.61 (basket); +0.37→+0.49 (full) | **Continuous (EWMAC) beats binary (tsmom) by ~0.1.** "slow" speeds = same Sharpe at **half the turnover** (724 vs 1207%). |
| 4 | **Volatility targeting** | basket +0.69→**+0.75** (DD 26%→19%); full +0.67→+0.65 | **Risk stabilizer.** Clear win on a *concentrated* book (raises Sharpe, cuts DD). ~neutral on a broadly diversified book (which self-stabilizes vol). |
| 5 | **Regime filtering** | basket: none +0.75, trend +0.74, vol +0.63, trend_vol +0.69; full: none +0.65, trend +0.57, vol +0.67, trend_vol +0.61 | **No Sharpe benefit — drop it.** At best trims basket DD slightly (18.8%→17.3% trend_vol) but lowers Sharpe and *raises* turnover; net-negative on the full universe. |

## Key takeaways
- **The edge is mostly RISK CONSTRUCTION, not signal cleverness.** Going equal-weight → cluster
  risk budgeting adds **+0.45 Sharpe** — far more than any signal choice. Diversification +
  correlation-aware sizing is the engine.
- **Two momentum styles stack:** time-series (EWMAC) + cross-sectional (xsmom) combined (+0.65)
  beats either alone — they're partly independent, so blending helps.
- **EWMAC > TSMOM**, and **slow speeds** give the same Sharpe at half the turnover → keep slow.
- **Vol targeting earns its place on the 5-basket** (the deployable book): +0.06 Sharpe and a
  7-pt DD reduction. On 48 instruments it's redundant with natural diversification.
- **Regime filtering (trend/vol gates) does not help here.** Trend-following signals already
  encode "don't fight the trend," so a 200d-SMA gate is double-counting; the vol gate de-risks
  into exactly the high-vol periods where trend P&L is made. Both cost turnover. **Not added to
  the champion.** (Built, tested lookahead-free in `src/cta/regime.py`, available behind `--regime`
  for future use, default off.)

## Conclusion
The validated 5-symbol champion (+0.746) is well-constructed: its Sharpe comes — in order — from
**cluster risk budgeting > cross-sectional momentum > continuous trend > vol targeting**, and
**regime filtering adds nothing**. No config change beats the locked champion; the ablation
confirms each retained block is pulling weight and the one new block (regime) is correctly omitted.
