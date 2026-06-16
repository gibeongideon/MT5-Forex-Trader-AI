# Session / Hour-of-Day Pattern Study — Results

> 2026-06-15. Tests whether trading only during specific session overlaps reveals
> an edge the all-hours models lack. Scripts: `scripts/session_profile.py` (Phase A),
> `scripts/backtest_meta_labeling.py --hours/--session/--gate-mode` (Phase B).
> Data: deep Dukascopy `data/{SYM}_M15_long.csv` (2015→2026, UTC, real spread).

## Verdict: NO-GO — hour-gating does not reveal a tradeable edge

## Phase A — raw model-free profiling (11 years, EUR/JPY/GBP)

Every UTC hour and every named session, all three majors:
- **Autocorrelation (lag-1) ≈ 0** and **variance ratio ≈ 1.0** → random-walk-like; no
  persistent trend or mean-reversion structure to exploit.
- Simple momentum and mean-reversion rules (ATR triple-barrier exits) have **negative
  net-of-spread expectancy in all 24 hours** (win rate always < 50%).
- KIND label = "—" (neither TREND nor REVERT) for **every** window.
- The London/NY overlap (13–17 UTC) is the most *active* (range ~11–14 pips vs ~4–6 in
  Asian) and the *least bad*, but still negative. Activity ≠ edge.
- Spreads: EURUSD ~0.3p in active hours (widen to ~1.4p at 21:00 UTC rollover); GBPUSD
  wider (0.8–1.3p); USDJPY ~0.4–0.7p. Wider spreads make off-peak windows even less viable.

## Phase B — session-gated meta-labeling (deep data, real per-bar spread, bootstrap CIs)

133 expanding folds (180d/30d), rule primary, train-mode gate (meta trains + trades
in-window), threshold sweep:

| Window | Best Sharpe | 95% CI | Win% | Trades |
| ------ | ----------- | ------ | ---- | ------ |
| EURUSD London/NY 13–17 UTC | −0.63 @thr 0.60 | [−1.21, −0.09] | 49.5% | 1,967 |
| USDJPY Asian 0–6 UTC | −1.70 @thr 0.65 | [−2.33, −1.07] | 39.8% | 402 |

GO criterion (pre-registered): net Sharpe ≥ +0.5 with bootstrap CI lower bound > 0 on
the confirm slice. **Both FAIL** — the least-bad (EUR LonNY @0.60) has its *entire* CI
below zero, and higher thresholds collapse trade counts (valid folds drop to 24/12).

## Conclusion

Across 11 years, 3–4 majors, every session, both descriptive structure (Phase A) and a
properly-calibrated meta-model gated to the overlaps (Phase B), there is **no leak-free
tradeable edge at M15**. This is consistent with [[project_leakage_findings]]: the
original "champions" were leakage; honest performance is flat-to-negative, and
restricting to high-liquidity hours does not change that. GBPUSD profiled (same result);
XAUUSD pending its download.

Methodology guards applied: pre-registered windows as the only confirmatory test;
discover/confirm temporal split available (`--discover-confirm`); real per-bar spread
(short broker files have spread=0 and are timezone-corrupt — fail-closed in the harness);
bootstrap CIs; minimum-trade / sparse-fold reporting.
