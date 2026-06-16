# Edge Search — Results (sessions, timeframes, stat-arb)

## Extension 2026-06-16 — timeframe sweep + cross-pair stat-arb (all negative)

**Timeframe (deep 11yr, all 4 symbols, meta-labeling, real spread, bootstrap CIs):**
- M15: dead (best −0.8 to −1.3; confirmed on both 2.4yr and 11yr).
- H1: dead (best ~−0.6 to −0.9).
- H4: looked like the "breakeven boundary" (GBPUSD H4 xgb @0.65 = +0.55 full-period,
  54.9% win) — BUT discover/confirm killed it: discover +0.31 → **confirm −2.07**.
  Pure threshold-overfit. No real H4 edge.

**Cross-pair stat-arb (EURUSD-GBPUSD spread mean-reversion), `scripts/statarb_probe.py`:**
- First run showed +1.27 to +2.38 OOS Sharpe — but a robustness sweep exposed a P&L
  BUG: the spread "return" used the rolling hedge ratio βₜ at exit vs β_entry at entry,
  injecting fake P&L (tell-tale: a config with 100% win / 0 drawdown). 
- **Corrected** (proper 2-leg P&L, β fixed at entry): edge **vanishes** — EURUSD-GBPUSD
  H1 z=2.5 discover −0.40 / confirm −0.27, ~−2bp/trade (≈ cost drag), every (win,zwin)
  config flat-to-negative. **No stat-arb edge.**

**Volatility-timing (`scripts/vol_timing_probe.py`, compression→breakout AND
expansion→fade, reuses validated barrier P&L, OOS confirm 2022-26):** no significant
edge either — best USDJPY H4 +0.47 / +0.18 with CIs straddling zero; everything else
flat-to-negative, no config with CI lower bound > 0.

**FINAL VERDICT — edge search exhausted.** Three independent hypothesis classes, all
leak-free / OOS / bug-audited on 11yr EUR/GBP/JPY/XAU:
  1. Directional ML (M15/H1/H4, all-hours + session-gated) — DEAD / overfit.
  2. Cross-pair relative-value mean-reversion — NO EDGE (corrected).
  3. Volatility-timing (breakout + fade) — NO EDGE.
No retail-tradeable, out-of-sample edge survives. The prior live "+3 to +25" champions
were leakage (encoder + MTF lookahead). A real edge would require NEW information
(order-flow / COT / news-sentiment / alt-data), a different instrument universe, or a
fundamentally lower-frequency (carry/trend portfolio) approach — none available in the
current M15/H-bar FX-majors data. DO NOT deploy the invalidated models with real capital.

---

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
