# CTA Champion — Deployment Plan (for future execution)

> Status: PLAN ONLY. Nothing here is built yet. Research is complete and the edge is
> validated (`data/CTA_FINDINGS.md`); this document is the roadmap to take it live.
> Reviewed by user → execute in stages, paper-trade before real capital.

## What we are deploying

The locked champion (validated, statistically significant — net Sharpe **+0.65** full 18yr,
95% CI [+0.18,+1.12], DD 24%, ~0 beta):

```
cta_backtest.py --sleeve combined --rebalance monthly --risk cluster
```
48-instrument daily momentum portfolio (EWMAC trend + cross-sectional momentum), risk
budgeted equally across asset classes, scaled to 10% portfolio vol, rebalanced monthly.

**The core gap to close:** this is a *daily, multi-instrument portfolio*; the existing live
bots (`pipeline_bot.py`) are *single-symbol, per-bar* traders. Deployment is a new runner,
not a tweak to the bots.

## Architecture

### 1. Universe → HFM symbol mapping (one-time, do FIRST — gates everything)
- New `scripts/cta_universe_map.py`: via the MT5 bridge, enumerate HFM symbols and map each
  of the 48 aliases → a tradeable HFM symbol (FX uses `.Z` suffix; indices/commodities/rates
  are broker CFDs with their own codes; some won't exist). Record per symbol: tradable flag,
  `volume_min`/`volume_step`/`volume_max`, `trade_contract_size`, `trade_tick_value`,
  digits, and swap (overnight) rates. Output `data/hfm_universe_map.json`.
- **Expected reality:** HFM likely offers FX majors/crosses + metals + a handful of indices,
  but maybe NOT all rates futures / ags / crypto. → we trade the available **liquid subset**.
- **MANDATORY re-validation:** re-run the champion backtest restricted to the tradeable
  subset. Fewer instruments / fewer asset classes weakens the cluster-diversification that
  drove +0.65 → **the deployable Sharpe will likely be lower**; measure it before committing.

### 2. Daily portfolio runner — `scripts/cta_live.py`
Runs once/day after a fixed cutoff (e.g. 21:05 UTC, post-NY close), via systemd timer/cron:
1. **Update data:** append latest daily bar per instrument (incremental Yahoo fetch) → panels.
2. **Compute target weights:** frozen champion pipeline (signals → `cluster_risk_weights` →
   `vol_target`) — reuse `src/cta/*` verbatim. Apply the monthly-rebalance gate (only change
   targets on month-end; hold between).
3. **Weights → target lots** (`src/cta/live_portfolio.py`, pure + unit-tested):
   `lots_i = (weight_i · target_gross · equity) / (price_i · contract_size_i)`, snapped to
   `volume_step`, clamped to `[volume_min, lot_cap]`; skip if below `volume_min`.
4. **Diff & execute:** compare target lots vs current net position per symbol; apply a
   **no-trade band** (skip diffs < buffer) to avoid churn; place/modify/close via
   `MT5Connector` to reach targets. Journal every action.
5. **Reconcile & log:** record target vs actual, gross/net exposure, realized vol.

### 3. Risk controls (hard, enforced in the runner)
- Portfolio 10% annualized vol target (already in pipeline) + a **max gross leverage cap**.
- Per-instrument **lot cap** (reuse `config.trading.max_lot`) + min-lot skip.
- **Daily loss / drawdown kill-switch** (flatten all + halt) — reuse BotBase 5% daily-loss rule.
- One runner instance, dedicated **magic number**; only manages its own positions.
- Overnight **swap awareness**: daily/monthly holds pay swap; flag instruments with large
  negative carry (swap can erode a 0.65-Sharpe edge).

## Validation / go-live protocol (staged — do not skip)
1. **Subset re-validation** (backtest on tradeable HFM universe) → confirm Sharpe still ≥ ~0.4
   with CI lower bound > 0. If it collapses, stop / reconsider universe.
2. **Dry-run paper** (`--dry-run`, ~4+ weeks): runner computes + journals target weights/orders
   but places nothing; verify targets match backtest, data pipeline is robust, no crashes.
3. **Demo-live on HFM** (the current demo account, dedicated magic): real order placement,
   small size; compare realized vs backtest expectation (expect a live-vs-backtest haircut).
4. **Go-live gate:** demo tracks backtest within reason over a meaningful window, costs/swaps
   confirmed acceptable, kill-switch tested. Only then real capital, starting small.

## Files (to build, in order)
| File | Purpose |
|------|---------|
| `scripts/cta_universe_map.py` | NEW — map 48 aliases → HFM tradable symbols + contract specs → `data/hfm_universe_map.json` |
| (re-validate) | re-run `cta_backtest.py` on the tradeable subset; record Sharpe/CI |
| `src/cta/live_portfolio.py` | NEW — pure weight→lots translation + position-diff logic (unit-tested) |
| `scripts/cta_live.py` | NEW — daily runner: data update → weights → diff → execute → journal; `--dry-run` |
| `tests/test_cta_live.py` | NEW — weight→lots rounding/caps, diff/no-trade-band, reconcile |
| systemd timer / cron | NEW — daily trigger post-NY-close |
| **Reuse as-is** | `src/cta/{signals,portfolio,panel,universe}.py`, `MT5Connector`, `TradeJournal`, lot cap |

## Key risks / open decisions (decide before/at build)
- **Tradeable-subset shrink** is the #1 risk — losing rates/ags/crypto diversification may cut
  the edge materially. Re-validation (step 1) is the gate.
- **Swap/overnight cost** on daily/monthly holds — not modeled in the backtest's per-trade cost;
  could be a real drag. Measure on demo.
- **Account granularity** — at small equity, 48 instruments at 10% vol may round many positions
  below `volume_min` → effectively a smaller, lumpier book. May need a larger account or a
  reduced universe.
- **Rebalance timing/slippage** — monthly rebalance executes a batch of orders at one time;
  spread/slippage on illiquid CFDs. No-trade band mitigates.
- **Data source split** — signals from Yahoo daily vs execution on HFM prices; small basis
  differences are fine for a daily strategy but confirm alignment.

**Bottom line:** the research edge is real and validated; the deployment risk is execution +
universe-availability, not the signal. Build in the order above, gate at subset re-validation,
and paper-trade before any real capital.
