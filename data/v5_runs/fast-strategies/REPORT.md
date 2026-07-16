# Fast (non-trend) strategy exploration — 2026-07-16

Brief: find faster, consistent, NON-trend edges across all instruments, single or
combined, incl. uncommon ideas. All backtests NET of spread. OOS = 2021+.

## What was tried (wide sweep)
| Idea | Result | Verdict |
|---|---|---|
| Overnight (close→open) vs intraday, per asset | best NDX/NIKKEI/GOLD/SILVER/NATGAS ~0.6-1.0 net | real but mixed |
| **Cross-sectional reversal (global indices)** | "Sharpe 3.5" | ❌ **FAKE** — non-synchronous close-times leak future info; US-only (synchronous) = −0.37 |
| Time-series 1-day reversal, per asset | best NDX 0.65, most ~0/neg | weak alone |
| Seasonality: turn-of-month | ~0.5 but = rest-of-month | not a distinct edge |
| Day-of-week | tiny, not tradeable net | no |
| **Ensemble of overnight + reversal** | **the finding** | ✅ see below |

## THE FINDING: diversified FAST ensemble (overnight + short-term reversal)
Same principle as the trend basket — combine many weak, ~uncorrelated (avg corr 0.02)
daily signals. Selected signals with positive 2017-2020 Sharpe, tested 2021+ OOS:
- **Combined fast ensemble: OOS Sharpe 1.30** (IS 1.32 → held OOS), consistent (2022 only −0.4).
- Decompose (OOS 2021+):
  - **Overnight-only: 1.19** — the driver, BUT hinges on trading the close→open gap (LIVE FILL CHECK needed; on 24h CFDs the gap may not be real/fillable).
  - **Reversal-only: 0.57** — modest but CLEANLY tradeable (close→close) and ~0 corr to trend (0.02).
- Correlation to trend basket: fast-all +0.43, reversal +0.02.
- **TREND + FAST (50/50): OOS Sharpe 1.44** vs trend alone 1.17 → genuine diversifying lift.

## Honest verdict
- No "get rich quick" fast scheme exists — standalone fast edges are weak; markets are efficient at short horizons.
- The DIVERSIFIED fast ensemble (~1.30 OOS, daily, non-trend, consistent through 2022) IS real and is a genuine complement to the trend book.
- Biggest caveat: the overnight component (the main driver) needs LIVE verification that the close→open gap is actually tradeable on these instruments. The reversal component (0.57) is safe but modest.
- Selection is a reasonable prior (overnight + reversal are documented effects) and held OOS — not pure data-mining, but not risk-free.

## Recommendation
Treat the fast ensemble as a DIVERSIFIER alongside the trend book (combined 1.44), NOT a
standalone replacement. Verify overnight fills on a demo before sizing the overnight sleeve;
the reversal sleeve can be deployed cleanly now. Scripts: data/v5_runs/fast-strategies/.
