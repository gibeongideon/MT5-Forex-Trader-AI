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

## VERIFICATION on HFM real instruments (2026-07-16) — everything died
Tools: `scripts/v5_overnight_verify.py`, `scripts/v5_fast_verify_all.py` (read-only vs demo 57482374).
Tested every fast signal on HFM's ACTUAL tradeable symbols (US500.F, US100.F, US30.F,
GER40, JPN225, AUS200, XAUUSD, XAGUSD), net of REAL spread:

| Signal | Ensemble SR (real) | Verdict |
|---|---|---|
| Overnight | −4.65 | DEAD — cash-index window ≠ HFM futures-CFD bar boundary |
| Intraday | +1.22 (0.97 corr to buy-hold) | JUST DRIFT, not distinct |
| Reversal R1/R2/R5 | +0.13 / +0.02 / −0.35 | NO EDGE (backtest 0.57 didn't hold) |

**CONCLUSION: no fast/non-trend strategy is deployable.** The backtest edges died on
(1) bar-boundary shifts (broker open/close ≠ cash times), (2) wider real spreads,
(3) drift-in-disguise. The durable, verified edge remains the TREND/DRIFT book.

**RULE for the future:** any fast (daily/intraday) backtest edge MUST be re-verified on
the broker's own tradeable symbols before building. Downloaded/cash data misrepresents
overnight windows and understates spreads. Slow trend/drift is robust precisely because
frictions are rounding errors over multi-week holds; fast strategies are the opposite.
