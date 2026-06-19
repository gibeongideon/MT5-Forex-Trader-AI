# GOLD Intraday Turning-Point + SL/TP — Multi-Timeframe Results (2026-06-19)

Trade the highest-probability turning points intraday: trend-scan turning-point label → P(up),
enter LONG on P≥thr / SHORT on P≤1−thr, exit via **ATR triple-barrier (SL=1×ATR, TP swept
1.5/2/3×ATR), force-close after the TF horizon**. Net of real spread + commission, R-units,
leak-free (encoder off, past-only features, forward labels as target only). Swept **4H / 2H / 1H /
30M / 15M × threshold {0.55–0.70} × R:R {1.5,2,3}**. Script: `scripts/gold_intraday_turning.py`.
GO = confirm Sharpe ≥+0.5, CI lower > 0, positive both discover halves, **positive net expectancy**.

## Verdict: NO-GO at every timeframe — monotonically worse as the bar shrinks

Best expectancy (avg R per trade, after cost) by TF — degrades cleanly with timeframe:

| TF | best avg R | best cell | win% | trades/yr | maxDD |
|----|-----------|-----------|------|-----------|-------|
| 4H | **+0.040** | 0.55 / 1:3 | 41% | 875 | 93% |
| 2H | +0.008 | 0.55 / 1:3 | 39% | 1,485 | 100% |
| 1H | +0.009 | 0.60 / 1:3 | 37% | 275 | 78% |
| 30M | **−0.084** | 0.55 / 1:3 | 34% | 1,940 | 100% |
| 15M | **−0.124** | 0.70 / 1:1.5 | 45% | 1,645 | 100% |

- **Expectancy falls and drawdowns deepen as the timeframe shrinks**: 4H barely-positive at best
  (+0.04R), 2H/1H ~breakeven, 30M/15M firmly negative (−0.08 to −0.28R). At low thresholds every
  sub-4H TF posts **100% drawdown — the account is wiped**. This is textbook cost domination: more
  bars → more trades → spread eats the ATR-sized edge.
- **Not one cell cleared the GO gate.** The eye-catching confirm Sharpes (4H 0.60/1:3 = +4.38;
  2H/1H positives) are **mirages**: they sit on tiny avg R (+0.01R), huge trade counts (annualization
  inflates Sharpe via √trades), and **negative discover** Sharpes — i.e. positive only in the
  2022–26 gold rally, negative 2016–21. Cardinal rule flags them; discover<0 disqualifies them.

## Conclusion
Gold has **no cost-surviving intraday turning-point edge** with SL/TP, on any of 4H/2H/1H/30M/15M.
4H is the least-bad (marginal +0.04R) but still fails the gate; everything below 1H is destroyed by
transaction cost. Consistent with every prior intraday finding (M15 next-bar was −29 Sharpe). The
per-trade SL/TP mechanic loses *faster* than the always-in flip version (`data/GOLD_TREND.md`),
which at least beat EWMAC. Intraday gold trading is not viable here; the only honest gold exposure
remains holding it / sizing it inside the diversified daily CTA basket.
