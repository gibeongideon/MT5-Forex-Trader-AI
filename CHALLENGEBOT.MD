# CHALLENGEBOT — FundingPips plan (FX majors / GOLD / SILVER)

Status: BOT BUILT + DRY-RUN VERIFIED (2026-07-14). Awaiting account purchase.
Bot: `scripts/v5_xau_challenge.py` (guard-first, 15-min cadence) + `configs/
v5_xau_challenge.json` + `src/v5/challenge_guards.py` (8 unit tests green) +
`deploy/xau-challenge.{service,timer}` (NOT enabled). Dry-runs verified:
normal pass (plan-only, in-sync) and synthetic guard trip (day −4.86% →
day_lock → flatten, no entries).

**Exact-bot simulation (discrete engine, c4_discrete_fpsim.py — these are
the numbers for the code that will actually run; realized vol is lower than
the continuous engine at the same risk_frac):**

| risk_frac | pass % | median | p75 | notes |
|-----------|--------|--------|-----|-------|
| 0.9%      | 92.6%  | 19.9 mo | 32.1 mo | safest |
| **1.1% (config default)** | **87.9%** | **15.2 mo** | **25.1 mo** | chosen |
| Stress @1.1%: weak-years 87.4%/21.6mo; edge−25% 82.5%/21.5mo; daily-limit
  failures 0.0% everywhere (worst historical day −3.0% + guard). |
Lab + all backtests: `data/v5_runs/challenge-lab/` (results.csv, campaign
scripts, final_book_stream.csv). Strategy base: XAU long-only champion
(`src/v5/xau_dual_signals.py::champion_signal`, research
`data/v5_runs/XAU_SHARPE1_RESEARCH.md`).

## 1. FundingPips ACTUAL rules (verified 2026-07-14, web)

**2-Step Standard (CHOSEN):** Phase 1 target 8%, Phase 2 target 5%; daily
loss 5%; max loss 10% STATIC on initial size; unlimited time; min 3 trading
days per phase; leverage up to 1:100.
**1-Step:** 10% target, 5% daily, 6% max static. Zero: trailing — avoid.

Rule mechanics that matter:
- **Daily loss** = 5% of max(day-start balance, day-start equity), INCLUDING
  floating P&L, resets 00:00 platform time (UTC+3).
- **Max loss** static, anchored to initial account size. Phase 2 = fresh
  account with full 10% budget.
- **Evaluation phase:** weekend holding ALLOWED, news trading ALLOWED, no
  consistency rule. (Funded/Master differs: weekend holds auto-closed
  Friday, 5-min news window rule, 35% consistency for on-demand payouts,
  3% single-trade profit cap — funded stage needs a re-plan, see §6.)
- **EAs:** full automation allowed with YOUR OWN EA + proof of ownership
  (our bot = own code, journals + git history = proof). Third-party EAs
  only as risk managers. Prohibited: HFT, latency/reverse arbitrage, gap
  trading, tick scalping, cross-account hedging, copy trading.
- Inactivity: trade at least every 30 days (champion trades ~3-4/mo — OK,
  but add a watchdog alert if flat > 21 days).

Sources: fundingpips.com/trading-objectives, help.fundingpips.com articles
(2-Step Standard, 1-Step, Understanding Trading Mechanics, News Trading &
Weekend Holding), proptradingvibes.com/blog/fundingpips-rules,
tradingfinder.com, tradetanto.com (cross-checked 2026-07-14).

## 2. Strategy research — what was tried (all net of costs, 2016+)

| Sleeve / idea                              | Sharpe | Verdict |
|--------------------------------------------|--------|---------|
| XAU champion H4 (deployed 360542 signal)    | **+1.02** | THE book |
| FX 7 majors D1 trend, fast speeds           | −0.36  | dead 2016+ |
| FX 7 majors D1 trend, slow speeds           | −0.39  | dead |
| FX per-pair scan (best: USDJPY)             | +0.17  | too weak |
| FX cross-sectional momentum (3m / 12m)      | −0.36 / −0.15 | dead |
| FX D1 mean-reversion (trend-gated)          | −0.15  | dead |
| FX carry (real 3m rates)                    | +0.06  | ~zero |
| SILVER D1 trend long/short                  | −0.03  | no |
| SILVER champion-recipe long-only            | +0.49 (0.48 on 2008+) | real but… |
| GOLD D1 long-only trend (reference)         | +0.65  | inferior to H4 champ |
| Combined XAU+FX books (any weight)          | 0.14–0.60 | FX poisons |
| XAU 85/silver 15, 80/20, 70/30              | ≈ XAU alone | non-additive (corr 0.5) |

**Conclusion: trade ONLY XAUUSD with the champion signal.** FX majors and
SILVER are permitted but not used: every FX stream tested is ≤ +0.17 (most
negative — the 2016-2026 FX regime killed retail trend/MR/carry), and
silver's 0.49 at 0.5 correlation leaves the pass curves statistically
identical while adding execution surface. Simplicity wins.

## 3. FundingPips 2-Step simulation (4,000 bootstrap paths/config)

Book = champion daily net stream normalized to 10% vol, scaled by k.
Exact rules: 8% → 5% targets, 10% static max loss, 5% daily, phase-2 reset.

| vol target | pass % | fail-daily | fail-maxloss | median | p75 |
|-----------|--------|------------|--------------|--------|------|
| 5%        | 98.4%  | 0.0%       | 1.1%         | 26.9 mo | 41.2 mo |
| 7%        | 94.8%  | 0.0%       | 5.2%         | 18.3 mo | 29.1 mo |
| 8%        | 91.4%  | 0.0%       | 8.6%         | 15.3 mo | 24.6 mo |
| **9%**    | **88.1%** | 0.0%    | 11.9%        | **13.0 mo** | **20.9 mo** |
| 10%       | 85.7%  | 0.0%       | 14.3%        | 11.0 mo | 18.1 mo |
| 12%       | 73.2%  | 9.9%       | 16.9%        | 7.9 mo  | 13.1 mo |

Stress at the chosen dial (9% vol):
- Weak-years-only stream (2016-2020, pre-bull): 84.8% pass, median 13.9 mo.
- Costs ×2: 87.1%, median 13.1 mo (turnover-light book — costs irrelevant).
- Edge haircut −25%: 76.9%, median 13.8 mo.
- **Intraday proxy (daily breach at 1.5× close-to-close move): pass drops
  to 61-75%, fail-daily jumps to 20-32% → an INTRADAY GUARD is mandatory,
  not optional (see §4.2). With the guard, daily breaches convert to small
  realized losses and the pass rate reverts toward the base row.**

**DECISION: 2-Step Standard, 100% XAUUSD champion signal, 9% ann vol
(risk_frac 0.9% per trade over the 3×ATR stop, conf scale on), hard
intraday guards. Expected: ~85-88% pass, median ~13 months, p75 ~21 months
— meets the <2-year requirement with margin at p75.**

## 4. Build spec — `scripts/v5_xau_challenge.py` (fork of v5_xau_dual.py)

1. Same engine/signal/reconcile as bot `champ` (magic new, e.g. 360550);
   `risk_frac` 0.01 → **0.009**; 3×ATR SL/trail; conf_risk_scale kept.
2. **Intraday daily guard (the critical addition):** timer every **15 min**
   (not hourly). Each pass: if equity ≤ day_anchor × (1 − 3.5%) → flatten,
   set no-trade flag until next 00:00 UTC+3. Day anchor = max(balance,
   equity) snapshot at first pass after 00:00 UTC+3, persisted to state
   JSON. Full H4 reconcile only on H4 boundaries; guard-only otherwise.
3. **Overall halt:** equity ≤ initial × (1 − 8%) → flatten + permanent HALT
   flag + notify (protects the account/retry fee before the firm's 10%).
4. **Phase tracker:** phase, phase-start balance, target in state JSON; on
   target: notify loudly, stop opening new positions (hold-to-close only)
   until manually moved to next phase credentials.
5. Guards unit-tested against synthetic equity paths BEFORE first --execute
   (guard must trip in the same pass that breaches its line).
6. Watchdog: alert if no trade in 21 days (inactivity rule) and if a pass
   fails 3× consecutively (bridge down).
7. Separate MT5 terminal profile + .env for the FundingPips account; never
   share the cent-account terminal. New user units xau-challenge.{service,
   timer} (OnCalendar *:00/15).

## 5. Residual risks (accepted, monitored)

- ~12-15% of paths fail even played correctly (drawdown before target) —
  budget one retry fee; halting at −8% preserves nothing financially but
  stops wasted months early.
- Weekend gap through the 3×ATR stop / daily guard: gold Monday gaps of
  ~5σ+ are rare but non-zero; evaluation allows weekend holds and the sim
  stream includes historical gaps close-to-close.
- Edge is half gold-drift: a hard bear regime pushes toward the weak-years
  numbers (still 85% / 14 mo) or worse; the −25% haircut row (77%) is the
  honest floor estimate.
- Funded (Master) stage has DIFFERENT rules (no weekend holds, news windows,
  3% single-trade cap, consistency for on-demand payouts) — the bot needs a
  funded-mode profile (auto-flatten Friday 20:45 server, skip entries in
  news windows, cap conf_risk high-bucket) BEFORE first funded trade. Plan
  that build after Phase 2 completes.

## 6. Execution sequence

1. Buy 2-Step Standard (size per capital comfort; rules identical across
   sizes; fee refunded after 4th payout).
2. Build + unit-test executor & guards (§4); 2-3 day dry-run (no --execute)
   on the challenge account.
3. Go live at 9% vol. Do NOT raise the dial after good months — the sim
   says the dial IS the strategy. Expect ~13 months median, up to ~21 (p75).
4. On Phase 1 pass: verify ≥3 trading days occurred (if not, make 0.01-lot
   token trades on separate days before final target push — target and
   min-days must BOTH be satisfied), repeat for Phase 2.
5. On funding: build funded-mode profile (§5 last bullet) before enabling.

## 7. Reproducibility

- Sleeve backtests + books: `data/v5_runs/challenge-lab/c1_sleeves.py`,
  `c2_fx_silver.py`; sim + stress: `c3_book_fpsim.py`; engine/simulator:
  `challenge_lab.py`. All runs appended to `results.csv`.
- Champion stream is byte-identical to the deployed 360542 signal module.


RULES for MASTER ACCOUNT ONCE FUNDED.NEED TO ADJUST BOT TO ADHERE TO THIS
https://help.fundingpips.com/hc/en-us/articles/34504137479441-News-Trading-Weekend-Holding