# V5 FINDINGS — settled experiments (do not repeat)

Running ledger of XAUUSD research that has been **run, measured, and closed**. Each
entry: what was tried, the honest result, and the verdict. If an idea here is
marked DISPROVEN / DEAD, do not re-run it without a materially new angle.

Conventions: net Sharpe from daily-resampled equity × √252; eval window = 2017+;
the live HFM cent account's true gold cost is **$0.34 spread** (the `XAUUSD_*_long.csv`
spread column understates by 10× — use `--fixed-spread-usd 0.34`, not the raw column).

---

## Champions currently trusted (positive net edge)

| Strategy | Net edge | Status | Notes |
|---|---|---|---|
| Sharpe-1.6 drift portfolio | eval SR **1.59**, DSR 0.9998 | backtested, **not deployed** | champion recipe across BTC/indices/XAU/silver + LS diversifiers; the real upside |
| Long-only XAU champion | eval **0.99–1.04**, live ~0.97 | **LIVE** (acct 360542) | H4, vol-targeted EWMAC+breakout, conc^1.5, long-only. `data/v5_runs/xau-longonly-champion/` |
| LS ensemble | SR **0.81** | **LIVE** (acct 360541) | long-short trend diversifier |

Single-XAU ceiling ≈ 1.06; the jump to 1.6 is multi-asset diversification only.

### FundingPips challenge — DIVERSIFY the book (2026-07-15)
Single-XAU passes the FP 2-Step only ~61% once the daily-loss rule is measured on *floating* P&L (day_safety=1.5 proxy) — all risk in one asset breaches the 5% intraday line ~32% of the time. Diversifying the same champion recipe across the tradeable drift classes {eq_us SPX/NDX/DJI, eq_eu DAX/FTSE/STOXX, eq_ap NIKKEI/ASX, crypto BTC/ETH, xau H4-champ, metal SILVER}, equal-class-risk at 9% vol, lifts realistic pass to **~80% (daily-loss fails 32%→7%), median ~9.4mo**. (Full basket incl. rates/energy = 93% but those aren't offered.) Engine `scripts/v5_basket_challenge.py` (`--backtest` validates eval SR 1.27 / 80% pass; `--targets` emits live per-symbol leverage). Live MT5 order-wiring deferred until the FP account is purchased (needs real symbol names + terminal). Single-XAU `v5_xau_challenge.py` remains the fallback. Detail in CHALLENGEBOT.MD / memory.

### Basket improvement experiments (2026-07-16, `data/v5_runs/basket-ls-experiment/`)
- **Long/short per sleeve (like the cent `ls` bot) — DISPROVEN, badly:** eval SR 1.26→0.23, pass 92%→48%. Drift assets trend UP; shorting them bleeds. Crash-hedge (shorts only in deep downtrends) also worse (0.97). Do NOT add shorts.
- **Sharpe-weighting sleeves = lookahead illusion:** 1.46/96% in-sample, but walk-forward (past-data weights) = 1.25/88.7% ≈ equal-class. Equal-class weighting is already near-optimal; keep it.
- **GENUINE robust win: portfolio-level VOLATILITY TARGETING** (scale book to constant trailing vol, causal). Eval SR **1.26→1.39**, +dd-scaler **→1.43**; pass 91.9%→94.3% @7% (96.8% @6%), and FASTER. Recommended upgrade — scale per-symbol target leverages by a trailing-vol scalar on the book's own returns. Report: `basket-ls-experiment/REPORT.md`.

### FundingPips — FOCUSED book beats the 12-instrument basket (2026-07-18, `scripts/v5_xau_focus_challenge.py`)
Prompted by a live basket drawdown scare (a −1% wobble at 3 days = pure noise at 7% vol). Tested XAU-focused subsets vs the deployed 6-class basket on the exact FP 2-Step sim (realistic day_safety=1.5, vol-target ON, eval 2017+):

| Book | #a | SR17 | SR21 | pass% | fail-DD | median |
|---|---|---|---|---|---|---|
| 6-class basket (was deployed) | 12 | 1.43 | 1.22 | 94.3 | 5.7 | 12.3mo |
| XAU only | 1 | 1.12 | 1.20 | 94.2 | 5.8 | 17.0mo |
| **XAU+BTC+NDX equal ⅓** | **3** | **1.71** | **1.28** | **98.7** | **1.4** | **11.4mo** |
| XAU-tilt 50%+BTC/NDX 25% | 3 | 1.69 | 1.33 | 98.8 | 1.2 | 11.6mo |
| +SILVER (eq ¼) | 4 | 1.56 | 1.19 | 98.2 | 1.8 | 12.2mo |

**XAU+BTC+NDX (equal ⅓) dominates the basket on every FP metric with ¼ the symbols.** Correlations 2017+: XAU/BTC 0.08, XAU/NDX 0.04, BTC/NDX 0.12 (truly independent); the old basket diluted into correlated index clones (SPX≈NDX≈DJI 0.86) and gold-correlated SILVER (0.58). **DEPLOYED: engine `CLASSES` + `configs/v5_basket_challenge.json` switched to the 3-asset focused book (old 6-class kept as `BASKET_FULL` for revert).** Verified: `--backtest` SR 1.706 / pass 98.7%; `--targets` emits XAU/BTC/NDX only.

---

## DISPROVEN / DEAD (do not re-run)

### 1. M15 next-bar fade (mean-reversion) — DEAD net of spread
- **Signal:** fade extreme closes (close near bar LOW → long next bar; near HIGH → short). `scripts/v5_xau_fade_backtest.py`, live paper `scripts/v5_xau_fade_paper.py`.
- **Gross edge is real:** zero-cost M15 hours{8,20,22} FLAT ≈ Sharpe +1.9–2.3, win 54.8%, consistent every year.
- **Dies net of the true $0.34 spread:** win 36%, Sharpe **−6.8**, total −90% over 11.4 yrs. Deployed hours-restricted config included. The per-trade edge is a few cents; the spread is ~10× it.
- **Break-even spread ≈ $0.04–0.07** (need institutional/raw). HFM Zero/Raw ~$0.10–0.15 all-in is still above it.
- **Live paper +$10 over 2 weeks was a lucky window** — 11-yr net is −18%/yr.
- **Verdict:** not deployable as a spread-crossing taker at any timeframe (M15/M30/H1/H4 all tested). Only theoretical path = maker/limit fills that *earn* the spread (untested, likely infeasible retail).

### 2. Martingale / anti-martingale overlays on the fade — DEAD (ruin machine)
- `scripts/v5_xau_fade_martingale.py` — engines flat / double4 (classic capped-4 doubling) / recover4 (deficit-targeted, ladder capped at 4 trades, reset on recovery).
- **On the NET signal (36% win): recover4 BUSTS 100% of 500 block-bootstrap paths, ruin by ~trade 1,266.** double4 −61%, flat −23%.
- **On a GROSS positive-edge signal (54.8% win): recover4 = +152% vs flat +6% / double4 +13%, 0% bust, DD 11%** — proves the martingale is a *lever*, not an edge-creator.
- **Verdict:** decision variable is solely sign(net edge). Base fade is net-negative → no martingale rescues it. Do not deploy any progressive sizing on a negative-edge base.

### 3. Turning-point DETECTION (buy/sell reversal accuracy) — real skill, NOT tradeable
- `scripts/v5_xau_turning_points.py` (accuracy), `scripts/v5_xau_turning_ml.py` (precision push), `scripts/v5_xau_turning_trade.py` (trade sim), `scripts/v5_xau_champion_plus_detector.py` (combine). Ground truth = ZigZag swings; all detectors causal; scored precision/recall/F1 vs a random-coverage baseline.
- **Detection works above chance, robustly (no decay to 2024+):** best rule = Bollinger z-score, buy precision ~50–58% vs 33.5% random (lift 1.5–1.9×). **Bottoms detect materially better than tops** (V-panics vs rounded tops) — every config. Fade candle-shape = **zero** turning skill (lift 1.00×).
- **70% precision IS reachable OOS** with a HistGradientBoosting model — **BUY/bottoms tol=±3 → 71% precision @ 22% recall (2.1× random), 75%@10%, 80%@5%, PR-AUC 0.60.** SELL/tops cannot (70% only @ ~9% recall). Rules alone cap ~62%. Top features: 5-bar ATR-momentum, rank-in-window, z-score, RSI7. CAVEAT: judge by LIFT — widening tol inflates absolute precision by raising the random base.
- **But precision ≠ P&L.** Standalone bottom-long (~27 flags/mo, ~6–16 trades/mo): only profitable exit was hold-48-bars (+35%, SR 0.81) which merely harvests up-drift and **LOST to buy&hold gold (+113%)** in the same 2024–26 bull window. Short holds flat-to-negative.
- **Combining with the champion FAILS.** Overlaying the OOS detector on the H4 champion at equal exposure (pure timing test): champion alone eval SR **0.99** → every overlay worse (mildest bottom-boost 0.88, strong 0.66; trim-tops 0.88/0.65; both 0.40; detector-only-long −0.02). Turnover explodes 9.8→64→427 (prob wobble churns spread). Reason: champion is TREND-following (more exposure after breakouts), detector is MEAN-REVERSION (buy dips) — antagonistic; reversion-timing dilutes the trend edge and pays cost.
- **Anti-martingale ('untimartingale') sizing on the detector trades DOESN'T help** (`scripts/v5_xau_detector_antimartingale.py`). Best base (hold-48) flat Sharpe 0.84 → anti 0.30–0.53 at 2–3× the drawdown; hold-12 anti clearly negative (SR −0.5 to −0.8). Root cause: **win/loss lag-1 autocorrelation ≈ 0 (+0.05 to −0.03) — wins do not cluster.** Progressive sizing (anti *or* martingale) only adds value when outcomes are serially correlated; here trades are ~independent, so pressing winners just adds variance with no expected-return gain → Sharpe falls. Sizing cannot create edge on a streak-less, buy&hold-losing base.
- **Verdict:** detector is real science but not a tradeable edge and does not improve the trend bot. Leave the champion alone (it handles entry via vol-targeting + buffer). Turning-point timing and trend-following are orthogonal-to-antagonistic.

### 3b. Fast (non-trend) strategy exploration (2026-07-16, `data/v5_runs/fast-strategies/`)
Wide sweep for faster, consistent, non-trend edges, all net of spread, OOS=2021+.
- **Cross-sectional reversal on GLOBAL indices = FAKE** (Sharpe 3.5): non-synchronous close times (NIKKEI/ASX close hours before US) leak future info. US-only synchronous = −0.37. Classic artifact — reject any cross-sectional signal on assets with different close times.
- Standalone fast edges are WEAK net of cost: 1-day reversal best NDX 0.65; turn-of-month = rest-of-month (no distinct edge); day-of-week tiny.
- **THE FINDING — diversified FAST ENSEMBLE (overnight + short-term reversal, ~26 daily signals, avg corr 0.02):** select on 2017-2020, deploy 2021+ OOS → **OOS Sharpe 1.30** (IS 1.32, held), consistent (2022 only −0.4). Decompose OOS: **overnight-only 1.19** (the driver, but needs LIVE fill check — close→open gap may not be tradeable on 24h CFDs), **reversal-only 0.57** (cleanly tradeable close→close, ~0 corr to trend). **TREND+FAST 50/50 = 1.44** vs trend alone 1.17 → real diversifying lift. No get-rich-quick fast scheme; the ensemble is a genuine COMPLEMENT to trend, not a replacement.
- **VERIFIED DEAD on HFM's real tradeable instruments (2026-07-16, `scripts/v5_overnight_verify.py` + `v5_fast_verify_all.py` vs demo 57482374):** overnight ensemble −4.65 (backtest used CASH-index close 16:00→open 09:30 window; HFM's futures-CFDs break at a different time → no premium, slightly negative); intraday +1.22 but 0.97 corr to buy-hold = JUST DRIFT; reversal R1/R2/R5 = +0.13/+0.02/−0.35 = no edge (backtest 0.57 doesn't hold on real closes/spreads). **NOTHING in the fast family is deployable.** The backtest edges died on: bar-boundary shifts, wider real spreads, and drift-in-disguise. **RULE: always verify a fast edge on the BROKER's own symbols before building — cash/downloaded data lies about overnight windows and spreads.** Durable edge stays the TREND/DRIFT book. Report: `data/v5_runs/fast-strategies/REPORT.md`.

### 3c. Fast / intraday TREND runner (2026-07-17, `data/v5_runs/fast-trend/`)
Hunt for a *short-term trend* bot to complement the slow H4 champion (more
trades/day). Distinct from the dead fade/reversal work — this is trend, not
mean-reversion. Engines: `scripts/v5_xau_fast_trend_lab.py` (vectorized
vol-target sweep, buffered, combined-book vs champion) + `v5_xau_fast_trend_discrete.py`
(real lot/stop engine, monkeypatched fast champion signal).
- **Long-only beats LS at every speed** (kill-the-shorts again). Trend edge is
  gross-positive at ALL speeds (breakout gross SR ≤1.3) but **net Sharpe falls
  with turnover — spread is the tax.** No intraday-specific alpha: **session-ORB
  and intraday-momentum are gross-positive, net-DEAD** (same failure as fade).
- **Carver no-trade buffer cuts turnover 3–4× at ~no net-Sharpe loss** — key
  lever for spread-bound fast books; but slowing it down raises corr-to-champion
  to 0.73–0.80.
- **A fast sleeve does NOT improve the combined book:** champ-alone 1.28 →
  best 50/50 combo 1.23. On one asset a faster book is a correlated weaker clone.
- **Discrete reality kills the vectorized illusion:** net SR ~1.0 (continuous)
  → **0.4–0.5** (real 3×ATR stops whipsaw at 38% win, $0.34 spread, quantized
  lots); **negative 2017/2018/2021, edge only in the 2024–26 bull** = leveraged
  bull-beta, not robust.
- **The killer is the cent spread — account-type-fixable.** M30 fast champion,
  ~19 trades/mo: **$0.34 → 0.50 (dead); $0.12 raw/ECN → 0.85 (deployable);
  $0.02 → 1.07.** Halving spread ~doubles net Sharpe. M30 fast = best config.
- **LIVE VPS-VERIFIED (2026-07-17, read-only bridge probe):** live cent XAUUSDc
  spread measured **$0.36** (not $0.34 — slightly worse) → discrete net SR 0.49,
  DEAD confirmed. Raw-tier gold IS real at HFM (**XAUUSDb $0.10 FULL-tradable on
  demo**, discrete 0.89; XAUUSDr $0.16 disabled) but **NOT visible on the live
  cent account group** → no tight-spread path on the current live account. Gold
  swap −$0.72/oz/night long (unmodeled). Probes: `scripts/vps_spread_probe.py`,
  `vps_symbol_tradability.py`.
- **Verdict:** NOT deployable on the cent account (verified $0.36). Viable as an
  *activity* play only on a raw-tier HFM gold account (~$0.10–0.16, e.g. XAUUSDb),
  which the live cent group can't reach; and even there it doesn't lift a
  champion+fast portfolio (correlated). Real lever for more trades = cross-ASSET
  trend (BTC/NDX), not cross-speed on XAU. Report: `data/v5_runs/fast-trend/REPORT.md`.

### 3d. Holding-constraint stress: weekend / overnight bans (2026-07-20, `scripts/v5_holding_constraints.py`)
Funded (Master) accounts restrict holding — FundingPips banned weekend holds on 2-Step Flex Masters (29-Jan-2026). Measured the damage on GOLD+ETH+DJI (D1 proxies), Flex rules, eval 2017+:

| Scenario | Sharpe | CAGR | maxDD | pass% |
|---|---|---|---|---|
| hold through (Evaluation) | **+1.14** | 7.7% | −9.4% | **98.0** |
| NO weekend holding | +0.71 | 4.5% | −11.1% | 68.2 |
| NO overnight holding | **−1.03** | −4.1% | −33.6% | **0.4** |

- **Overnight ban is FATAL — strategy goes NEGATIVE.** For indices/gold essentially all long-run drift happens overnight (close→open); being flat every night hands away the edge while paying ~500 crossings/yr vs ~12. No dial fixes it, and an intraday replacement is already ruled out (§3c). **Do not run this book anywhere overnight holding is banned.**
- **Weekend ban is survivable AND adaptable: DROP THE CRYPTO SLEEVE.** GOLD+ETH+DJI 0.71/68.2% → **GOLD+DJI 1.08/79.1%** (GOLD+DJI+SPX 1.05/77.9%; NDX as third is worse on pass, 67%, too volatile for the daily line). Reason: crypto trades **24/7**, so a forced Friday-flat misses real moves, whereas gold/indices are closed anyway and only lose the Monday gap.
- **Action deferred to funding:** full runbook in `FUNDED-STAGE-PLAN.md` (config `classes` swap + a weekend auto-flat feature the executor does not yet have + ±5min news window). Nothing to change during Evaluation — weekend AND overnight holding are explicitly allowed there.

### 3e. Profit-take + dip re-entry overlay — DISPROVEN (2026-07-20, `scripts/v5_profit_take_overlay.py`)
Tested "bank the gains and buy back lower": when the book is up >= X% since entry, close everything; re-enter after price dips Y% from the exit level (or a max-wait timeout). Rationale was that floating profit counts against the firm's daily line, so realising it should protect the account. FTMO book (XAU+BTC+NDX), eval 2017+:

| Variant | Sharpe | CAGR | maxDD | pass% | median | %in-mkt |
|---|---|---|---|---|---|---|
| **baseline hold-through** | **+1.71** | **12.1** | −12.3 | **98.6** | **13.4** | 100 |
| take1.5/dip0.5/10d | 1.55 | 9.5 | −12.3 | 97.0 | 14.6 | 90.9 |
| take1.5/dip1.0/10d | 1.64 | 9.7 | −12.3 | 97.4 | 13.9 | 87.1 |
| take1.0/dip0.5/10d | 1.35 | 8.0 | −12.3 | 94.8 | 16.3 | 88.3 |
| take1.5/dip2.0/20d | 1.27 | 6.7 | −10.4 | 95.1 | 17.2 | 76.8 |

- **Every variant loses**, and performance tracks **%in-market** almost linearly — time out of the market is pure cost. maxDD does NOT improve (−12.3% throughout) except in the variant that gives up 45% of the CAGR.
- **The protective rationale is void at the live dial**: fail-daily is already **0.0%** at 7% vol, so there is no floating-profit risk to insure against — you truncate the right tail for nothing.
- **It does not rescue high-vol sprinting either** (FLEX rules, where 43–71% of failures ARE daily breaches): 10% dial 56.1% → 50.6/50.9/57.6; 14% 41.0 → 33.7/37.0/43.1; 18% 26.8 → 24.7/25.4/24.8. Tighter takes make **fail-daily WORSE** (43.0→47.9%) because a slower equity curve means more days exposed, i.e. more chances to catch a bad one. Only very loose settings (take 3%, 15d wait) are within noise of baseline — and those barely leave the market.
- **Verdict: do not add profit-taking or dip re-entry.** Trend books earn from a few large sustained winners; capping them removes the tail that funds all the small losses. Caveat: modelled at daily resolution, so a real intraday +1.5% trigger is approximated by exiting at that day's close — but the effect is large and monotone, so the conclusion holds.

### 3f. Multi-timeframe sweep M30/H1/H4 x speed, at REAL tight spreads (2026-07-20, `scripts/v5_multi_tf_trend.py`)
Re-opened the fast-trend question now that tight-spread gold is reachable (**FundingPips XAUUSDmicro $0.12** measured — the tier where §3c said fast trend *becomes* viable; FTMO XAUUSD $0.45; cent $0.36). Discrete engine, long-only champion recipe, eval 2017+.

Net Sharpe @ $0.12 (best case for fast configs):

| TF | ultra | vfast | fast | med | slow | trades/mo | maxDD |
|---|---|---|---|---|---|---|---|
| M30 | 0.68 | 0.88 | 0.93 | 0.87 | 0.87 | ~20 | −27 to −45% |
| H1 | 0.59 | 0.65 | 0.82 | 0.94 | 0.98 | ~11 | −21 to −33% |
| **H4** | 0.59 | 1.03 | 1.09 | **1.27** | 1.05 | ~3.3 | **−9 to −15%** |

- **"Sensitive to trend changes" is WORSE, universally.** `ultra` is the worst config at EVERY timeframe and EVERY spread (0.59–0.68). Faster reaction buys noise, not earlier trend detection.
- **H4 beats M30/H1 even at $0.12** (1.27 vs 0.93/0.98) with 1/3 the drawdown. So intraday is not merely cost-limited — the signal itself is worse. Tight spreads do NOT rescue intraday trend; §3c's "deployable at $0.12" was about a *single* fast config, and H4 still dominates it.
- **Spread sensitivity tracks turnover**: M30 (20 tr/mo) 0.93→0.47 as spread goes $0.12→$0.45; H4 (3.3 tr/mo) 1.09→1.01 (near-immune, consistent with the champion's known spread-invariance).
- **APPARENT WIN THAT FAILED VALIDATION — do not adopt.** `H4/med` scored 1.27 vs champion `H4/slow` 1.05, with better DD (−9.5%) and worst-year (−0.46). But (a) **split-sample: med 0.89 vs slow 0.96 in 2017-2020** — the entire edge is 2021+ (1.53 vs 1.13), i.e. regime-specific; (b) **sharp parameter peak**: trail 2.0/2.5/**3.0**/3.5/4.0 → 0.82/1.02/**1.27**/1.14/1.02, a robust edge has a flat surface; (c) selected from 45 swept configs (multiple testing). **Verdict: keep the champion speed set. The H4 slow champion remains the right choice.**

### 3g. FTMO vol-dial + 4th-sleeve instrument search (2026-07-20, `scripts/v5_instrument_search.py`, `v5_book_speed_sweep.py`)
**Book-level SPEED is exhausted — the curve is FLAT.** Running the whole 3-asset book 3x faster changes the median finish by 0.2mo (FTMO 13.2 vs 13.4mo, pass 98.2 vs 98.6). Flat (not peaked) = the live speed choice is robust, and no speed tuning will make the challenge finish sooner.

**FTMO VOL DIAL — 9% is the efficient point, NOT 10%** (FTMO rules, XAU+BTC+NDX):

| dial | pass% | fail-daily | median | note |
|---|---|---|---|---|
| 7% | 98.8 | 0.0 | 13.5mo | LIVE |
| 8% | 97.9 | 0.0 | 11.6mo | |
| **9%** | **96.4** | **0.0** | **10.3mo** | **−3.2mo for −2.4pp — best trade** |
| 10% | 87.4 | **8.1** | 8.6mo | daily limit starts binding — cliff |
| 12% | 65.6 | 29.9 | 6.2mo | |

The 5%-daily line is never approached until 10%; that is where fail-daily jumps 0→8.1% and pass drops 11pp for only 1.7 more months. **Do not go past 9%.**

**INSTRUMENT SEARCH — no 4th sleeve clears the bar.** Screened all 24 FTMO-tradeable instruments we hold D1 history for (FX excluded, dead post-2016) on standalone SR + correlation, then tested each as an equal-risk 4th sleeve. Screen correctly rejected the redundant ones: SPX (r=0.86 to NDX), ETH (0.64 to BTC), SILVER (0.58 to XAU), DJI (0.60 to NDX). Six candidates passed to stage 2:

| book | Sharpe | maxDD | pass% | median | 17-20 | 21-26 |
|---|---|---|---|---|---|---|
| **BASE XAU+BTC+NDX** | **1.71** | −12.3% | **98.6** | **13.4mo** | +2.28 | +1.28 |
| + BRENT (UKOIL) | 1.68 | **−9.7%** | **99.1** | 13.8mo | +2.14 | +1.34 |
| + NIKKEI (JP225) | **1.80** | −13.6% | 97.9 | **12.8mo** | +2.27 | +1.45 |
| + SOL (SOLUSD) | 1.74 | −12.9% | 98.2 | 13.1mo | +2.26 | +1.34 |
| + PALL / + DAX / + LTC | 1.62/1.52/1.41 | | 98.8/97.1/84.8 | slower | | |

- **Nothing satisfies "raise pass% AND cut median AND hold in both half-samples."** BRENT raises pass (+0.5pp) and cuts drawdown materially (−12.3%→−9.7%, and it is the LEAST correlated asset found: max r=0.05) but is 0.4mo slower. NIKKEI is fastest (−0.6mo) and highest Sharpe but costs 0.7pp pass and deepens DD. All differences are small/within noise.
- **Verdict: keep the 3-asset book.** It is already near-optimal; a 4th sleeve adds execution surface for no reliable gain. BRENT is the only one worth revisiting *if* the goal shifts from speed to drawdown reduction.
- **The ONLY reliable speed lever left is the vol dial (7%→9%, −3.2mo).**

### 4. Earlier disproven overlays (see memory for detail)
- **Per-trade probability sizing / meta-labeling** — fails twice; vol-targeting only cuts drawdown, adds no return.
- **Gold-silver spread** — corr 0.79 but z-spread edge is pre-2015-only, dead OOS 2017+.
- **Session / regime / carry / RL single-XAU overlays** — none beat the plain trend champion.

---

## Operational notes
- Live dual bots reconcile hourly via `xau-dual` user timer; the systemd service intermittently marks `failed` on a hung `winedevice.exe` at teardown (trading completes & exits 0 first) — cosmetic but leaves orphan wine procs; teardown fix pending.

_Last updated 2026-07-15._
